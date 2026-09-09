"""
Production API layer for the movie recommender.

WHAT: A FastAPI service exposing the two-stage recommender (ALS candidate
generation + LightGBM re-ranking) as a versioned HTTP endpoint, with request
validation, structured errors, and startup-time model loading.

WHY: A Streamlit demo is fine for a human clicking through a browser. It is not
how a recommender is actually consumed in production -- other services call it
over an API. Building this layer demonstrates the engineering half of "ML
engineer" (not just training models, but serving them reliably): input
validation, graceful error handling for unknown users, and a clean contract
(OpenAPI docs auto-generated at /docs) that another team could integrate against
without reading your training code.

HOW: All heavy artifacts (ALS model, LightGBM ranker, feature tables) are loaded
ONCE at process startup via FastAPI's lifespan handler, not per-request. Each
request does only cheap work: one ALS candidate lookup + one small LightGBM
batch prediction.

RUN:
    uvicorn api.main:app --reload --port 8000
    open http://localhost:8000/docs for interactive API docs
"""

import os
import pickle
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import load_npz

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

STATE = {}


class Recommendation(BaseModel):
    movie_id: int
    title: str
    reason: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    model_version: str
    recommendations: List[Recommendation]


def load_artifacts():
    train_matrix = load_npz(os.path.join(DATA_DIR, "processed", "train_matrix.npz")).tocsr()
    train_df = pd.read_pickle(os.path.join(DATA_DIR, "processed", "train_df.pkl"))
    with open(os.path.join(DATA_DIR, "processed", "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "als_final.pkl"), "rb") as f:
        als = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "lgbm_ranker.pkl"), "rb") as f:
        ranker_bundle = pickle.load(f)

    movies_path_csv = os.path.join(DATA_DIR, "raw", "movies.csv")
    movies_path_dat = os.path.join(DATA_DIR, "raw", "movies.dat")
    if os.path.exists(movies_path_csv):
        movies_df = pd.read_csv(movies_path_csv)
    else:
        movies_df = pd.read_csv(movies_path_dat, sep="::", engine="python",
                                 names=["movieId", "title", "genres"], encoding="latin-1")

    item_pop = train_df.groupby("movie_idx").size()
    item_avg_rating = train_df.groupby("movie_idx")["rating"].mean()
    user_avg_rating = train_df.groupby("user_idx")["rating"].mean()
    user_num_ratings = train_df.groupby("user_idx").size()

    idx2movie = mappings["idx2movie"]
    movieidx2genres = {}
    for midx, mid in idx2movie.items():
        row = movies_df.loc[movies_df["movieId"] == mid]
        movieidx2genres[midx] = set(row.iloc[0]["genres"].split("|")) if not row.empty else set()

    user_genre_profile = {}
    tmp = train_df.copy()
    tmp["genres_set"] = tmp["movie_idx"].map(movieidx2genres)
    for uidx, grp in tmp.groupby("user_idx"):
        profile = {}
        for genres, rating in zip(grp["genres_set"], grp["rating"]):
            for g in genres:
                profile[g] = profile.get(g, 0) + rating
        user_genre_profile[uidx] = profile

    STATE.update({
        "train_matrix": train_matrix,
        "train_df": train_df,
        "mappings": mappings,
        "als": als,
        "ranker": ranker_bundle["model"],
        "feature_cols": ranker_bundle["feature_cols"],
        "movies_df": movies_df,
        "item_pop": item_pop,
        "item_avg_rating": item_avg_rating,
        "user_avg_rating": user_avg_rating,
        "user_num_ratings": user_num_ratings,
        "movieidx2genres": movieidx2genres,
        "user_genre_profile": user_genre_profile,
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="Movie Recommender API",
    description="Two-stage recommender: ALS candidate generation + LightGBM re-ranking",
    version="2.0.0",
    lifespan=lifespan,
)


def genre_overlap_score(user_idx, movie_idx):
    profile = STATE["user_genre_profile"].get(user_idx, {})
    genres = STATE["movieidx2genres"].get(movie_idx, set())
    if not profile or not genres:
        return 0.0
    return sum(profile.get(g, 0.0) for g in genres)


def find_similar_liked_movie(user_idx, movie_id, min_rating=4):
    movies_df = STATE["movies_df"]
    train_df = STATE["train_df"]
    rec_row = movies_df.loc[movies_df["movieId"] == movie_id]
    if rec_row.empty or pd.isna(rec_row.iloc[0]["genres"]):
        return None, set()
    rec_genres = set(rec_row.iloc[0]["genres"].split("|"))
    history = train_df[(train_df["user_idx"] == user_idx) & (train_df["rating"] >= min_rating)]
    history = history.merge(movies_df, on="movieId")
    if history.empty:
        return None, set()
    history = history.assign(shared=history["genres"].apply(
        lambda g: set(g.split("|")) & rec_genres if pd.notna(g) else set()
    ))
    history = history[history["shared"].apply(len) > 0]
    if history.empty:
        return None, set()
    history = history.assign(shared_count=history["shared"].apply(len))
    best = history.sort_values(["shared_count", "rating"], ascending=False).iloc[0]
    return best["title"], best["shared"]


def explain(user_idx, movie_id):
    similar_title, shared = find_similar_liked_movie(user_idx, movie_id)
    if similar_title:
        return f"Because you rated \"{similar_title}\" highly, and both share {', '.join(sorted(shared))}."
    return "Based on overall patterns in your rating history."


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.version}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = 10, candidates: int = 200):
    user2idx = STATE["mappings"]["user2idx"]
    idx2movie = STATE["mappings"]["idx2movie"]
    movies_df = STATE["movies_df"]

    if user_id not in user2idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id {user_id}. "
                             "This user has no rating history in the training set "
                             "(cold-start users are not yet supported by this endpoint).")
    if not (1 <= k <= 50):
        raise HTTPException(status_code=400, detail="k must be between 1 and 50.")

    user_idx = user2idx[user_id]
    train_matrix = STATE["train_matrix"]
    als = STATE["als"]
    ranker = STATE["ranker"]
    feature_cols = STATE["feature_cols"]

    cand_ids, cand_scores = als.recommend(
        user_idx, train_matrix[user_idx], N=candidates, filter_already_liked_items=True
    )
    feats = pd.DataFrame(
        [[s, STATE["item_pop"].get(m, 0), STATE["item_avg_rating"].get(m, 0.0),
          STATE["user_avg_rating"].get(user_idx, 0.0), STATE["user_num_ratings"].get(user_idx, 0),
          genre_overlap_score(user_idx, m)] for m, s in zip(cand_ids, cand_scores)],
        columns=feature_cols,
    )
    rerank_scores = ranker.predict(feats)
    order = np.argsort(-rerank_scores)[:k]

    results = []
    for i in order:
        movie_idx = cand_ids[i]
        movie_id = idx2movie[movie_idx]
        title_row = movies_df.loc[movies_df["movieId"] == movie_id]
        title = title_row.iloc[0]["title"] if not title_row.empty else str(movie_id)
        results.append(Recommendation(
            movie_id=int(movie_id),
            title=title,
            reason=explain(user_idx, movie_id),
            score=float(rerank_scores[i]),
        ))

    return RecommendationResponse(user_id=user_id, model_version=app.version, recommendations=results)
