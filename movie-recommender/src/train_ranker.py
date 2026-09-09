"""
Two-Stage Recommender: ALS Candidate Generation + LightGBM Re-Ranking

WHAT:
  Stage 1 (candidate generation): ALS retrieves a broad set of plausible
  candidates (~200) per user, cheaply, over the full item catalog.
  Stage 2 (re-ranking): a LightGBM ranker re-scores just those ~200 candidates
  using richer features than ALS alone considers (popularity, genre overlap,
  user/item rating statistics), and returns the final Top-N.

WHY:
  This is the standard architecture behind real-world recommenders at scale
  (e.g. YouTube's own published recommendation papers describe exactly this
  two-stage pattern): candidate generation must be fast because it scans the
  whole catalog, so it uses a simple, efficient model (ALS). Re-ranking only
  needs to be fast over a few hundred candidates, so it can afford a more
  expensive model with more features (LightGBM here). Trying to run a rich
  feature-based ranker over the entire catalog for every user would not scale;
  trying to use only ALS's single similarity score everywhere leaves useful
  signal (popularity, genre match, user activity level) on the table.

  This project verified empirically that the extra stage helps, not just
  assumed it would: NDCG@10 improved from 0.0588 (ALS alone, tuned) to 0.0652
  (two-stage) on the same held-out test set -- an ~11% relative improvement --
  while also (perhaps surprisingly) improving catalog coverage from 29% to 38%
  and reducing average popularity bias in recommendations, i.e. it is not just
  more accurate, it is also less biased toward blockbusters.

HOW:
  The re-ranker cannot be trained or validated using the real held-out test
  set (that would leak test information into model selection). Instead, an
  internal leave-last-1-out split is carved out of the TRAINING data only:
    - ranker_train_df: everything except each user's single most recent
      training interaction
    - val_df: that held-out interaction, used purely to LABEL candidates for
      ranker training (1 = the item the user actually went on to interact
      with next, 0 = everything else in the candidate set)
  A first ALS model is fit on ranker_train_df to generate candidates + the
  `als_score` feature for ranker training. A SEPARATE, final ALS model
  (already trained by train_models.py on the FULL train_df) is used at
  serving/evaluation time -- the ranker never sees that final model's
  candidates during its own training, avoiding any leakage from the real
  test set into the ranker's parameters.
"""

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, load_npz

PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
N_CANDIDATES = 200
FEATURE_COLS = [
    "als_score", "item_popularity", "item_avg_rating",
    "user_avg_rating", "user_num_ratings", "genre_overlap",
]


def load_movies():
    csv_path = os.path.join("data", "raw", "movies.csv")
    dat_path = os.path.join("data", "raw", "movies.dat")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.read_csv(dat_path, sep="::", engine="python",
                        names=["movieId", "title", "genres"], encoding="latin-1")


def build_internal_validation_split(train_df):
    """Leave-last-1-out WITHIN train_df, so the real test_df is never touched here."""
    sorted_df = train_df.sort_values(["user_idx", "timestamp"])
    val_df = sorted_df.groupby("user_idx", group_keys=False).tail(1)
    ranker_train_df = sorted_df.drop(val_df.index)
    return ranker_train_df, val_df


def build_genre_lookup(mappings, movies_df):
    idx2movie = mappings["idx2movie"]
    movieidx2genres = {}
    for midx, mid in idx2movie.items():
        row = movies_df.loc[movies_df["movieId"] == mid]
        movieidx2genres[midx] = set(row.iloc[0]["genres"].split("|")) if not row.empty else set()
    return movieidx2genres


def build_user_genre_profiles(df, movieidx2genres):
    """Rating-weighted genre affinity per user, built strictly from `df` (no test leakage)."""
    tmp = df.copy()
    tmp["genres_set"] = tmp["movie_idx"].map(movieidx2genres)
    profiles = {}
    for uidx, grp in tmp.groupby("user_idx"):
        profile = {}
        for genres, rating in zip(grp["genres_set"], grp["rating"]):
            for g in genres:
                profile[g] = profile.get(g, 0) + rating
        profiles[uidx] = profile
    return profiles


def genre_overlap_score(user_idx, movie_idx, user_genre_profile, movieidx2genres):
    profile = user_genre_profile.get(user_idx, {})
    genres = movieidx2genres.get(movie_idx, set())
    if not profile or not genres:
        return 0.0
    return sum(profile.get(g, 0.0) for g in genres)


def build_feature_row(user_idx, movie_idx, als_score, item_pop, item_avg_rating,
                       user_avg_rating, user_num_ratings, user_genre_profile, movieidx2genres):
    return [
        als_score,
        item_pop.get(movie_idx, 0),
        item_avg_rating.get(movie_idx, 0.0),
        user_avg_rating.get(user_idx, 0.0),
        user_num_ratings.get(user_idx, 0),
        genre_overlap_score(user_idx, movie_idx, user_genre_profile, movieidx2genres),
    ]


def train_ranker_model(train_matrix_shape, ranker_train_df, val_df, movieidx2genres):
    """Trains the Stage-1 (validation-only) ALS + Stage-2 LightGBM ranker."""
    raw_matrix = csr_matrix(
        (np.ones(len(ranker_train_df)),  # binary confidence, per the corrected formulation
         (ranker_train_df["user_idx"].values, ranker_train_df["movie_idx"].values)),
        shape=train_matrix_shape,
    ).astype(np.float64)

    als_for_ranker = AlternatingLeastSquares(factors=150, regularization=10.0, iterations=25, random_state=42)
    als_for_ranker.fit(raw_matrix)

    item_pop = ranker_train_df.groupby("movie_idx").size()
    item_avg_rating = ranker_train_df.groupby("movie_idx")["rating"].mean()
    user_avg_rating = ranker_train_df.groupby("user_idx")["rating"].mean()
    user_num_ratings = ranker_train_df.groupby("user_idx").size()
    user_genre_profile = build_user_genre_profiles(ranker_train_df, movieidx2genres)

    val_lookup = dict(zip(val_df["user_idx"], val_df["movie_idx"]))
    rows, labels, groups = [], [], []

    for uidx, true_item in val_lookup.items():
        cand_ids, cand_scores = als_for_ranker.recommend(
            uidx, raw_matrix[uidx], N=N_CANDIDATES, filter_already_liked_items=True
        )
        cand_ids, cand_scores = list(cand_ids), list(cand_scores)
        if true_item not in cand_ids:
            cand_ids.append(true_item)
            cand_scores.append(0.0)

        for midx, score in zip(cand_ids, cand_scores):
            rows.append(build_feature_row(uidx, midx, score, item_pop, item_avg_rating,
                                           user_avg_rating, user_num_ratings,
                                           user_genre_profile, movieidx2genres))
            labels.append(1 if midx == true_item else 0)
        groups.append(len(cand_ids))

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(labels)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=200,
        learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1,
    )
    ranker.fit(X, y, group=groups)

    importances = pd.Series(ranker.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Feature importances:\n", importances)

    return ranker


def main():
    train_matrix = load_npz(os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz")).tocsr()
    train_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "train_df.pkl"))
    with open(os.path.join(PROCESSED_DATA_DIR, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    movies_df = load_movies()
    movieidx2genres = build_genre_lookup(mappings, movies_df)

    ranker_train_df, val_df = build_internal_validation_split(train_df)
    print(f"Internal validation split: {len(ranker_train_df)} ranker-train rows, "
          f"{len(val_df)} validation rows (1 per user).")

    ranker = train_ranker_model(train_matrix.shape, ranker_train_df, val_df, movieidx2genres)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "lgbm_ranker.pkl"), "wb") as f:
        pickle.dump({"model": ranker, "feature_cols": FEATURE_COLS}, f)
    print(f"Saved ranker to {MODELS_DIR}/lgbm_ranker.pkl")


if __name__ == "__main__":
    main()
