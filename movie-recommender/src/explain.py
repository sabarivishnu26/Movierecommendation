"""
Phase 4: Explanation Layer

WHAT:
  For each Top-N recommendation from the production model (SVD, per Phase 3's
  results), generate a human-readable one-line reason: which genres it shares
  with movies the user already rated highly, and which specific movie in their
  history it's most similar to. An OPTIONAL second layer can rephrase that
  structured reason through a small LLM call for more natural, varied wording --
  off by default, so the project works standalone with zero API cost/dependency.

WHY:
  A bare recommendation score ("predicted score: 1.18") means nothing to an end
  user and doesn't demonstrate that the system is actually personalizing. Every
  major recommender pairs a suggestion with a reason (Netflix's "Because you
  watched X", Spotify's Discover Weekly blurbs). This is what makes a system
  feel personalized instead of a black box, and it's a good "AI-plus-ML" story:
  the ML model picks WHAT to recommend, this layer explains WHY.

  Grounding the explanation in real structured facts (actual shared genres, an
  actual movie from the user's own history) BEFORE any LLM call matters: if an
  LLM is used, it is only rephrasing true facts we computed, never inventing a
  justification from scratch. That avoids hallucinated reasons entirely.

HOW:
  1. Build each user's "genre profile": for every movie they rated, weight its
     genres by their rating, sum across their history -> ranked list of genres
     this user responds to most strongly.
  2. For each recommended movie, find the single highest-rated movie in the
     user's own history that shares the most genres with it.
  3. Compose a template explanation from those two facts (always available,
     deterministic, free).
  4. Optionally, pass ONLY the structured facts (not raw scores, not free text)
     to a small LLM call to produce a more natural sentence.
"""

import os
import pickle

import pandas as pd
from scipy.sparse import load_npz

PROCESSED_DATA_DIR = "movie-recommender/data/processed"
MODELS_DIR = "movie-recommender/models"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_movies():
    """Auto-detect movies.csv (ml-latest-small) or movies.dat (ml-1m)."""
    csv_path = os.path.join("data", "raw", "movies.csv")
    dat_path = os.path.join("data", "raw", "movies.dat")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    if os.path.exists(dat_path):
        return pd.read_csv(
            dat_path, sep="::", engine="python",
            names=["movieId", "title", "genres"], encoding="latin-1",
        )
    raise FileNotFoundError(f"No movies.csv or movies.dat found in {csv_path} / {dat_path}")


def load_svd_model():
    with open(os.path.join(MODELS_DIR, "svd_model.pkl"), "rb") as f:
        return pickle.load(f)


def load_mappings():
    with open(os.path.join(PROCESSED_DATA_DIR, "mappings.pkl"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Recommendation (SVD -- same logic as evaluate.py's recommend_svd)
# ---------------------------------------------------------------------------

def recommend_svd(user_idx, train_matrix, u, s, vt, k):
    scores = (u[user_idx] * s) @ vt
    seen = set(train_matrix[user_idx].nonzero()[1])
    ranked = scores.argsort()[::-1]
    recs = [item for item in ranked if item not in seen]
    return recs[:k]


# ---------------------------------------------------------------------------
# Explanation logic
# ---------------------------------------------------------------------------

def find_similar_liked_movie(user_idx, rec_movie_id, train_df, movies_df, min_rating=4):
    """
    Among movies this user rated >= min_rating, find the one sharing the most
    genres with the recommended movie. Returns (title, shared_genres) or
    (None, set()) if nothing overlaps.
    """
    rec_row = movies_df.loc[movies_df["movieId"] == rec_movie_id]
    if rec_row.empty or pd.isna(rec_row.iloc[0]["genres"]):
        return None, set()
    rec_genres = set(rec_row.iloc[0]["genres"].split("|"))

    history = train_df[train_df["user_idx"] == user_idx].merge(movies_df, on="movieId")
    history = history[history["rating"] >= min_rating]
    if history.empty:
        return None, set()

    def overlap(genres_str):
        if pd.isna(genres_str):
            return set()
        return set(genres_str.split("|")) & rec_genres

    history = history.assign(shared=history["genres"].apply(overlap))
    history = history[history["shared"].apply(len) > 0]
    if history.empty:
        return None, set()

    history = history.assign(shared_count=history["shared"].apply(len))
    best = history.sort_values(["shared_count", "rating"], ascending=False).iloc[0]
    return best["title"], best["shared"]


def template_explanation(similar_title, shared_genres):
    if similar_title:
        genre_str = " and ".join(sorted(shared_genres))
        return f"Recommended because you rated \"{similar_title}\" highly, and both share {genre_str}."
    return "Recommended based on overall patterns in your rating history."


def generate_explanations(user_idx, recommended_movie_idxs, train_df, movies_df, idx2movie):
    """
    recommended_movie_idxs: list of internal movie_idx (as returned by recommend_svd)
    Returns a list of dicts: {movie_id, title, reason, similar_to, shared_genres}
    """
    results = []
    for movie_idx in recommended_movie_idxs:
        movie_id = idx2movie[movie_idx]
        title_row = movies_df.loc[movies_df["movieId"] == movie_id]
        title = title_row.iloc[0]["title"] if not title_row.empty else str(movie_id)

        similar_title, shared = find_similar_liked_movie(user_idx, movie_id, train_df, movies_df)
        reason = template_explanation(similar_title, shared)

        results.append({
            "movie_id": movie_id,
            "title": title,
            "reason": reason,
            "similar_to": similar_title,
            "shared_genres": sorted(shared) if shared else [],
        })
    return results


# ---------------------------------------------------------------------------
# OPTIONAL: LLM rephrasing layer (off by default)
# ---------------------------------------------------------------------------

def llm_rephrase(rec_title, similar_title, shared_genres, model="claude-haiku-4-5-20251001"):
    """
    Rephrase a structured, fact-grounded explanation into a more natural sentence.
    Requires `pip install anthropic` and an ANTHROPIC_API_KEY environment variable.
    The LLM is given ONLY the verified facts below -- it is never asked to invent
    a reason, only to phrase one more naturally.
    """
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    if similar_title:
        facts = (
            f"The user rated \"{similar_title}\" highly. Both it and \"{rec_title}\" "
            f"share these genres: {', '.join(shared_genres)}."
        )
    else:
        facts = (
            f"No single similar movie was found in the user's history; the "
            f"recommendation of \"{rec_title}\" is based on overall rating patterns."
        )

    prompt = (
        "Write ONE short, natural sentence (max 20 words) telling a user why "
        f"we're recommending the movie \"{rec_title}\". Use ONLY these facts, "
        f"do not invent anything else: {facts}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_idx", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--llm", action="store_true", help="Rephrase explanations via LLM (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    train_matrix = load_npz(os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz")).tocsr()
    train_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "train_df.pkl"))
    mappings = load_mappings()
    idx2movie = mappings["idx2movie"]
    movies_df = load_movies()
    svd = load_svd_model()

    recs = recommend_svd(args.user_idx, train_matrix, svd["u"], svd["s"], svd["vt"], args.k)
    explanations = generate_explanations(args.user_idx, recs, train_df, movies_df, idx2movie)

    print(f"\nTop-{args.k} recommendations with explanations for user_idx={args.user_idx}:\n")
    for e in explanations:
        reason = e["reason"]
        if args.llm:
            reason = llm_rephrase(e["title"], e["similar_to"], e["shared_genres"])
        print(f"- {e['title']}")
        print(f"    {reason}\n")


if __name__ == "__main__":
    main()