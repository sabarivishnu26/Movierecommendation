"""
Phase 1: Data Preprocessing — Explicit Ratings -> Implicit Feedback

WHAT this does:
  1. Loads raw MovieLens ratings (userId, movieId, rating, timestamp)
  2. Encodes raw user/movie IDs into contiguous 0-indexed indices (needed for matrix ops)
  3. Converts explicit ratings into implicit "confidence" scores
  4. Splits the data using a TIME-BASED leave-last-N-out strategy (not random split)
  5. Builds a sparse user-item confidence matrix
  6. Saves everything the modeling phase (Phase 2) will need

WHY implicit feedback instead of raw ratings:
  Real-world recommenders rarely see "explicit" preference (1-5 stars). They see
  interactions: did the user watch/click/stream it or not? We simulate that here by
  treating every MovieLens rating as a positive interaction, and using the star value
  only as a signal of HOW confident we are that the interaction reflects genuine interest
  (a 5-star rating is stronger evidence of interest than a 1-star rating, but both mean
  "the user chose to watch and engage with this movie").

WHY time-based split instead of random split:
  A random split lets the model "see the future" — training on interactions that
  happened after some of the test interactions. A time-based split (holding out each
  user's most RECENT interactions) mirrors the real task: "given what a user has done
  so far, predict what they'll do next." This is standard practice in recommender
  systems research and evaluation.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

RAW_DATA_DIR = "movie-recommender/data/raw"
PROCESSED_DATA_DIR = "movie-recommender/data/processed"

# Hu, Koren & Volinsky (2008) style confidence weighting: confidence = 1 + alpha * rating.
# alpha controls how much we trust higher ratings as a stronger interaction signal.
# We use a smaller alpha (15) than the paper's default (40) because MovieLens ratings
# are explicit (1-5) rather than raw implicit counts (e.g. play counts), so they already
# carry more signal per unit than something like "number of times a song was played."
ALPHA = 15

# How many of each user's most recent interactions to hold out for testing.
N_HOLDOUT = 2

# Minimum interactions a user must have to be included at all. Users with too few
# ratings can't be meaningfully split into train/test, and including them just adds
# noise to evaluation.
MIN_INTERACTIONS = 5


def load_ratings(path=None):
    """Load raw ratings.csv (userId, movieId, rating, timestamp)."""
    path = path or os.path.join(RAW_DATA_DIR, "ratings.csv")
    ratings = pd.read_csv(path)
    required_cols = {"userId", "movieId", "rating", "timestamp"}
    missing = required_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv is missing expected columns: {missing}")
    return ratings


def filter_sparse_users(ratings, min_interactions=MIN_INTERACTIONS):
    """Drop users with fewer than `min_interactions` ratings."""
    counts = ratings.groupby("userId")["movieId"].transform("count")
    return ratings[counts >= min_interactions].copy()


def encode_ids(ratings):
    """
    Map raw MovieLens userId/movieId (arbitrary, non-contiguous integers) to
    contiguous 0-indexed positions. Sparse matrices and ALS/BPR libraries expect
    row/column indices to run 0..n-1 with no gaps.
    """
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}

    ratings = ratings.copy()
    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["movie_idx"] = ratings["movieId"].map(movie2idx)

    return ratings, user2idx, movie2idx, idx2user, idx2movie


def add_confidence(ratings, alpha=ALPHA):
    """
    Turn the explicit 1-5 rating into an implicit-feedback confidence score.
    Every interaction (regardless of star value) is treated as a positive signal;
    `confidence` just reflects HOW positive we believe it to be.
    """
    ratings = ratings.copy()
    ratings["confidence"] = 1 + alpha * ratings["rating"]
    return ratings


def time_based_split(ratings, n_holdout=N_HOLDOUT):
    """
    For each user, sort their interactions by timestamp and hold out the last
    `n_holdout` as the test set. Everything earlier becomes training data.
    Users end up with (their_total_interactions - n_holdout) training rows.
    """
    ratings = ratings.sort_values(["user_idx", "timestamp"])
    test_df = ratings.groupby("user_idx", group_keys=False).tail(n_holdout)
    train_df = ratings.drop(test_df.index)
    return train_df, test_df


def build_sparse_matrix(df, n_users, n_items, value_col="confidence"):
    """
    Build a (n_users x n_items) sparse matrix from a long-format interactions
    dataframe. This is the matrix ALS/BPR will factorize in Phase 2.
    """
    mat = csr_matrix(
        (df[value_col].values, (df["user_idx"].values, df["movie_idx"].values)),
        shape=(n_users, n_items),
    )
    return mat


def run_pipeline(raw_path=None):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    ratings = load_ratings(raw_path)
    ratings = filter_sparse_users(ratings, MIN_INTERACTIONS)
    ratings, user2idx, movie2idx, idx2user, idx2movie = encode_ids(ratings)
    ratings = add_confidence(ratings, ALPHA)

    train_df, test_df = time_based_split(ratings, N_HOLDOUT)

    n_users = len(user2idx)
    n_items = len(movie2idx)
    train_matrix = build_sparse_matrix(train_df, n_users, n_items)

    # --- Save everything Phase 2 (modeling) will need ---
    save_npz(os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz"), train_matrix)
    train_df.to_pickle(os.path.join(PROCESSED_DATA_DIR, "train_df.pkl"))
    test_df.to_pickle(os.path.join(PROCESSED_DATA_DIR, "test_df.pkl"))

    with open(os.path.join(PROCESSED_DATA_DIR, "mappings.pkl"), "wb") as f:
        pickle.dump(
            {
                "user2idx": user2idx,
                "movie2idx": movie2idx,
                "idx2user": idx2user,
                "idx2movie": idx2movie,
            },
            f,
        )

    # --- Sanity report ---
    print("=== Phase 1: Data Preprocessing Summary ===")
    print(f"Users kept (>= {MIN_INTERACTIONS} interactions): {n_users}")
    print(f"Unique movies: {n_items}")
    print(f"Train interactions: {len(train_df)}")
    print(f"Test interactions:  {len(test_df)}")
    print(f"Train matrix sparsity: {1 - train_matrix.nnz / (n_users * n_items):.4%}")
    print(f"Saved to: {PROCESSED_DATA_DIR}/")

    return train_matrix, train_df, test_df, user2idx, movie2idx, idx2user, idx2movie


if __name__ == "__main__":
    run_pipeline()