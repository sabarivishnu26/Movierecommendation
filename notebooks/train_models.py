"""
Phase 2: Core ML — ALS and BPR for Implicit Feedback

WHAT:
  Train two ranking-oriented matrix factorization models on the implicit
  confidence matrix produced in Phase 1:
    - ALS  (Alternating Least Squares)          -> implicit.als.AlternatingLeastSquares
    - BPR  (Bayesian Personalized Ranking)       -> implicit.bpr.BayesianPersonalizedRanking

WHY:
  ALS:
    Standard SVD treats missing matrix entries as "unknown" and only reconstructs
    observed values. For implicit feedback that's wrong -- a missing entry usually
    means "we don't know if they'd like it," not "neutral." ALS (Hu, Koren &
    Volinsky, 2008) treats every (user, item) pair as either positive (observed,
    weighted by confidence) or negative (unobserved, weak negative signal), and
    factorizes with that weighting. This is the standard approach for
    implicit-feedback CF at industry scale.

  BPR:
    ALS still optimizes a reconstruction-style loss. BPR instead optimizes ranking
    directly: for each observed interaction it samples an item the user did NOT
    interact with, and updates the model so the observed item scores higher than
    the sampled negative. This "learning to rank" objective is architecturally
    closer to what we'll evaluate with (Precision@K, NDCG@K) than reconstruction
    loss is -- which is exactly the comparison Phase 3 will make explicit.

  Training both lets Phase 3 empirically answer: does directly optimizing for
  ranking (BPR) actually beat a reconstruction-based approach (ALS) on ranking
  metrics? That comparison is the project's central ML insight.

HOW:
  implicit's models expect a (users x items) CSR matrix whose nonzero values ARE
  the confidence scores -- exactly what Phase 1 already built and saved.
    factors        = size of the latent taste space (hidden dimensions like
                      "affinity for sci-fi", learned automatically, not labeled)
    regularization  = penalizes large factor values to reduce overfitting
    iterations      = number of training passes
"""

import argparse
import os
import pickle
import time

import pandas as pd
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

PROCESSED_DATA_DIR = "movie-recommender/data/processed"
MODELS_DIR = "movie-recommender/models"


def load_train_matrix(path=None):
    path = path or os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz")
    return load_npz(path)

def load_mappings(path=None):
    path = path or os.path.join(PROCESSED_DATA_DIR, "mappings.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def train_als(train_matrix, factors=50, regularization=0.1, iterations=20, random_state=42):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    start = time.time()
    model.fit(train_matrix.tocsr())
    print(f"[ALS] trained in {time.time() - start:.1f}s "
          f"(factors={factors}, reg={regularization}, iters={iterations})")
    return model


def train_bpr(train_matrix, factors=50, regularization=0.01, iterations=100,
              learning_rate=0.05, random_state=42):
    model = BayesianPersonalizedRanking(
        factors=factors,
        regularization=regularization,
        learning_rate=learning_rate,
        iterations=iterations,
        random_state=random_state,
    )
    start = time.time()
    model.fit(train_matrix.tocsr())
    print(f"[BPR] trained in {time.time() - start:.1f}s "
          f"(factors={factors}, reg={regularization}, iters={iterations})")
    return model


def sanity_check(model, train_matrix, idx2movie, movies_df=None, user_idx=0, n=5, label=""):
    """
    Print Top-N recommendations for one user as a manual sanity check.
    filter_already_liked_items=True excludes movies already in their training
    interactions, so we're only looking at genuinely "new" recommendations.
    """
    item_ids, scores = model.recommend(
        user_idx, train_matrix[user_idx], N=n, filter_already_liked_items=True
    )
    print(f"\nTop-{n} {label} recommendations for user_idx={user_idx}:")
    for item_idx, score in zip(item_ids, scores):
        movie_id = idx2movie[item_idx]
        title = movie_id
        if movies_df is not None:
            row = movies_df.loc[movies_df["movieId"] == movie_id]
            if not row.empty:
                title = row.iloc[0]["title"]
        print(f"  movie_idx={item_idx:<6} movieId={movie_id:<8} score={score:.3f}   {title}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, default=50)
    parser.add_argument("--als_reg", type=float, default=0.1)
    parser.add_argument("--als_iters", type=int, default=20)
    parser.add_argument("--bpr_reg", type=float, default=0.01)
    parser.add_argument("--bpr_iters", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    train_matrix = load_train_matrix()
    mappings = load_mappings()
    idx2movie = mappings["idx2movie"]

    movies_df = None
    movies_path = os.path.join("data", "raw", "movies.csv")
    if os.path.exists(movies_path):
        movies_df = pd.read_csv(movies_path)

    print(f"Train matrix: {train_matrix.shape[0]} users x {train_matrix.shape[1]} items, "
          f"{train_matrix.nnz} interactions")

    als_model = train_als(train_matrix, factors=args.factors,
                           regularization=args.als_reg, iterations=args.als_iters)
    bpr_model = train_bpr(train_matrix, factors=args.factors,
                           regularization=args.bpr_reg, iterations=args.bpr_iters)

    sanity_check(als_model, train_matrix, idx2movie, movies_df, user_idx=0, n=5, label="ALS")
    sanity_check(bpr_model, train_matrix, idx2movie, movies_df, user_idx=0, n=5, label="BPR")

    with open(os.path.join(MODELS_DIR, "als_model.pkl"), "wb") as f:
        pickle.dump(als_model, f)
    with open(os.path.join(MODELS_DIR, "bpr_model.pkl"), "wb") as f:
        pickle.dump(bpr_model, f)

    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()