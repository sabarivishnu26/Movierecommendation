"""
Phase 3: Evaluation — Ranking Metrics (Precision@K, Recall@K, MAP@K, NDCG@K)

WHAT:
  Evaluate every trained model (Popularity baseline, SVD baseline, ALS, BPR)
  against the held-out test interactions from Phase 1, using metrics built for
  RANKING quality rather than rating-prediction error.

WHY these metrics instead of RMSE/MAE:
  Implicit feedback has no ground-truth "rating" to compare a prediction against
  -- only "did the user interact with this item or not." The right question is
  "did we rank the items the user actually went on to watch near the top of our
  list?" -- which is exactly what these metrics measure:
    Precision@K -> of the K items we recommended, how many were actually relevant?
    Recall@K    -> of all relevant items for this user, how many did we surface in K?
    MAP@K       -> like precision, but rewards relevant items appearing EARLIER
                   in the ranked list (rank-based discount)
    NDCG@K      -> same idea as MAP, with a smoother logarithmic position discount
                   (the standard metric used in real ranking/search evaluation)

WHY a Popularity baseline:
  A well-known trap in recommender research: it's easy to build a system that's
  secretly no better than "recommend the most popular movies to everyone."
  Beating this baseline is the minimum bar for proving genuine personalization.

WHY an SVD baseline:
  This ties back to the original (explicit-ratings) project. Running classic SVD
  matrix factorization on the SAME implicit confidence matrix, evaluated with the
  SAME ranking metrics, gives a direct, fair, quantified answer to: "did switching
  to ranking-optimized implicit models (ALS/BPR) actually help?"

HOW:
  For every user with test interactions, each model produces a ranked list of
  Top-K unseen items. We compare that ranked list against the user's actual
  held-out (test) items and average each metric across all users.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.linalg import svds

PROCESSED_DATA_DIR = "movie-recommender/data/processed"
MODELS_DIR = "movie-recommender/models"
K = 10  # Top-K cutoff for all metrics


# ---------------------------------------------------------------------------
# Ranking metric functions
# ---------------------------------------------------------------------------

def precision_recall_at_k(recommended, relevant, k):
    if not relevant:
        return None, None
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    precision = hits / k
    recall = hits / len(relevant)
    return precision, recall


def average_precision_at_k(recommended, relevant, k):
    if not relevant:
        return None
    recommended_k = recommended[:k]
    hits = 0
    score = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return None
    recommended_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(recommended_k) if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Recommenders (all return a ranked list of item indices, excluding train items)
# ---------------------------------------------------------------------------

def popularity_ranking(train_matrix):
    """Rank all items by total training interaction count, descending."""
    item_counts = np.asarray((train_matrix > 0).sum(axis=0)).ravel()
    return np.argsort(-item_counts)


def recommend_popularity(user_idx, train_matrix, popularity_order, k):
    seen = set(train_matrix[user_idx].nonzero()[1])
    recs = [item for item in popularity_order if item not in seen]
    return recs[:k]


def train_svd_baseline(train_matrix, k_factors=50):
    """
    Classic SVD baseline via scipy.sparse.linalg.svds, run on the SAME implicit
    confidence matrix as ALS/BPR (for a fair, apples-to-apples comparison).
    Returns low-rank factors U, S, Vt such that U @ diag(S) @ Vt approximates
    the training matrix.
    """
    max_k = min(train_matrix.shape) - 1
    k_factors = min(k_factors, max_k)
    u, s, vt = svds(train_matrix.astype(np.float64), k=k_factors)
    return u, s, vt


def recommend_svd(user_idx, train_matrix, u, s, vt, k):
    scores = (u[user_idx] * s) @ vt
    seen = set(train_matrix[user_idx].nonzero()[1])
    ranked = np.argsort(-scores)
    recs = [item for item in ranked if item not in seen]
    return recs[:k]


def recommend_implicit_model(model, user_idx, train_matrix, k):
    """Works for both ALS and BPR models (same `implicit` API)."""
    item_ids, _scores = model.recommend(
        user_idx, train_matrix[user_idx], N=k, filter_already_liked_items=True
    )
    return list(item_ids)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def build_test_relevance(test_df):
    """user_idx -> set of movie_idx the user actually interacted with in test."""
    return test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()


def evaluate_recommender(recommend_fn, test_relevance, k=K):
    """
    recommend_fn(user_idx) -> ranked list of item indices.
    Averages Precision@K, Recall@K, MAP@K, NDCG@K across all users who have
    test interactions.
    """
    precisions, recalls, aps, ndcgs = [], [], [], []
    for user_idx, relevant in test_relevance.items():
        recommended = recommend_fn(user_idx)
        p, r = precision_recall_at_k(recommended, relevant, k)
        ap = average_precision_at_k(recommended, relevant, k)
        ndcg = ndcg_at_k(recommended, relevant, k)
        if p is not None:
            precisions.append(p)
            recalls.append(r)
            aps.append(ap)
            ndcgs.append(ndcg)

    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"MAP@{k}": np.mean(aps),
        f"NDCG@{k}": np.mean(ndcgs),
    }


def main():
    train_matrix = load_npz(os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz")).tocsr()
    test_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "test_df.pkl"))
    test_relevance = build_test_relevance(test_df)

    results = {}

    # --- Popularity baseline ---
    pop_order = popularity_ranking(train_matrix)
    results["Popularity"] = evaluate_recommender(
        lambda u: recommend_popularity(u, train_matrix, pop_order, K), test_relevance
    )

    # --- SVD baseline ---
    u, s, vt = train_svd_baseline(train_matrix, k_factors=50)
    results["SVD"] = evaluate_recommender(
        lambda uidx: recommend_svd(uidx, train_matrix, u, s, vt, K), test_relevance
    )

    # --- ALS ---
    als_path = os.path.join(MODELS_DIR, "als_model.pkl")
    if os.path.exists(als_path):
        with open(als_path, "rb") as f:
            als_model = pickle.load(f)
        results["ALS"] = evaluate_recommender(
            lambda uidx: recommend_implicit_model(als_model, uidx, train_matrix, K),
            test_relevance,
        )
    else:
        print(f"[skip] {als_path} not found -- run train_models.py first")

    # --- BPR ---
    bpr_path = os.path.join(MODELS_DIR, "bpr_model.pkl")
    if os.path.exists(bpr_path):
        with open(bpr_path, "rb") as f:
            bpr_model = pickle.load(f)
        results["BPR"] = evaluate_recommender(
            lambda uidx: recommend_implicit_model(bpr_model, uidx, train_matrix, K),
            test_relevance,
        )
    else:
        print(f"[skip] {bpr_path} not found -- run train_models.py first")

    # --- Report ---
    results_df = pd.DataFrame(results).T
    print("\n=== Phase 3: Ranking Metric Comparison (Top-{}) ===".format(K))
    print(results_df.round(4).to_string())

    os.makedirs("movie-recommender/results", exist_ok=True)
    results_df.to_csv("movie-recommender/results/model_comparison.csv")
    print("\nSaved to movie-recommender/results/model_comparison.csv")


if __name__ == "__main__":
    main()