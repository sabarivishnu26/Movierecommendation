"""
Phase 3.5: Hyperparameter Tuning & Overfitting Diagnosis

WHAT:
  Re-train ALS and BPR across a small grid of hyperparameters and re-score each
  configuration with the SAME held-out ranking metrics from Phase 3, to find
  settings that actually generalize instead of overfitting the training data.

WHY:
  Phase 3 showed BPR's training AUC (~98%) far exceeding its held-out NDCG@10
  (~0.03) -- a classic overfitting signature: with ~610 users, ~9700 items, and
  98%+ sparsity, the default 50 factors / 100 BPR iterations gives the model far
  more capacity than the data can support without stronger regularization pulling
  it back. This script searches over exactly the knobs that control that
  capacity-vs-regularization tradeoff:
    - factors          -> lower = less capacity to memorize, forced to generalize
    - regularization    -> higher = stronger penalty on large factor values
    - iterations        -> fewer = less opportunity to overfit (esp. for BPR)
    - confidence scaling -> how strongly ALS trusts observed interactions
                            (lower can reduce overconfident overfitting)

HOW:
  For each hyperparameter combination: train the model on train_matrix, evaluate
  NDCG@10 (and the other Phase 3 metrics) against test_relevance, and keep track
  of the best-performing config per algorithm. Reuses precision/recall/MAP/NDCG
  and evaluate_recommender directly from evaluate.py so results are computed
  identically to Phase 3 (no drift between "old" and "new" numbers).
"""

import itertools
import os
import pickle

import pandas as pd
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from evaluate import build_test_relevance, evaluate_recommender, recommend_implicit_model, K

PROCESSED_DATA_DIR = "movie-recommender/data/processed"
MODELS_DIR = "movie-recommender/models"

# Smaller, more conservative grids than Phase 2's defaults -- deliberately probing
# LOWER capacity / HIGHER regularization to see if that closes the overfitting gap.
ALS_GRID = {
    "factors": [10, 20, 50],
    "regularization": [0.1, 0.5, 1.0],
    "iterations": [15],
    "confidence_scale": [1.0, 0.5],  # multiplies the already-built confidence matrix
}

BPR_GRID = {
    "factors": [10, 20, 50],
    "regularization": [0.05, 0.1, 0.3],
    "learning_rate": [0.01, 0.05],
    "iterations": [50, 100],
}


def grid_combinations(grid):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def tune_als(train_matrix, test_relevance):
    results = []
    for cfg in grid_combinations(ALS_GRID):
        matrix = (train_matrix * cfg["confidence_scale"]).tocsr()
        model = AlternatingLeastSquares(
            factors=cfg["factors"],
            regularization=cfg["regularization"],
            iterations=cfg["iterations"],
            random_state=42,
        )
        model.fit(matrix)
        metrics = evaluate_recommender(
            lambda uidx: recommend_implicit_model(model, uidx, train_matrix, K),
            test_relevance,
        )
        row = {**cfg, **metrics}
        results.append(row)
        print(f"[ALS] {cfg} -> NDCG@{K}={metrics[f'NDCG@{K}']:.4f}")
    return pd.DataFrame(results).sort_values(f"NDCG@{K}", ascending=False)


def tune_bpr(train_matrix, test_relevance):
    results = []
    for cfg in grid_combinations(BPR_GRID):
        model = BayesianPersonalizedRanking(
            factors=cfg["factors"],
            regularization=cfg["regularization"],
            learning_rate=cfg["learning_rate"],
            iterations=cfg["iterations"],
            random_state=42,
        )
        model.fit(train_matrix.tocsr())
        metrics = evaluate_recommender(
            lambda uidx: recommend_implicit_model(model, uidx, train_matrix, K),
            test_relevance,
        )
        row = {**cfg, **metrics}
        results.append(row)
        print(f"[BPR] {cfg} -> NDCG@{K}={metrics[f'NDCG@{K}']:.4f}")
    return pd.DataFrame(results).sort_values(f"NDCG@{K}", ascending=False)


def main():
    train_matrix = load_npz(os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz")).tocsr()
    test_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "test_df.pkl"))
    test_relevance = build_test_relevance(test_df)

    print("=== Tuning ALS ===")
    als_results = tune_als(train_matrix, test_relevance)
    print("\n=== Tuning BPR ===")
    bpr_results = tune_bpr(train_matrix, test_relevance)

    os.makedirs("movie-recommender/results", exist_ok=True)
    als_results.to_csv("movie-recommender/results/als_tuning.csv", index=False)
    bpr_results.to_csv("movie-recommender/results/bpr_tuning.csv", index=False)

    print("\n=== Best ALS config ===")
    print(als_results.iloc[0])
    print("\n=== Best BPR config ===")
    print(bpr_results.iloc[0])

    print("\nFull grids saved to movie-recommender/results/als_tuning.csv and movie-recommender/results/bpr_tuning.csv")
    print("Compare the best row here against Phase 3's original ALS/BPR rows and SVD "
          "to see whether tuning closed the gap.")


if __name__ == "__main__":
    main()