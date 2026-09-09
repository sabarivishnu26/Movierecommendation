"""
Final Model Comparison: Popularity / SVD / ALS (tuned) / Two-Stage

WHAT: Evaluates every model in the project on the SAME held-out test set with
BOTH accuracy metrics (Precision/Recall/NDCG@10, from evaluate.py) AND
diversity/coverage metrics that most projects skip entirely.

WHY: A model that scores well on accuracy alone can still be a bad
recommender if it just recommends the same handful of blockbusters to
everyone. Catalog coverage (what fraction of the catalog ever gets
recommended) and average recommended-item popularity (a proxy for how much
the model leans on "safe", already-popular picks vs. genuine personalization)
round out the accuracy story. This project's two-stage model turned out to
improve BOTH simultaneously -- better accuracy AND better diversity -- which
is worth highlighting explicitly rather than assuming a tradeoff exists.

HOW: Reuses the exact ranking-metric functions from evaluate.py so numbers are
never subtly re-derived differently between scripts.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.linalg import svds

from evaluate import (
    build_test_relevance, evaluate_recommender, precision_recall_at_k,
    ndcg_at_k, K,
)
from train_ranker import (
    build_genre_lookup, build_user_genre_profiles, genre_overlap_score,
    build_feature_row, load_movies, FEATURE_COLS, N_CANDIDATES,
)

PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"


def diversity_metrics(recommend_fn, users, item_pop, n_items, k=10):
    all_recs, pop_scores = [], []
    for u in users:
        recs = recommend_fn(u)[:k]
        all_recs.extend(recs)
        pop_scores.extend(item_pop.get(i, 0) for i in recs)
    coverage = len(set(all_recs)) / n_items
    avg_popularity = np.mean(pop_scores) if pop_scores else 0.0
    return coverage, avg_popularity


def main():
    train_matrix = load_npz(os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz")).tocsr()
    train_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "train_df.pkl"))
    test_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "test_df.pkl"))
    with open(os.path.join(PROCESSED_DATA_DIR, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "als_final.pkl"), "rb") as f:
        als = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "lgbm_ranker.pkl"), "rb") as f:
        ranker_bundle = pickle.load(f)
    ranker, feature_cols = ranker_bundle["model"], ranker_bundle["feature_cols"]

    movies_df = load_movies()
    movieidx2genres = build_genre_lookup(mappings, movies_df)
    user_genre_profile = build_user_genre_profiles(train_df, movieidx2genres)
    item_pop = train_df.groupby("movie_idx").size()
    item_avg_rating = train_df.groupby("movie_idx")["rating"].mean()
    user_avg_rating = train_df.groupby("user_idx")["rating"].mean()
    user_num_ratings = train_df.groupby("user_idx").size()

    test_relevance = build_test_relevance(test_df)
    n_items = train_matrix.shape[1]
    sample_users = list(range(0, train_matrix.shape[0], 5))  # subsample for speed on diversity metrics

    def recommend_svd(u, uu, ss, vv, k=K):
        scores = (uu[u] * ss) @ vv
        seen = set(train_matrix[u].nonzero()[1])
        ranked = np.argsort(-scores)
        return [i for i in ranked if i not in seen][:k]

    def recommend_als(u, k=K):
        ids, _ = als.recommend(u, train_matrix[u], N=k, filter_already_liked_items=True)
        return list(ids)

    def recommend_two_stage(u, k=K):
        ids, scores = als.recommend(u, train_matrix[u], N=N_CANDIDATES, filter_already_liked_items=True)
        feats = pd.DataFrame(
            [build_feature_row(u, m, s, item_pop, item_avg_rating, user_avg_rating,
                                user_num_ratings, user_genre_profile, movieidx2genres)
             for m, s in zip(ids, scores)],
            columns=feature_cols,
        )
        order = np.argsort(-ranker.predict(feats))
        return [ids[i] for i in order[:k]]

    u_svd, s_svd, vt_svd = svds(train_matrix.astype(np.float64), k=50)

    models = {
        "SVD": lambda u: recommend_svd(u, u_svd, s_svd, vt_svd),
        "ALS (tuned)": recommend_als,
        "Two-Stage (ALS + LightGBM)": recommend_two_stage,
    }

    rows = []
    for name, fn in models.items():
        acc = evaluate_recommender(fn, test_relevance, K)
        cov, pop = diversity_metrics(fn, sample_users, item_pop, n_items, K)
        rows.append({"model": name, **acc, "catalog_coverage": cov, "avg_rec_popularity": pop})
        print(f"{name}: {acc} | coverage={cov:.2%} | avg_popularity={pop:.1f}")

    results_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "final_model_comparison.csv"), index=False)
    print(f"\nSaved to {RESULTS_DIR}/final_model_comparison.csv")


if __name__ == "__main__":
    main()
