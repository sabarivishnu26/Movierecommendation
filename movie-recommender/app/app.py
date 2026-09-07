"""
Phase 5: Deployment -- Streamlit Demo App

WHAT:
  A small interactive web app: pick a user, see their Top-N movie
  recommendations from the SVD model (the empirical winner from Phase 3),
  each with a genre-grounded explanation from Phase 4. Also shows the user's
  own top-rated movies for context, so a viewer can sanity-check "do these
  recommendations actually make sense given what this user liked?"

WHY:
  This turns three phases of notebook/script work into something a reviewer can
  actually click through in a browser -- the single biggest jump in perceived
  quality for a portfolio project. It also doubles as a live sanity check: if
  recommendations look wrong for a user whose taste is obvious from their
  history, that's immediately visible here instead of buried in a metrics table.

HOW:
  All the heavy artifacts (train matrix, SVD factors, movies table, mappings)
  are loaded ONCE via st.cache_resource, not on every interaction. Recommending
  for a newly-selected user is cheap (one matrix-vector product), so the app
  stays responsive even though the underlying data is ~1M interactions.

RUN LOCALLY:
    streamlit run app/app.py

DEPLOY (Streamlit Community Cloud):
    1. Push this project to a public GitHub repo, including:
         data/processed/train_matrix.npz
         data/processed/train_df.pkl
         data/processed/mappings.pkl
         models/svd_model.pkl
         data/raw/movies.dat (or movies.csv)
       (the large raw ratings file is NOT needed at runtime -- leave it out)
    2. Go to share.streamlit.io, connect the repo, set the main file path to
       app/app.py, and deploy.
"""

import os
import pickle
import sys

import pandas as pd
import streamlit as st
from scipy.sparse import load_npz

# Allow importing from wherever the pipeline scripts (explain.py, etc.) live --
# checks both "src" and "notebooks", at both one and two directory levels above
# this file, since project layouts vary (e.g. notebooks/ as a sibling of the
# repo root vs. nested inside it).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    os.path.dirname(_APP_DIR),                       # one level up (../)
    os.path.dirname(os.path.dirname(_APP_DIR)),      # two levels up (../../)
]
for _root in _CANDIDATE_ROOTS:
    for _candidate in ("src", "notebooks"):
        _candidate_path = os.path.join(_root, _candidate)
        if os.path.isdir(_candidate_path):
            sys.path.insert(0, _candidate_path)

from explain import (  # noqa: E402
    load_movies,
    recommend_svd,
    generate_explanations,
)

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_resource
def load_everything():
    train_matrix = load_npz(os.path.join(PROCESSED_DATA_DIR, "train_matrix.npz")).tocsr()
    train_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "train_df.pkl"))
    with open(os.path.join(PROCESSED_DATA_DIR, "mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "svd_model.pkl"), "rb") as f:
        svd = pickle.load(f)
    movies_df = load_movies()
    return train_matrix, train_df, mappings, svd, movies_df


def get_user_top_rated(user_idx, train_df, movies_df, n=5):
    history = train_df[train_df["user_idx"] == user_idx].merge(movies_df, on="movieId")
    history = history.sort_values("rating", ascending=False)
    return history[["title", "rating", "genres"]].head(n)


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
    st.title("🎬 Implicit-Feedback Movie Recommender")
    st.caption(
        "SVD matrix factorization trained on implicit-feedback-weighted MovieLens "
        "ratings, with genre-grounded explanations for each recommendation."
    )

    train_matrix, train_df, mappings, svd, movies_df = load_everything()
    user2idx = mappings["user2idx"]

    col1, col2 = st.columns([2, 1])
    with col1:
        raw_user_id = st.selectbox(
            "Choose a user",
            options=sorted(user2idx.keys()),
            format_func=lambda uid: f"User {uid}",
        )
    with col2:
        n_recs = st.slider("Recommendations", min_value=3, max_value=15, value=5)

    user_idx = user2idx[raw_user_id]

    st.subheader("This user's top-rated movies")
    top_rated = get_user_top_rated(user_idx, train_df, movies_df)
    if top_rated.empty:
        st.write("No rating history available for this user.")
    else:
        for _, row in top_rated.iterrows():
            st.write(f"⭐ {row['rating']:.0f} -- **{row['title']}** _{row['genres']}_")

    st.subheader(f"Top {n_recs} recommendations")
    rec_idxs = recommend_svd(user_idx, train_matrix, svd["u"], svd["s"], svd["vt"], n_recs)
    explanations = generate_explanations(
        user_idx, rec_idxs, train_df, movies_df, mappings["idx2movie"]
    )

    for e in explanations:
        with st.container(border=True):
            st.markdown(f"**{e['title']}**")
            st.caption(e["reason"])


if __name__ == "__main__":
    main()