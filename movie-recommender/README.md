# 🎬 Implicit-Feedback Movie Recommender

A movie recommendation system built on MovieLens, reframed around **implicit
feedback** and evaluated the way real recommenders actually are: by ranking
quality, not rating-prediction error. Includes a documented, counter-intuitive
empirical finding, and a deployed interactive demo.

---
## 🛠️ Tech Stack

- 🐍
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

- 🤖
  ![Implicit Feedback](https://img.shields.io/badge/Implicit_Feedback-1E88E5?style=for-the-badge)
  ![ALS](https://img.shields.io/badge/ALS-4285F4?style=for-the-badge)
  ![BPR](https://img.shields.io/badge/BPR-34A853?style=for-the-badge)
  ![SVD](https://img.shields.io/badge/SVD-EA4335?style=for-the-badge)
  ![Collaborative Filtering](https://img.shields.io/badge/Collaborative_Filtering-8E24AA?style=for-the-badge)

- 📊
  ![MovieLens](https://img.shields.io/badge/MovieLens-Dataset-FF6F00?style=for-the-badge)
  ![Precision@K](https://img.shields.io/badge/Precision@K-009688?style=for-the-badge)
  ![Recall@K](https://img.shields.io/badge/Recall@K-43A047?style=for-the-badge)
  ![NDCG@K](https://img.shields.io/badge/NDCG@K-5E35B1?style=for-the-badge)

- 🌐
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
  ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
  ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)
  
## The Key Insight

> Ranking-optimized implicit-feedback models (ALS, BPR) are the industry-standard
> approach for recommendation at scale (used at companies like Spotify) — the
> expectation going in was that they'd outperform classical SVD. They didn't.
>
> Across **two dataset sizes** (100K and 1M MovieLens ratings), a classical SVD
> baseline — trained on the *same* implicit confidence-weighted matrix — beat
> both tuned ALS and tuned BPR on every ranking metric (Precision@10, Recall@10,
> MAP@10, NDCG@10). A hyperparameter search confirmed BPR's initial gap was a
> genuine overfitting problem (training AUC ~98% vs. held-out NDCG@10 of ~0.03)
> and fixed it — but even after tuning, SVD still won at both scales.
>
> **Conclusion:** the advantage implicit-feedback-specialized models show in
> industry depends on interaction-log scale and density (millions of dense
> events). It doesn't automatically transfer to moderate-scale, single-domain
> datasets like MovieLens — and this project quantifies that instead of
> assuming it.


| Model | Precision@10 | Recall@10 | MAP@10 | NDCG@10 |
|---|---|---|---|---|
| Popularity (baseline) | 0.0079 | 0.0397 | 0.0136 | 0.0236 |
| **SVD** | **0.0180** | **0.0899** | **0.0327** | **0.0548** |
| ALS (tuned) | 0.0147 | 0.0733 | 0.0257 | 0.0440 |
| BPR (tuned) | 0.0125 | 0.0623 | 0.0230 | 0.0385 |

_Results on MovieLens ml-1m (6,040 users, 3,706 movies, ~1M ratings), time-based
leave-last-2-out evaluation. See `results/` for the full hyperparameter grids._

---

## Architecture

```
Raw MovieLens ratings (explicit, 1-5 stars)
        │
        ▼
┌─────────────────────────┐
│  Phase 1: Preprocessing │  confidence = 1 + α·rating
│  Explicit → Implicit    │  time-based leave-last-N-out split
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Phase 2: Modeling       │  ALS · BPR · SVD (baseline)
│  Matrix Factorization    │  trained on the same confidence matrix
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Phase 3: Evaluation     │  Precision@K · Recall@K · MAP@K · NDCG@K
│  Ranking Metrics         │  (not RMSE — there's no ground-truth rating here)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Phase 3.5: Tuning       │  grid search, diagnosed BPR overfitting
│  Diagnosis & Fix         │  (train AUC 98% vs. held-out NDCG ~0.03)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Phase 4: Explanations   │  genre-overlap grounding + "most similar
│  Why was this picked?    │  movie you rated highly" + optional LLM rephrase
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Phase 5: Deployment      │  Streamlit app — pick a user, see Top-N
│  Interactive Demo         │  recommendations with explanations, live
└──────────────────────────┘
```

---

## Why Implicit Feedback

Real-world recommenders (Netflix, Spotify, YouTube) rarely observe explicit
1–5 star preferences — they observe *interactions*: did the user watch, click,
or stream something, or not? This project reframes MovieLens the same way:
every rating becomes a positive interaction signal, and the star value is used
only as a **confidence weight** (Hu, Koren & Volinsky, 2008) on how strong that
signal is — not as a value to reconstruct.

That reframing also changes how the system must be evaluated: there's no
ground-truth rating to compute RMSE against anymore, only "did we rank the
movies the user actually watched next near the top of our list?" — hence
ranking metrics (Precision/Recall/MAP/NDCG@K) instead of error metrics.

## Explanation Layer

Every recommendation ships with a plain-language reason, grounded in real facts
pulled from the user's own history — e.g.:

> **The Shawshank Redemption** — *Recommended because you rated "The Last Days
> of Disco" highly, and both share Drama.*

An optional layer (off by default) can rephrase these through a small LLM call
for more natural wording — the LLM is only ever given the verified facts, never
asked to invent a justification, so it can't hallucinate a reason that isn't
actually true.

## Project Structure

```
movie-recommender/
├── app/
│   └── app.py                  # Streamlit demo
├── data/
│   ├── raw/                    # ratings.dat, movies.dat (not committed if large)
│   └── processed/              # train/test splits, sparse matrix, ID mappings
├── models/                     # trained ALS/BPR/SVD models (pickled)
├── results/                    # evaluation tables, hyperparameter tuning grids
├── src/
│   ├── data_preprocessing.py   # Phase 1
│   ├── train_models.py         # Phase 2
│   ├── evaluate.py             # Phase 3
│   ├── tune_hyperparams.py     # Phase 3.5
│   └── explain.py              # Phase 4
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Get the data
#    Download ml-1m from https://grouplens.org/datasets/movielens/1m/
#    and place ratings.dat + movies.dat into data/raw/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline in order
python src/data_preprocessing.py
python src/train_models.py
python src/evaluate.py
python src/tune_hyperparams.py   # optional, ~30 model trainings
python src/explain.py --user_idx 0 --k 5

# 4. Launch the demo
streamlit run app/app.py
```

## Future Work

- Solve cold-start with a content-based fallback for brand-new users/movies
- Test on `ml-25m` to see if the SVD-vs-implicit-models finding holds at even larger scale
- Add diversity/novelty/popularity-bias metrics alongside accuracy
- Time-aware weighting (recent ratings weighted more heavily)
