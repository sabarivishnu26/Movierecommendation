# 🎬 Implicit-Feedback Movie Recommender

A movie recommendation system built on MovieLens, reframed around **implicit
feedback** and evaluated the way real recommenders actually are: by ranking
quality, not rating-prediction error. Includes a documented bug hunt that
overturned an initial finding, a two-stage candidate-generation + re-ranking
architecture verified to beat every single-model baseline, and both an
interactive demo and a production API.

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
  ![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge)
  ![Collaborative Filtering](https://img.shields.io/badge/Collaborative_Filtering-8E24AA?style=for-the-badge)
  ![Two--Stage Ranking](https://img.shields.io/badge/Two--Stage_Ranking-FF7043?style=for-the-badge)

- 📊
  ![MovieLens](https://img.shields.io/badge/MovieLens-Dataset-FF6F00?style=for-the-badge)
  ![Precision@K](https://img.shields.io/badge/Precision@K-009688?style=for-the-badge)
  ![Recall@K](https://img.shields.io/badge/Recall@K-43A047?style=for-the-badge)
  ![NDCG@K](https://img.shields.io/badge/NDCG@K-5E35B1?style=for-the-badge)
  ![Coverage & Diversity](https://img.shields.io/badge/Coverage_%26_Diversity-00897B?style=for-the-badge)

- 🚀
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
  ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## The Key Insight (and the bug that hid it)

> The first pass at this project found something counter-intuitive: classical
> SVD appeared to beat both tuned ALS and tuned BPR on every ranking metric,
> across two dataset scales (100K and 1M ratings) — the opposite of what
> recommender-systems theory predicts for implicit-feedback-specialized models.
>
> Rather than accept a surprising result at face value, it was independently
> audited and the claim was re-verified from scratch. **The root cause: the
> confidence-weighting formula (`1 + 15 × rating`) gave observed interactions
> 16–76× more weight than unobserved ones in ALS's loss function** — a
> hyperparameter inherited from a paper tuned for a very different domain
> (TV-watching-duration counts) and never re-validated for this one. SVD uses
> an *unweighted* loss, so it was largely immune to this specific distortion,
> which is exactly why it looked artificially superior.
>
> **After the fix** (binary confidence, regularization re-tuned via a real grid
> search over alpha × regularization × factors), ALS legitimately beats SVD —
> matching the original theoretical expectation. A subsequent two-stage
> architecture (ALS candidate generation + LightGBM re-ranking) beats both.

| Model | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage | Avg. Rec. Popularity |
|---|---|---|---|---|---|
| Popularity (baseline) | 0.0079 | 0.0397 | 0.0236 | — | — |
| SVD | 0.0180 | 0.0899 | 0.0548 | 21.1% | 1396.6 |
| ALS (corrected, tuned) | 0.0193 | 0.0963 | 0.0588 | 29.0% | 1162.5 |
| **Two-Stage (ALS + LightGBM)** | **0.0207** | **0.1033** | **0.0652** | **38.1%** | **1087.5** |

_Results on MovieLens ml-1m (6,040 users, 3,706 movies, ~1M ratings), time-based
leave-last-2-out evaluation. The two-stage model improves accuracy AND
diversity simultaneously — it isn't just more accurate, it also recommends
from a much wider slice of the catalog instead of leaning on blockbusters.
Full search grid in `results/als_alpha_reg_search.csv`; the original
(pre-fix) SVD-wins result is kept in `results/model_comparison.csv` as a
documented part of the project's history, not hidden._

---

## Architecture

```
Raw MovieLens ratings (explicit, 1-5 stars)
        │
        ▼
┌──────────────────────────┐
│  Phase 1: Preprocessing  │  confidence = 1 + α·rating  (α=0, verified via grid search)
│  Explicit → Implicit     │  time-based leave-last-N-out split, leakage-checked
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 2: Modeling        │  ALS · BPR · SVD (baseline)
│  Matrix Factorization     │  trained on the same confidence matrix
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 3: Evaluation      │  Precision@K · Recall@K · MAP@K · NDCG@K
│  Ranking Metrics          │  (not RMSE — there's no ground-truth rating here)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 3.5: Bug Hunt       │  independently audited a surprising result,
│  Diagnosis & Fix           │  found & fixed the confidence-weighting bug
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 4: Explanations    │  genre-overlap grounding + "most similar
│  Why was this picked?     │  movie you rated highly" + optional LLM rephrase
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 5: Two-Stage        │  Stage 1: ALS retrieves ~200 candidates (recall)
│  Candidate Gen + Re-Rank   │  Stage 2: LightGBM re-ranks with richer features (precision)
└────────────┬─────────────┘
             ▼
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐    ┌──────────────┐
│Streamlit│    │  FastAPI      │  /recommend/{user_id}, /health
│  Demo   │    │  + Docker     │  auto-generated OpenAPI docs
└─────────┘    └──────────────┘
```

---

## Why Implicit Feedback

Real-world recommenders (Netflix, Spotify, YouTube) rarely observe explicit
1–5 star preferences — they observe *interactions*: did the user watch, click,
or stream something, or not? This project reframes MovieLens the same way:
every rating becomes a positive interaction signal, and (after the confidence
bug fix) treated as a binary signal rather than an aggressively re-weighted
one — not as a value to reconstruct.

That reframing also changes how the system must be evaluated: there's no
ground-truth rating to compute RMSE against anymore, only "did we rank the
movies the user actually watched next near the top of our list?" — hence
ranking metrics (Precision/Recall/MAP/NDCG@K), plus coverage/diversity metrics
to catch a model that's accurate but secretly just recommending blockbusters.

## Two-Stage Recommendation

Single-model recommenders force a tradeoff: a model cheap enough to score an
entire catalog per user (like ALS) can't afford rich features; a model rich
enough for strong ranking can't afford to run over the whole catalog. Real
large-scale recommenders (this pattern is described in YouTube's own published
recommendation papers) solve this with two stages:

1. **Candidate generation (ALS):** cheaply retrieves ~200 plausible movies per
   user from the full catalog. Optimized for *recall* — don't miss good items.
2. **Re-ranking (LightGBM):** re-scores only those ~200 candidates using
   richer features — the ALS score itself, item popularity, item/user average
   rating, and genre overlap with the user's own rating-weighted genre
   profile. Optimized for *precision* — get the order right.

The re-ranker is trained on a leave-last-1-out split carved out of *training*
data only, so it never sees the real test set during its own training —
avoiding the exact kind of leakage this project already caught once.

## Explanation Layer

Every recommendation ships with a plain-language reason, grounded in real facts
pulled from the user's own history — e.g.:

> **The Lion King** — *Recommended because you rated "Cinderella" highly, and
> both share Animation, Children's, and Musical.*

An optional layer (off by default) can rephrase these through a small LLM call
for more natural wording — the LLM is only ever given the verified facts, never
asked to invent a justification, so it can't hallucinate a reason that isn't
actually true.

## Production API

A FastAPI service wraps the two-stage recommender for machine-to-machine
consumption (not just a human clicking through a browser):

- `GET /recommend/{user_id}?k=10` — Top-N recommendations with explanations
- `GET /health` — liveness check
- Pydantic request/response validation, proper HTTP status codes (404 for
  unknown users, 400 for invalid parameters)
- Auto-generated interactive docs at `/docs`
- Containerized with a `Dockerfile` including a container healthcheck

**Known limitation, stated honestly:** cold-start users (no rating history)
currently return a 404 rather than a fallback recommendation — a named next
step, not a hidden gap.

## Project Structure

```
movie-recommender/
├── api/
│   └── main.py                    # FastAPI production service
├── app/
│   └── app.py                     # Streamlit demo
├── src/
│   ├── data_processing.py         # Phase 1 (confidence bug fix, leakage fix)
│   ├── train_models.py            # Phase 2 (corrected ALS defaults)
│   ├── evaluate.py                # Phase 3 (ranking metrics)
│   ├── tune_hyperparameter.py     # Phase 3.5 (original tuning grid)
│   ├── explain.py                 # Phase 4 (explanation layer)
│   ├── train_ranker.py            # Phase 5 (two-stage LightGBM re-ranker)
│   └── evaluate_final.py          # Phase 5 (final comparison + diversity metrics)
├── data/
│   ├── raw/                       # ratings.dat, movies.dat (not committed if large)
│   └── processed/                 # train/test splits, sparse matrix, ID mappings
├── models/                        # trained ALS/BPR/SVD/LightGBM models (pickled)
├── results/                       # evaluation tables, hyperparameter search grids
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
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
python src/data_processing.py
python src/train_models.py
python src/evaluate.py
python src/tune_hyperparameter.py   # optional, ~30 model trainings
python src/train_ranker.py
python src/evaluate_final.py
python src/explain.py --user_idx 0 --k 5

# 4. Launch the interfaces
streamlit run app/app.py                       # human-facing demo
uvicorn api.main:app --reload --port 8000       # machine-facing API, see /docs

# 5. (Optional) Run the API in Docker
docker build -t movie-recommender-api .
docker run -p 8000:8000 movie-recommender-api
```

## Future Work

- Cold-start fallback (content-based genre similarity for brand-new users)
- Test the two-stage architecture's diversity gain on `ml-25m` at larger scale
- Experiment tracking (MLflow) for the hyperparameter/architecture history
- CI/CD (GitHub Actions: tests + Docker build on push)
- Cloud deployment of the API (single container, no orchestration needed at this scale)

---
