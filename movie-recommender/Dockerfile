# Movie Recommender API -- production container
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ api/
COPY src/ src/
COPY data/processed/ data/processed/
COPY data/raw/movies.dat data/raw/movies.dat
COPY models/als_final.pkl models/als_final.pkl
COPY models/lgbm_ranker.pkl models/lgbm_ranker.pkl

EXPOSE 8000

# Basic container health check hitting the API's own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
