# Movie Recommendation System

A full-stack movie recommendation web application using Machine Learning.

## Features
- **User-Based Collaborative Filtering**: Recommends movies based on similar users.
- **Item-Based Collaborative Filtering**: Recommends movies similar to the one selected.
- **Matrix Factorization (SVD)**: Uses latent factors to predict preferences using the Surprise library.

## Project Structure
```text
movie-recommender/
├── data/                # Place movies.csv and ratings.csv here
├── src/                 # Core ML logic
│   ├── data_loader.py
│   ├── user_cf.py
│   ├── item_cf.py
│   ├── svd_model.py
│   └── recommender.py
├── backend/             # Flask API server
│   └── app.py
├── frontend/            # UI
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── models/              # Serialized ML models (e.g., svd_model.pkl)
├── requirements.txt     # Python dependencies
└── README.md            # Instructions
```

## Setup Instructions

### 1. Data Preparation
Make sure to place `movies.csv` and `ratings.csv` in the `data/` directory. If they don't exist, the UI will load, but the API will return an error when recommending.

### 2. Local Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running Locally
Start the Flask backend (which also serves the frontend UI on port 5000):
```bash
cd backend
python app.py
```
Open your browser and visit `http://localhost:5000`.

## Deployment Instructions

### Backend (Render)
1. Push your code to a GitHub repository.
2. Sign up on [Render](https://render.com) and create a new **Web Service**.
3. Connect your repository.
4. Set the Root Directory to empty or `.`.
5. Set the Build Command to: `pip install -r requirements.txt`
6. Set the Start Command to: `gunicorn backend.app:app`
7. Click **Create Web Service**.

### Frontend (Vercel)
The existing setup serves the frontend locally via Flask's template system. If you wish to host the frontend entirely separate from the Flask app on Vercel for maximum performance:
1. Update `frontend/templates/index.html` to remove Flask `{{ url_for() }}` templating and use standard relative paths (e.g., `href="style.css"` and `src="script.js"`).
2. Move `index.html`, `style.css`, and `script.js` into a single flattened directory (or configure Vercel paths).
3. Update the `fetch()` call in `script.js` from pointing to `/recommend` to your deployed Render URL (e.g., `https://your-render-app.onrender.com/recommend`). You may need to enable CORS in `app.py` for this to work (`pip install flask-cors` and `CORS(app)`).
4. Push these changes and import the project into Vercel, pointing the root directory to your separated frontend folder.
