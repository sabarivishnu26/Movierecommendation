import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'svd_model.pkl')

def load_svd_model():
    """Loads the trained SVD model from disk."""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"SVD model not found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading SVD model: {e}")
        return None

def recommend_svd(movie_id, movies_df, ratings_df=None, n_recommendations=5):
    """
    SVD based recommendations.
    Uses the Surprise library's SVD model to recommend movies.
    Since SVD predicts ratings for user-item pairs, we can find items
    with similar latent factors or use a mock user profile.
    For this demo, we simulate finding similar items using the model's item factors
    if available, otherwise we use a fallback mechanism.
    """
    model = load_svd_model()
    
    # If model couldn't be loaded or doesn't have item factors (qi), use a fallback approach
    if not model or not hasattr(model, 'qi'):
        print("SVD model unavailable or lacks item factors. Using fallback.")
        # Fallback to popular movies just to ensure the UI works
        return list(movies_df.head(n_recommendations)['title'])
        
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get item inner id
        try:
            inner_id = model.trainset.to_inner_iid(movie_id)
        except ValueError:
            # Item not part of the trainset
            return list(movies_df.head(n_recommendations)['title'])
            
        # Get item factors
        item_factors = model.qi
        
        # Calculate similarity between this item and all others
        target_factors = item_factors[inner_id].reshape(1, -1)
        similarities = cosine_similarity(target_factors, item_factors).flatten()
        
        # Get indices of most similar items (excluding the item itself)
        # argsort sorts ascending, so we take the last N items (highest similarity)
        import numpy as np
        similar_indices = similarities.argsort()[-(n_recommendations + 1):][::-1]
        
        # Convert inner ids back to raw ids
        top_movie_ids = []
        for idx in similar_indices:
            if idx != inner_id:
                try:
                    raw_id = model.trainset.to_raw_iid(idx)
                    top_movie_ids.append(raw_id)
                    if len(top_movie_ids) == n_recommendations:
                        break
                except ValueError:
                    continue
                    
        # Map to titles
        return movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
        
    except Exception as e:
        print(f"Error generating SVD recommendations: {e}")
        return list(movies_df.head(n_recommendations)['title'])
