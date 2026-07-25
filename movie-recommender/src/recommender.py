import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from user_cf import recommend_user_cf
from item_cf import recommend_item_cf
from svd_model import recommend_svd

def get_recommendations(movie_title, model_type):
    """
    Main function to get movie recommendations based on title and selected model.
    model_type can be 'user', 'item', or 'svd'.
    """
    movies_df, ratings_df = load_data()
    
    if movies_df is None or ratings_df is None:
        return {"error": "Dataset not found. Please ensure data/movies.csv and data/ratings.csv exist."}
        
    # Standardize movie title search (case-insensitive substring match as a fallback)
    # First, try exact match
    exact_match = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
    if not exact_match.empty:
        movie_id = exact_match.iloc[0]['movieId']
        actual_title = exact_match.iloc[0]['title']
    else:
        # Try substring match
        substring_match = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
        if not substring_match.empty:
            movie_id = substring_match.iloc[0]['movieId']
            actual_title = substring_match.iloc[0]['title']
        else:
            return {"error": f"Movie '{movie_title}' not found in the dataset."}
            
    # Get recommendations based on selected model
    recommendations = []
    
    if model_type == 'user':
        recommendations = recommend_user_cf(movie_id, movies_df, ratings_df)
    elif model_type == 'item':
        recommendations = recommend_item_cf(movie_id, movies_df, ratings_df)
    elif model_type == 'svd':
        recommendations = recommend_svd(movie_id, movies_df, ratings_df)
    else:
        return {"error": "Invalid model type selected. Choose 'user', 'item', or 'svd'."}
        
    return {
        "model": "User-Based CF" if model_type == 'user' else "Item-Based CF" if model_type == 'item' else "SVD",
        "input_movie": actual_title,
        "recommendations": recommendations,
        "mode_id": str(model_type)
    }
