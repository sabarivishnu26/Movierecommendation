import pandas as pd
import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_data():
    """
    Loads the explicit ratings and movies datasets.
    """
    movies_path = os.path.join(DATA_DIR, 'movies.csv')
    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
    
    try:
        # For production robustness, handle missing files gracefully
        if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
            print(f"Data files not found in {DATA_DIR}. Please ensure movies.csv and ratings.csv are present.")
            return None, None
            
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        
        return movies_df, ratings_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
