import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_item_cf(movie_id, movies_df, ratings_df, n_recommendations=5):
    """
    Item-Based Collaborative Filtering using Cosine Similarity.
    Recommends movies that are similar to the given movie based on user ratings.
    """
    try:
        # Create item-user matrix
        # In a real production scenario, item similarities should be precomputed.
        movie_user_matrix = ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        
        if movie_id not in movie_user_matrix.index:
            # Fallback for new/unknown movies
            return list(movies_df.head(n_recommendations)['title'])
            
        # Calculate item similarity
        item_similarity = cosine_similarity(movie_user_matrix)
        item_similarity_df = pd.DataFrame(item_similarity, index=movie_user_matrix.index, columns=movie_user_matrix.index)
        
        # Get similar movies
        similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)
        
        # Exclude the queried movie
        similar_movies = similar_movies.drop(labels=[movie_id], errors='ignore')
        
        # Get top N movie IDs
        top_movie_ids = similar_movies.head(n_recommendations).index
        
        # Map to titles
        return movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
        
    except Exception as e:
        print(f"Error in Item CF: {e}")
        # Graceful degradation
        return list(movies_df.head(n_recommendations)['title'])
