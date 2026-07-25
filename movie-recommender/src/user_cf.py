import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_user_cf(movie_id, movies_df, ratings_df, n_recommendations=5):
    """
    User-Based Collaborative Filtering using Cosine Similarity.
    Recommends movies by finding users who liked the given movie,
    finding similar users, and returning what those similar users liked.
    """
    try:
        # Create a user-item matrix (reduced size for performance if needed)
        # In a real production scenario, this matrix should be precomputed.
        user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        if movie_id not in user_movie_matrix.columns:
            # Fallback for new/unknown movies: return popular movies
            return list(movies_df.head(n_recommendations)['title'])
            
        # 1. Find users who rated this movie highly (>= 4.0)
        movie_ratings = user_movie_matrix[movie_id]
        fans = movie_ratings[movie_ratings >= 4.0].index
        
        if len(fans) == 0:
            return list(movies_df.head(n_recommendations)['title'])
            
        # 2. Filter matrix to only relevant users to speed up cosine similarity
        # Here we just use all users for simplicity assuming a small demo dataset
        user_similarity = cosine_similarity(user_movie_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
        
        # 3. Get top 10 similar users based on the fans of the movie
        similar_users = user_similarity_df.loc[fans].mean().sort_values(ascending=False).index[1:11]
        
        # 4. Get movies liked by these similar users
        recommendations = user_movie_matrix.loc[similar_users].mean().sort_values(ascending=False)
        
        # Filter out the input movie
        recommendations = recommendations.drop(labels=[movie_id], errors='ignore')
        
        # Get top N movie IDs
        top_movie_ids = recommendations.head(n_recommendations).index
        
        # Map to titles
        return movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
        
    except Exception as e:
        print(f"Error in User CF: {e}")
        # Graceful degradation
        return list(movies_df.head(n_recommendations)['title'])
