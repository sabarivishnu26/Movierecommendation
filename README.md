🎬 Movie Recommendation System
📌 Problem Statement

Recommender systems help users discover relevant content from large catalogs.
This project builds and compares multiple collaborative filtering approaches to recommend movies using user rating behavior.

The objective is to:

  * Predict user preferences
  * Handle sparse rating data
  * Compare recommendation techniques
  * Identify the most scalable model

📊 Dataset

This project uses the MovieLens Small Dataset.

Dataset Details

  * Source: GroupLens Research
  * Users: ~600
  * Movies: ~9,000
  * Ratings: ~100,000
  * Rating Scale: 0.5 – 5.0

Files Used
ratings.csv → user-movie ratings
movies.csv → movie metadata

🔬 Approach

The project follows an incremental recommender system pipeline:

  1.Exploratory Data Analysis
  2.User–Item Matrix Construction
  3.User-Based Collaborative Filtering
  4.Item-Based Collaborative Filtering
  5.Matrix Factorization (SVD)
  6.Model Evaluation & Comparison


🤖 Models Used

User-Based Collaborative Filtering
• Finds users with similar rating behavior
• Recommends movies liked by similar users
• Advantages: Simple, Intuitive
• Limitations: Sparse data problem, Poor scalability

Item-Based Collaborative Filtering
• Finds movies with similar audience patterns
• Recommends movies similar to those user liked
• Advantages: More stable, Scales better, Widely used in industry
• Limitations: Cold-start for new movies

Matrix Factorization (SVD)
• Learns latent user preferences and movie features
• Predicts ratings using learned embeddings
• Advantages: Handles sparsity, High prediction accuracy, Production-grade technique
• Limitations: Computationally expensive, Requires training

Evaluation
Models were evaluated using Root Mean Square Error (RMSE).
RMSE measures prediction accuracy between actual ratings and predicted ratings. Lower RMSE
indicates better performance.

Results
  Model RMSE
  User-Based CF 1.096
  Item-Based CF 0.905
  SVD Add your value

Key Observations
• User-based CF suffers from sparse data
• Item-based CF significantly improves stability
• SVD provides best overall predictive performance

Tech Stack
  Programming: Python
  Libraries: Pandas, NumPy, Scikit-learn, Surprise, Matplotlib, Seaborn
  Tools: Jupyter Notebook, Git & GitHub
  
Project Structure
  movie-recommendation-system/
  • data/
  • notebooks/
  • outputs/
  • src/
  • README.md
  • requirements.txt
  
Future Improvements
• Hybrid recommendation system
• Deep learning-based recommenders
• Web application using Streamlit / Flask
• Cloud deployment
• Real-time recommendation API

Learning Outcomes
• Recommender system design
• Collaborative filtering algorithms
• Matrix factorization techniques
• Model evaluation strategies
• ML project structuring

How to Run the Project

  git clone
  cd movie-recommendation-system
  pip install -r requirements.txt
