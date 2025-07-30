"""
    User-based CF
"""

import pandas as pd
import numpy as np
from typing import List,Tuple

class UserBasedCF:

    def __init__(self):
        self.ratings_matrix = None
        self.user_similarity = None
    
    def fit(self, ratings_df: pd.DataFrame):
        '''
        Train the CF model
        
        '''
        print("Building user-item matrix...")
        self.ratings_matrix = ratings_df.pivot_table(index = 'user_id', columns = 'movie_id', values = 'rating').fillna(0)

        print(f"Matrix shape:{self.ratings_matrix.shape}")
        print("Model training completed!")

    def predict_rating(self, user_id:int, movie_id: int) -> float:
        """predict rating for a user-movie pair"""

        # cold start
        if user_id not in self.ratings_matrix.index:
            return 3.0
        # cold start
        if movie_id not in self.ratings_matrix.columns:
            return 3.0
        user_ratings = self.ratings_matrix.loc[user_id]
        user_avg = user_ratings[user_ratings > 0].mean()
        
        return user_avg if not pd.isna(user_avg) else 3.0
    

def test_collaborative_filtering():
        """Test the collaborative filtering model"""
        # Load data (copy the function from data_loader)
        ratings_file = 'data/raw/ml-1m/ratings.dat'
        ratings = pd.read_csv(ratings_file, sep='::', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            engine='python')
        
        print(f"Loaded {len(ratings)} ratings")
        
        # Use subset for testing
        small_ratings = ratings.head(10000)
        
        # Create and train model
        model = UserBasedCF()
        model.fit(small_ratings)
        
        # Test prediction
        test_user = small_ratings['user_id'].iloc[0]
        test_movie = small_ratings['movie_id'].iloc[0]
        
        prediction = model.predict_rating(test_user, test_movie)
        print(f"Predicted rating for user {test_user}, movie {test_movie}: {prediction:.2f}")
    
if __name__ == "__main__":
        test_collaborative_filtering()
