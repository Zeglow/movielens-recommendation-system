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


    
    def calculate_user_similarity(self, user1_id:int, user2_id:int) -> float:
        user1_ratings = self.ratings_matrix.loc[user1_id] #.loc是什么
        user2_ratings = self.ratings_matrix.loc[user2_id]

        common_movies = (user1_ratings > 0) & (user2_ratings > 0)

        if not common_movies.any():
            return 0.0
        u1_common = user1_ratings[common_movies]
        u2_common = user2_ratings[common_movies]

        dot_product = np.dot(u1_common, u2_common)
        norm1 = np.linalg.norm(u1_common)
        norm2 = np.linalg.norm(u2_common)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def predict_rating(self, user_id:int, movie_id: int, k_neighbors: int = 10) -> float:
        """predict rating for a user-movie pair"""

        # cold start
        if user_id not in self.ratings_matrix.index:
            return 3.0
        # cold start
        if movie_id not in self.ratings_matrix.columns:
            return 3.0
        
        movie_raters = self.ratings_matrix[movie_id] > 0

        similar_users = []
        for other_user in self.ratings_matrix.index[movie_raters]:
             if other_user != user_id:
                  similarity = self.calculate_user_similarity(user_id, other_user)
                  if similarity > 0:
                       similar_users.append((other_user, similarity))
        similar_users.sort(key=lambda x: x[1], reverse = True)
        top_users = similar_users[:k_neighbors]

        if not top_users:
             return 3.0
        
        weighted_sum = 0
        similarity_sum = 0
        for other_user, similarity in top_users:
            rating = self.ratings_matrix.loc[other_user, movie_id]
            weighted_sum += similarity * rating 
            similarity_sum += similarity
        if similarity_sum == 0:
             return 3.0
        return weighted_sum / similarity_sum


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
        test_user = small_ratings['user_id'].iloc[3]
        test_movie = small_ratings['movie_id'].iloc[3]
        
        prediction = model.predict_rating(test_user, test_movie)
        print(f"Predicted rating for user {test_user}, movie {test_movie}: {prediction:.2f}")
    
if __name__ == "__main__":
        test_collaborative_filtering()

