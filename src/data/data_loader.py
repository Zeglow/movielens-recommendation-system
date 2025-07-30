import pandas as pd
import os

def load_ratings():
    """Load MovieLens 1M ratings data"""
    ratings_file = 'data/raw/ml-1m/ratings.dat'
    
    if not os.path.exists(ratings_file):
        print(f"File not found: {ratings_file}")
        print("Please download ml-1m.zip and extract to data/raw/")
        return None
    
    ratings = pd.read_csv(ratings_file, sep='::', 
                         names=['user_id', 'movie_id', 'rating', 'timestamp'],
                         engine='python')
    print(f"Loaded {len(ratings)} ratings")
    return ratings

# 在文件末尾添加这几行
if __name__ == "__main__":
    print("Testing data loader...")
    ratings = load_ratings()
    if ratings is not None:
        print("Success! First 5 rows:")
        print(ratings.head())