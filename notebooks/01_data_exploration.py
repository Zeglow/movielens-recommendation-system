"""
MovieLens Data Exploration

Basic analysis of the MovieLens dataset to understand:
- Data structure and quality
- User behavior patterns  
- Movie distribution
- Rating patterns
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import load_ratings
import pandas as pd

def explore_basic_stats():
    """Print basic dataset statistics"""
    print("=== Loading MovieLens Data ===")
    ratings = load_ratings()
    
    if ratings is None:
        return
    
    print(f"\n=== Basic Statistics ===")
    print(f"Total ratings: {len(ratings):,}")
    print(f"Unique users: {ratings['user_id'].nunique():,}")
    print(f"Unique movies: {ratings['movie_id'].nunique():,}")
    print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    
    return ratings
def explore_user_behavior(ratings):
    """Analyze user rating patterns"""
    print(f"\n=== User Behavior Analysis ===")
    
    # Ratings per user
    user_counts = ratings.groupby('user_id').size()
    print(f"Ratings per user - Mean: {user_counts.mean():.1f}, Median: {user_counts.median()}")
    print(f"Most active user rated {user_counts.max()} movies")
    print(f"Least active user rated {user_counts.min()} movies")
    
    # Rating distribution
    print(f"\n=== Rating Distribution ===")
    rating_dist = ratings['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / len(ratings)) * 100
        print(f"Rating {rating}: {count:,} ({percentage:.1f}%)")

if __name__ == "__main__":
    ratings = explore_basic_stats()
    if ratings is not None:
        explore_user_behavior(ratings)