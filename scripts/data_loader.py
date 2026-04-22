"""
Data loading and preprocessing for The Movies Dataset
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict

class MovieDataLoader:
    """Load and preprocess The Movies Dataset from Kaggle"""
    
    def __init__(self, data_path: str = '../data/'):
        self.data_path = Path(data_path)
        self.movies = None
        self.ratings = None
        self.credits = None
        self.keywords = None
    
    def load_all(self, use_small_ratings: bool = True) -> None:
        """Load all dataset files"""
        print("Loading datasets...")
        
        # Load movies metadata
        self.movies = pd.read_csv(self.data_path / 'movies_metadata.csv')
        print(f"✓ Loaded movies: {self.movies.shape}")
        
        # Load ratings (use small version for initial exploration)
        ratings_file = 'ratings_small.csv' if use_small_ratings else 'ratings.csv'
        try:
            self.ratings = pd.read_csv(self.data_path / ratings_file)
            print(f"✓ Loaded ratings ({ratings_file}): {self.ratings.shape}")
        except FileNotFoundError:
            print(f"✗ Ratings file not found: {ratings_file}")
        
        # Load credits
        try:
            self.credits = pd.read_csv(self.data_path / 'credits.csv')
            print(f"✓ Loaded credits: {self.credits.shape}")
        except FileNotFoundError:
            print("✗ Credits file not found")
        
        # Load keywords
        try:
            self.keywords = pd.read_csv(self.data_path / 'keywords.csv')
            print(f"✓ Loaded keywords: {self.keywords.shape}")
        except FileNotFoundError:
            print("✗ Keywords file not found")
    
    def explore_movies(self) -> None:
        """Print overview of movies dataset"""
        if self.movies is None:
            print("Movies not loaded. Run load_all() first.")
            return
        
        print("\n=== MOVIES DATASET ===")
        print(f"Shape: {self.movies.shape}")
        print(f"\nColumns: {self.movies.columns.tolist()}")
        print(f"\nData types:\n{self.movies.dtypes}")
        print(f"\nMissing values:\n{self.movies.isnull().sum()}")
        print(f"\nFirst row:\n{self.movies.iloc[0]}")
    
    def explore_ratings(self) -> None:
        """Print overview of ratings dataset"""
        if self.ratings is None:
            print("Ratings not loaded. Run load_all() first.")
            return
        
        print("\n=== RATINGS DATASET ===")
        print(f"Shape: {self.ratings.shape}")
        print(f"Columns: {self.ratings.columns.tolist()}")
        print(f"Rating distribution:\n{self.ratings['rating'].value_counts().sort_index()}")
        print(f"Date range: {self.ratings['timestamp'].min()} to {self.ratings['timestamp'].max()}")
    
    def get_movie_features(self) -> pd.DataFrame:
        """Extract and clean movie features for modeling"""
        df = self.movies.copy()
        
        # Convert budget and revenue to numeric
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        
        # Parse genres from JSON string
        def parse_genres(x):
            try:
                return [g['name'] for g in json.loads(x)]
            except:
                return []
        
        df['genres'] = df['genres'].apply(parse_genres)
        
        # Extract runtime
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
        
        # Extract year from release_date
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year
        
        # Select relevant columns
        features = df[['id', 'title', 'budget', 'revenue', 'runtime', 
                       'year', 'genres', 'vote_average', 'vote_count']].copy()
        
        # Remove rows with missing critical values
        features = features.dropna(subset=['budget', 'revenue', 'runtime', 'year'])
        
        # Filter for reasonable values
        features = features[(features['budget'] > 0) & (features['revenue'] > 0)]
        
        return features
    
    def get_rating_stats(self) -> pd.DataFrame:
        """Get average rating per movie"""
        if self.ratings is None:
            print("Ratings not loaded.")
            return None
        
        return self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count']
        }).round(2)


if __name__ == '__main__':
    # Example usage
    loader = MovieDataLoader()
    loader.load_all(use_small_ratings=True)
    
    loader.explore_movies()
    loader.explore_ratings()
    
    print("\n=== MOVIE FEATURES ===")
    features = loader.get_movie_features()
    print(f"Shape: {features.shape}")
    print(features.head())
