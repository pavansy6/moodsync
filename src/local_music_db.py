# import random

# class LocalMusicFetcher:
#     """
#     A drop-in replacement for the deprecated Spotify audio-features API.
#     Currently simulates acoustic data to unblock the data pipeline.
#     """
#     def __init__(self):
#         print("Initialized Local Music Database (Simulation Mode)")

#     def get_audio_features(self, song, artist):
#         # For pipeline testing, we generate plausible random acoustic features.
#         # Later, you will replace this logic with a pandas lookup against a Kaggle CSV.
#         return {
#             'song': song,
#             'artist': artist,
#             'valence': round(random.uniform(0.1, 0.9), 3),
#             'energy': round(random.uniform(0.1, 0.9), 3),
#             'tempo': round(random.uniform(70.0, 180.0), 1),
#             'danceability': round(random.uniform(0.2, 0.9), 3),
#             'acousticness': round(random.uniform(0.0, 0.9), 3)
#         }

import pandas as pd
import os

class LocalMusicFetcher:
    def __init__(self, db_path="data/dataset.csv"):
        print(f"Loading local Spotify database from {db_path}...")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Could not find the database at {db_path}. Please download it from Kaggle and place it in the data folder.")
            
        # Load the dataset into memory. 
        # We use low_memory=False because these datasets can be quite large.
        self.df = pd.read_csv(db_path, low_memory=False)
        
        # Standardize column names to lowercase just in case the Kaggle dataset uses uppercase
        self.df.columns = self.df.columns.str.lower()
        
        # Drop rows missing crucial song name or artist data
        # Note: Different Kaggle datasets name their artist column 'artist', 'artists', or 'track_artist'.
        # We will try to dynamically find the right ones.
        self.track_col = 'track_name' if 'track_name' in self.df.columns else 'name'
        self.artist_col = 'artists' if 'artists' in self.df.columns else 'artist'
        
        self.df = self.df.dropna(subset=[self.track_col, self.artist_col])
        print(f"Database loaded successfully with {len(self.df)} tracks available for searching.")

    def get_audio_features(self, song, artist):
        song_lower = str(song).lower()
        artist_lower = str(artist).lower()
        
        # Relaxed matching: Check if the extracted text is CONTAINED in the database columns
        match = self.df[
            (self.df[self.track_col].astype(str).str.lower().str.contains(song_lower, regex=False, na=False)) & 
            (self.df[self.artist_col].astype(str).str.lower().str.contains(artist_lower, regex=False, na=False))
        ]
        
        if not match.empty:
            track_data = match.iloc[0]
            required_features = ['valence', 'energy', 'tempo', 'danceability', 'acousticness']
            
            if all(feature in self.df.columns for feature in required_features):
                return {
                    'song': song, # Keep original extracted name for context
                    'artist': artist,
                    'valence': float(track_data['valence']),
                    'energy': float(track_data['energy']),
                    'tempo': float(track_data['tempo']),
                    'danceability': float(track_data['danceability']),
                    'acousticness': float(track_data['acousticness'])
                }
        
        return None
        
        # Return None if the song/artist combo wasn't found in the CSV
        return None