import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class MusicRecommender:
    def __init__(self, dataset_path="data/dataset.csv"):
        print("Loading Kaggle database...")
        # Load the dataset
        self.df = pd.read_csv(dataset_path, low_memory=False)
        self.df.columns = self.df.columns.str.lower()
        
        # Drop rows with missing values in our target columns
        self.feature_cols = ['valence', 'energy', 'danceability', 'acousticness']
        self.df = self.df.dropna(subset=self.feature_cols + ['track_name', 'artists'])
        
        # Fit the K-Nearest Neighbors model
        self.knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        self.knn.fit(self.df[self.feature_cols])

    def emotion_to_audio_features(self, emotion_vector):
        """
        Maps HuggingFace emotions (Joy, Sadness, Anger, etc.) to target Spotify features.
        """
        # Define baseline acoustic profiles for pure emotions
        profiles = {
            'joy':     {'valence': 0.85, 'energy': 0.80, 'danceability': 0.75, 'acousticness': 0.10},
            'sadness': {'valence': 0.15, 'energy': 0.20, 'danceability': 0.30, 'acousticness': 0.85},
            'anger':   {'valence': 0.20, 'energy': 0.90, 'danceability': 0.40, 'acousticness': 0.05},
            'fear':    {'valence': 0.30, 'energy': 0.50, 'danceability': 0.40, 'acousticness': 0.50},
            'neutral': {'valence': 0.50, 'energy': 0.50, 'danceability': 0.50, 'acousticness': 0.50}
        }

        # Calculate a weighted average of features based on the user's emotion scores
        target_features = {feature: 0.0 for feature in self.feature_cols}
        
        for emotion, weight in emotion_vector.items():
            if emotion in profiles:
                for feature in self.feature_cols:
                    target_features[feature] += profiles[emotion][feature] * weight
                    
        return target_features

    def recommend(self, emotion_vector, top_k=5):
        # 1. Get the ideal acoustic features for the user's mood
        target = self.emotion_to_audio_features(emotion_vector)
        target_array = np.array([[target['valence'], target['energy'], target['danceability'], target['acousticness']]])
        
        # 2. Find the closest matching songs in the Kaggle dataset
        distances, indices = self.knn.kneighbors(target_array, n_neighbors=top_k)
        
        # 3. Retrieve the song data
        recommendations = self.df.iloc[indices[0]].copy()
        
        # Add the target coordinates to the output so we can see what the model aimed for
        recommendations['target_valence'] = target['valence']
        recommendations['target_energy'] = target['energy']
        
        return recommendations