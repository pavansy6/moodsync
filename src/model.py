from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class MusicRecommender:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.song_database = None
        self.feature_cols = ['valence', 'energy', 'tempo', 'danceability', 'acousticness']
        self.emotion_cols = ['sadness', 'joy', 'anger', 'fear', 'neutral']

    def train(self, df):
        self.song_database = df.copy()
        X = df[self.emotion_cols]
        Y = df[self.feature_cols]
        self.model.fit(X, Y)

    def recommend(self, emotion_vector, top_k=5):
        # Convert user emotion dictionary to DataFrame
        user_emotions = pd.DataFrame([emotion_vector])[self.emotion_cols]
        
        # Predict target audio features
        predicted_features = self.model.predict(user_emotions)[0]
        
        # Calculate similarity between predicted features and all songs in database
        db_features = self.song_database[self.feature_cols].values
        similarities = cosine_similarity([predicted_features], db_features)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return self.song_database.iloc[top_indices][['song', 'artist', 'valence', 'energy']]