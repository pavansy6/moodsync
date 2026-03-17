import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree

class AcousticMoodRecommender:
    def __init__(self, dataset_path="data/strict_clean.csv"):
        print("Loading clean dataset and mathematical scalers...")
        self.df = pd.read_csv(dataset_path)
        
        self.features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
        
        self.scaler = MinMaxScaler()
        self.scaled_audio_matrix = self.scaler.fit_transform(self.df[self.features])
        
        self.tree = KDTree(self.scaled_audio_matrix)
        
        print("Loading Emotion Classifier...")
        # Returns 7 Ekman emotions: anger, disgust, fear, joy, neutral, sadness, surprise
        self.emotion_model = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base", 
            top_k=None
        )

    def _map_emotion_to_acoustics(self, emotion_scores):
        """
        Maps psychological emotion probabilities directly to scaled Spotify features.
        """
        # Define the acoustic profile of pure emotions (0.0 to 1.0 scale)
        # [valence, energy, danceability, acousticness, tempo]
        profiles = {
            'joy':      np.array([0.90, 0.85, 0.80, 0.10, 0.70]),
            'sadness':  np.array([0.10, 0.15, 0.20, 0.85, 0.20]),
            'anger':    np.array([0.15, 0.95, 0.40, 0.05, 0.85]),
            'fear':     np.array([0.25, 0.60, 0.30, 0.70, 0.60]),
            'surprise': np.array([0.70, 0.80, 0.60, 0.20, 0.75]),
            'disgust':  np.array([0.20, 0.70, 0.40, 0.30, 0.60]),
            'neutral':  np.array([0.50, 0.50, 0.50, 0.50, 0.50])
        }
        
        # Calculate the weighted average target vector based on the user's text
        target_vector = np.zeros(5)
        for score_dict in emotion_scores:
            emotion = score_dict['label']
            weight = score_dict['score']
            target_vector += profiles[emotion] * weight
            
        return target_vector

    def recommend(self, user_text, top_k=5):
        # 1. Extract exact emotion probabilities from text
        raw_emotions = self.emotion_model(user_text[:512])[0]
        
        # 2. Map emotions to an ideal 5D acoustic coordinate
        ideal_scaled_features = self._map_emotion_to_acoustics(raw_emotions)
        
        # 3. Find the closest mathematical matches in our scaled dataset
        distances, indices = self.tree.query([ideal_scaled_features], k=top_k)
        
        # 4. Format the output
        recommendations = self.df.iloc[indices[0]].copy()
        
        # Calculate a simple "Match %" based on mathematical distance
        max_dist = np.sqrt(len(self.features)) # Maximum possible Euclidean distance in a scaled 5D space
        match_percentages = (1 - (distances[0] / max_dist)) * 100
        recommendations['match_accuracy'] = match_percentages.round(1)
        
        return recommendations, raw_emotions