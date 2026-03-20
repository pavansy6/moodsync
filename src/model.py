import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AcousticMoodRecommender:
    def __init__(self, dataset_path="data/strict_clean.csv"):
        print("Initializing Acoustic Math Engine...")
        self.df = pd.read_csv(dataset_path)
        
        # Initialize Groq Client
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file.")
        self.client = Groq(api_key=api_key)
        
        # The features we care about mathematically
        self.features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
        
        # Scale the features so tempo (0-200) doesn't overpower valence (0-1)
        self.scaler = MinMaxScaler()
        self.scaled_audio_matrix = self.scaler.fit_transform(self.df[self.features])
        
        # Build KDTree for sub-second nearest-neighbor searches
        self.tree = KDTree(self.scaled_audio_matrix)

    def extract_emotion_via_llm(self, user_text):
        """Calls Groq API to return exact Ekman emotion probabilities as JSON."""
        system_prompt = """
        Analyze the emotional content of the user's text. 
        Output ONLY a valid JSON object scoring the text from 0.0 to 1.0 across these exact 7 categories:
        {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "neutral": 0.0}
        Do not include any other text, markdown formatting, or explanation.
        """
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON string returned by the LLM
        return json.loads(response.choices[0].message.content)

    def _map_emotion_to_acoustics(self, emotion_scores):
        """Maps psychological emotion probabilities directly to scaled Spotify features."""
        profiles = {
            'joy':      np.array([0.90, 0.85, 0.80, 0.10, 0.70]),
            'sadness':  np.array([0.10, 0.15, 0.20, 0.85, 0.20]),
            'anger':    np.array([0.15, 0.95, 0.40, 0.05, 0.85]),
            'fear':     np.array([0.25, 0.60, 0.30, 0.70, 0.60]),
            'surprise': np.array([0.70, 0.80, 0.60, 0.20, 0.75]),
            'disgust':  np.array([0.20, 0.70, 0.40, 0.30, 0.60]),
            'neutral':  np.array([0.50, 0.50, 0.50, 0.50, 0.50])
        }
        
        target_vector = np.zeros(5)
        for emotion, weight in emotion_scores.items():
            if emotion in profiles:
                target_vector += profiles[emotion] * float(weight)
                
        return target_vector

    def recommend(self, user_text, top_k=5):
        # 1. Ask LLM for emotion vector
        raw_emotions = self.extract_emotion_via_llm(user_text)
        
        # 2. Map emotions to ideal 5D acoustic coordinate
        ideal_scaled_features = self._map_emotion_to_acoustics(raw_emotions)
        
        # 3. Find closest mathematical matches
        distances, indices = self.tree.query([ideal_scaled_features], k=top_k)
        
        # 4. Format output
        recommendations = self.df.iloc[indices[0]].copy()
        
        max_dist = np.sqrt(len(self.features)) 
        match_percentages = (1 - (distances[0] / max_dist)) * 100
        recommendations['match_accuracy'] = match_percentages.round(1)
        
        return recommendations, raw_emotions