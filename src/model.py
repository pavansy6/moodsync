import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MusicRecommender:
    def __init__(self, dataset_path="data/dataset.csv"):
        print("Loading Kaggle database and Hugging Face model...")
        
        # 1. Load the dataset
        self.df = pd.read_csv(dataset_path, low_memory=False)
        self.df.columns = self.df.columns.str.lower()
        
        # 2. Filter data for performance and quality
        # We only take the most popular 15,000 tracks to keep embedding times reasonable on a CPU
        if 'popularity' in self.df.columns:
            self.df = self.df[self.df['popularity'] > 40].copy()
        self.df = self.df.head(15000).reset_index(drop=True)
        
        # Drop rows with missing values
        self.df = self.df.dropna(subset=['valence', 'energy', 'track_name', 'artists'])
        
        # 3. Generate "Vibe Descriptions" for each track
        print("Synthesizing track descriptions...")
        self.df['vibe_description'] = self.df.apply(self._generate_vibe_text, axis=1)
        
        # 4. Load the Hugging Face Sentence Transformer
        # This model is specifically optimized for semantic similarity matching
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 5. Pre-compute the embeddings for all tracks
        print("Encoding track embeddings (this may take 30-60 seconds on first run)...")
        self.track_embeddings = self.model.encode(self.df['vibe_description'].tolist(), show_progress_bar=True)

    def _generate_vibe_text(self, row):
        """Translates acoustic math back into descriptive text for the AI to read."""
        v = row['valence']
        e = row['energy']
        
        # Valence labels
        if v > 0.7: mood = "happy, joyful, and positive"
        elif v < 0.3: mood = "sad, melancholic, and emotional"
        else: mood = "neutral and balanced"
            
        # Energy labels
        if e > 0.7: intensity = "highly energetic, loud, and intense"
        elif e < 0.3: intensity = "calm, slow, and relaxing"
        else: intensity = "moderately paced"
            
        genre = str(row.get('track_genre', 'music'))
        
        return f"A {mood} {genre} track. It is {intensity}."

    def recommend(self, user_text, top_k=5):
        # 1. Convert the user's emotional text into a vector
        user_embedding = self.model.encode([user_text])
        
        # 2. Calculate cosine similarity between the user text and all track descriptions
        similarities = cosine_similarity(user_embedding, self.track_embeddings)[0]
        
        # 3. Get the top K matching indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 4. Return the recommended rows
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations