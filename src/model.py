import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MusicRecommender:
    def __init__(self, dataset_path="data/dataset.csv"):
        # Define paths for our cached artifacts
        self.embeddings_path = "data/track_embeddings.npy"
        self.df_path = "data/filtered_tracks.pkl"
        
        # Always load the model first
        print("Loading Hugging Face model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Check if we have already generated and cached the embeddings
        if os.path.exists(self.embeddings_path) and os.path.exists(self.df_path):
            print("Found cached artifacts. Loading embeddings and dataframe from disk...")
            self.df = pd.read_pickle(self.df_path)
            self.track_embeddings = np.load(self.embeddings_path)
            print(f"Successfully loaded {len(self.df)} tracks instantly.")
            
        else:
            print("No cache found. Processing Kaggle database and generating embeddings...")
            
            # 1. Load the raw dataset
            self.df = pd.read_csv(dataset_path, low_memory=False)
            self.df.columns = self.df.columns.str.lower()
            
            # 2. Filter data for performance and quality
            if 'popularity' in self.df.columns:
                self.df = self.df[self.df['popularity'] > 40].copy()
            self.df = self.df.head(15000).reset_index(drop=True)
            self.df = self.df.dropna(subset=['valence', 'energy', 'track_name', 'artists'])
            
            # 3. Generate "Vibe Descriptions"
            print("Synthesizing track descriptions...")
            self.df['vibe_description'] = self.df.apply(self._generate_vibe_text, axis=1)
            
            # 4. Pre-compute the embeddings
            print("Encoding track embeddings (this will take 30-60 seconds)...")
            self.track_embeddings = self.model.encode(self.df['vibe_description'].tolist(), show_progress_bar=True)
            
            # 5. Cache the artifacts to disk for future runs
            print("Saving artifacts to disk...")
            self.df.to_pickle(self.df_path)
            np.save(self.embeddings_path, self.track_embeddings)
            print("Caching complete!")

    def _generate_vibe_text(self, row):
        v = row['valence']
        e = row['energy']
        
        if v > 0.7: mood = "happy, joyful, and positive"
        elif v < 0.3: mood = "sad, melancholic, and emotional"
        else: mood = "neutral and balanced"
            
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