import os
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class OnlineMoodRecommender:
    def __init__(self):
        print("Initializing V2 Online Engine...")
        
        # 1. Initialize Groq (The Brain)
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY missing from .env")
        self.groq_client = Groq(api_key=groq_key)
        
        # 2. Initialize Spotify (The Catalog)
        spot_id = os.environ.get("SPOTIPY_CLIENT_ID")
        spot_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
        if not spot_id or not spot_secret:
            raise ValueError("Spotify credentials missing from .env")
            
        auth_manager = SpotifyClientCredentials(client_id=spot_id, client_secret=spot_secret)
        self.spotify = spotipy.Spotify(auth_manager=auth_manager)

    def get_llm_recommendations(self, user_text):
        """Asks Llama 3 to act as an expert DJ and return exactly 5 songs."""
        system_prompt = """
        You are an expert, highly empathetic music curator. Analyze the user's emotional state from their text.
        Recommend exactly 5 real, well-known songs that perfectly match their exact emotional vibe.
        
        You MUST respond in strict JSON format matching this exact schema:
        {
            "recommendations": [
                {"song": "Title", "artist": "Artist Name", "reason": "A short 1-sentence explanation of why it fits"}
            ]
        }
        Do not include any other text or markdown outside the JSON object.
        """
        
        response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama3-8b-8192",
            temperature=0.5, # slightly higher temperature for creative song choices
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content).get("recommendations", [])

    def enrich_with_spotify(self, llm_songs):
        """Searches Spotify for the LLM's suggested songs to get album art and audio."""
        enriched_tracks = []
        
        for item in llm_songs:
            # search Spotify using the track and artist name
            query = f"track:{item['song']} artist:{item['artist']}"
            results = self.spotify.search(q=query, type='track', limit=1)
            
            tracks = results.get('tracks', {}).get('items', [])
            
            if tracks:
                track = tracks[0]
                enriched_tracks.append({
                    "title": track['name'],
                    "artist": track['artists'][0]['name'],
                    "album_art": track['album']['images'][0]['url'] if track['album']['images'] else None,
                    "preview_url": track['preview_url'], # The 30-second MP3
                    "spotify_url": track['external_urls']['spotify'],
                    "reason": item['reason']
                })
                
        return enriched_tracks

    def recommend(self, user_text):
        # Step 1: get the semantic matches from Groq
        llm_songs = self.get_llm_recommendations(user_text)
        
        # Step 2: grab the live data from Spotify
        final_playlist = self.enrich_with_spotify(llm_songs)
        
        return final_playlist