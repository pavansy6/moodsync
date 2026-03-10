import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

class SpotifyFetcher:
    def __init__(self, client_id, client_secret):
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def get_audio_features(self, song, artist):
        query = f"track:{song} artist:{artist}"
        results = self.sp.search(q=query, type='track', limit=1)
        
        if not results['tracks']['items']:
            return None
            
        track_id = results['tracks']['items'][0]['id']
        features = self.sp.audio_features(track_id)[0]
        
        if features:
            return {
                'song': song,
                'artist': artist,
                'valence': features['valence'],
                'energy': features['energy'],
                'tempo': features['tempo'],
                'danceability': features['danceability'],
                'acousticness': features['acousticness']
            }
        return None