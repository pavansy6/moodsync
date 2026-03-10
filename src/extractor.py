import re
import spacy

nlp = spacy.load("en_core_web_sm")

class SongExtractor:
    @staticmethod
    def extract_song_mentions(text):
        songs = []
        # Case-insensitive, allows standard punctuation
        pattern1 = re.compile(r'([a-zA-Z0-9\s\',]+)\s+by\s+([a-zA-Z0-9\s\',]+)', re.IGNORECASE)
        pattern2 = re.compile(r'([a-zA-Z0-9\s\',]+)\s+-\s+([a-zA-Z0-9\s\',]+)', re.IGNORECASE)
        
        for match in pattern1.finditer(text):
            song = match.group(1).strip()
            artist = match.group(2).strip()
            if len(song) < 30 and len(artist) < 30: # Prevent capturing massive paragraphs
                songs.append({'song': song, 'artist': artist})
            
        for match in pattern2.finditer(text):
            artist = match.group(1).strip()
            song = match.group(2).strip()
            if len(song) < 30 and len(artist) < 30:
                songs.append({'song': song, 'artist': artist})
            
        return songs