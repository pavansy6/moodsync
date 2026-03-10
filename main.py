import os
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.scraper import RedditScraper
from src.extractor import SongExtractor
from src.local_music_db import LocalMusicFetcher
from src.nlp_processor import EmotionAnalyzer

SUBREDDITS = [
    "Music", "depression", "Spotify", "LetsTalkMusic", 
    "indieheads", "popheads", "RnB", "hiphopheads", 
    "casualconversation", "sad"
]
OUTPUT_FILE = "data/final_dataset.csv"

def main():
    print("Starting the Mood-Aware Music Data Pipeline...")

    # 1. Scrape Reddit Data
    print(f"Scraping posts from: {', '.join(SUBREDDITS)}...")
    scraper = RedditScraper(subreddits=SUBREDDITS)
    
    # Run multiple specific queries to grab different batches of posts
    search_queries = ["song by", "listening to", "track", "lyrics", "reminds me of"]
    
    all_reddit_dfs = []
    for q in search_queries:
        print(f"  -> Searching for keyword: '{q}'...")
        df = scraper.fetch_posts(query=q, limit=100)
        all_reddit_dfs.append(df)
        
    # Combine all scraped posts and drop any duplicates
    reddit_df = pd.concat(all_reddit_dfs).drop_duplicates(subset=['title'])
    print(f"Scraped a total of {len(reddit_df)} unique posts.")

    # 2. Extract Songs
    print("Extracting song mentions...")
    extracted_data = []
    for _, row in reddit_df.iterrows():
        text_to_analyze = row['title'] + " " + row['text']
        songs_found = SongExtractor.extract_song_mentions(text_to_analyze)
        
        for song_info in songs_found:
            extracted_data.append({
                'reddit_text': text_to_analyze,
                'song': song_info['song'],
                'artist': song_info['artist']
            })
    
    song_df = pd.DataFrame(extracted_data)
    print(f"Found {len(song_df)} potential songs.")

    if song_df.empty:
        print("No songs found. Try increasing the scrape limit or tweaking the regex. Exiting.")
        return

    # 3. Fetch Local Audio Features (Replaces Spotify API)
    print("Fetching audio features from local database...")
    music_db = LocalMusicFetcher()
    
    music_data = []
    for _, row in song_df.iterrows():
        features = music_db.get_audio_features(row['song'], row['artist'])
        if features:
            features['reddit_text'] = row['reddit_text']
            music_data.append(features)
            
    final_df = pd.DataFrame(music_data)
    print(f"Matched {len(final_df)} songs.")

    # 4. Analyze Emotions in Reddit Text
    print("Analyzing text emotions using HuggingFace...")
    analyzer = EmotionAnalyzer()
    
    emotions_list = []
    for text in final_df['reddit_text']:
        emotions = analyzer.get_emotion_vector(text)
        emotions_list.append(emotions)
        
    emotions_df = pd.DataFrame(emotions_list)
    
    # 5. Combine and Save Dataset
    print("Saving final dataset...")
    final_dataset = pd.concat([final_df, emotions_df], axis=1)
    
    os.makedirs("data", exist_ok=True)
    final_dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"Pipeline complete! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()