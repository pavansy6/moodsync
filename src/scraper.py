import requests
import pandas as pd
import time
import spacy

nlp = spacy.load("en_core_web_sm")

class RedditScraper:
    def __init__(self, subreddits):
        self.subreddits = subreddits
        # A custom User-Agent is strictly required by Reddit's API rules
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MoodMusicBot/1.0'}

    def fetch_posts(self, query="music", limit=100):
        data = []
        for sub in self.subreddits:
            # Added &sort=top&t=all to get the highest quality, most detailed posts
            url = f"https://www.reddit.com/r/{sub}/search.json?q={query}&limit={limit}&restrict_sr=1&sort=top&t=all"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                posts = response.json().get('data', {}).get('children', [])
                for post in posts:
                    p = post['data']
                    data.append({
                        'subreddit': sub,
                        'title': p.get('title', ''),
                        'text': p.get('selftext', ''),
                        'upvotes': p.get('ups', 0),
                        'timestamp': p.get('created_utc', 0)
                    })
            time.sleep(1.5) # Slightly longer sleep to be safe with Reddit's API
        return pd.DataFrame(data)
