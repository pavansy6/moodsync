import pandas as pd
import os

def clean_dataset(input_path="data/dataset.csv", output_path="data/strict_clean.csv"):
    print("Loading raw Kaggle data...")
    if not os.path.exists(input_path):
        print(f"Error: Place your Kaggle dataset at {input_path}")
        return

    df = pd.read_csv(input_path, low_memory=False)
    df.columns = df.columns.str.lower()
    
    initial_count = len(df)
    
    # 1. Drop missing data
    df = df.dropna(subset=['track_name', 'artists', 'valence', 'energy', 'tempo', 'danceability', 'acousticness'])
    
    # 2. The ASCII Nuke: Remove ANY row with non-English/non-standard characters
    print("Purging non-ASCII tracks...")
    df = df[df['track_name'].astype(str).map(lambda x: x.isascii())]
    df = df[df['artists'].astype(str).map(lambda x: x.isascii())]
    
    # 3. Filter audiobooks/noise and extreme durations
    if 'speechiness' in df.columns:
        df = df[df['speechiness'] < 0.5]
    if 'duration_ms' in df.columns:
        df = df[(df['duration_ms'] >= 90000) & (df['duration_ms'] <= 420000)] # 1.5 to 7 mins
    
    # 4. Remove exact duplicates
    df = df.drop_duplicates(subset=['track_name', 'artists'])
    
    final_count = len(df)
    print(f"Purged {initial_count - final_count} garbage tracks.")
    print(f"Saved {final_count} pristine tracks to {output_path}.")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_dataset()