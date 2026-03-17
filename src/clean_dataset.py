import pandas as pd
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Enable progress bar for pandas apply
tqdm.pandas()

def is_english(text):
    """Safely attempts to detect if a string is English."""
    try:
        # We only return True if the detected language is English
        return detect(str(text)) == 'en'
    except LangDetectException:
        # If it's just punctuation, numbers, or unreadable, discard it
        return False

def main():
    input_file = "data/dataset.csv"
    output_file = "data/cleaned_dataset.csv"
    
    print(f"Loading raw dataset from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}.")
        return

    df = pd.read_csv(input_file, low_memory=False)
    df.columns = df.columns.str.lower()
    initial_count = len(df)
    print(f"Initial track count: {initial_count}")

    # --- CLEANING PIPELINE ---

    # 1. Drop rows with missing essential data
    print("Dropping rows with missing values...")
    df = df.dropna(subset=['track_name', 'artists', 'valence', 'energy', 'speechiness', 'duration_ms'])
    
    # 2. Remove Duplicates
    print("Removing duplicate tracks...")
    df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
    
    # 3. Filter out spoken word / audiobooks (speechiness > 0.66 is usually spoken word)
    print("Filtering out podcasts and audiobooks...")
    df = df[df['speechiness'] < 0.6]
    
    # 4. Filter duration (Keep tracks between 1 minute and 10 minutes)
    print("Filtering out tracks that are too short or too long...")
    # duration_ms is in milliseconds (60000 ms = 1 minute, 600000 ms = 10 minutes)
    df = df[(df['duration_ms'] >= 60000) & (df['duration_ms'] <= 600000)]
    
    # 5. Language Detection (This takes the longest)
    print("Detecting languages and keeping only English tracks (This may take a few minutes)...")
    # We apply the detection to the track name
    df['is_en'] = df['track_name'].progress_apply(is_english)
    df = df[df['is_en'] == True]
    
    # Drop the temporary column
    df = df.drop(columns=['is_en'])

    # --- SAVE ---
    final_count = len(df)
    print("\n--- Cleaning Complete ---")
    print(f"Tracks removed: {initial_count - final_count}")
    print(f"Final clean track count: {final_count}")
    
    df.to_csv(output_file, index=False)
    print(f"Saved clean dataset to {output_file}")

if __name__ == "__main__":
    main()