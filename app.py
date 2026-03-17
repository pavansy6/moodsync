import streamlit as st
import pandas as pd
import plotly.express as px
from src.model import AcousticMoodRecommender

st.set_page_config(page_title="MoodSync: Acoustic Engine", layout="wide")
st.title("MoodSync: Acoustic Math Engine")

@st.cache_resource
def load_system():
    return AcousticMoodRecommender("data/strict_clean.csv")

with st.spinner("Spinning up the Acoustic Math Engine..."):
    recommender = load_system()

user_text = st.text_area("How are you feeling?", "I am incredibly angry and need to break something.")

if st.button("Calculate Soundtrack", type="primary"):
    with st.spinner("Processing emotional vectors..."):
        
        # Run the engine
        recommendations, raw_emotions = recommender.recommend(user_text, top_k=5)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Psychological Profile")
            emo_df = pd.DataFrame(raw_emotions).rename(columns={'label': 'Emotion', 'score': 'Probability'})
            fig_bar = px.bar(emo_df, x='Emotion', y='Probability', template='plotly_dark')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.subheader("Acoustic Matches")
            display_df = recommendations[['track_name', 'artists', 'match_accuracy', 'valence', 'energy']]
            display_df['match_accuracy'] = display_df['match_accuracy'].astype(str) + "%"
            st.dataframe(display_df, use_container_width=True, hide_index=True)