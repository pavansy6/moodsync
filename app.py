import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.model import AcousticMoodRecommender

st.set_page_config(page_title="MoodSync Recommender", layout="wide")
st.title("MoodSync: Emotion-to-Acoustic Engine")
st.markdown("Powered by Llama 3 API and K-Nearest Neighbors.")

@st.cache_resource
def load_system():
    if not os.path.exists("data/strict_clean.csv"):
        st.error("Missing data/strict_clean.csv. Run python clean_dataset.py first.")
        st.stop()
    return AcousticMoodRecommender("data/strict_clean.csv")

with st.spinner("Initializing Acoustic Math Engine..."):
    recommender = load_system()

user_text = st.text_area("How are you feeling?", "I am totally burnt out and just want to lie in the dark.", height=100)

if st.button("Calculate Soundtrack", type="primary"):
    with st.spinner("Analyzing text via Groq LLM & mapping acoustic vectors..."):
        
        try:
            recommendations, raw_emotions = recommender.recommend(user_text, top_k=5)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Extracted Psychological Profile")
                emo_df = pd.DataFrame(list(raw_emotions.items()), columns=['Emotion', 'Probability'])
                fig_bar = px.bar(emo_df, x='Emotion', y='Probability', template='plotly_dark')
                fig_bar.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col2:
                st.subheader("Nearest Acoustic Matches")
                display_df = recommendations[['track_name', 'artists', 'match_accuracy', 'valence', 'energy']]
                display_df['match_accuracy'] = display_df['match_accuracy'].astype(str) + "%"
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"An error occurred: {e}. Check your API key and internet connection.")