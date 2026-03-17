import streamlit as st
import pandas as pd
import plotly.express as px
import os

from src.model import MusicRecommender

st.set_page_config(page_title="MoodSync Recommender", layout="wide")
st.title("MoodSync: Hugging Face Semantic Search")
st.markdown("Type out how you are feeling. Our AI will semantically match your mood to the vibe of 15,000 popular tracks.")

@st.cache_resource
def load_recommender():
    return MusicRecommender(dataset_path="data/dataset.csv") 

try:
    with st.spinner("Loading AI model and encoding music database... (This takes about a minute on startup)"):
        recommender = load_recommender()
except FileNotFoundError:
    st.error("Could not find data/dataset.csv.")
    st.stop()

# --- USER INPUT ---
user_text = st.text_area("How are you feeling right now?", "I feel totally exhausted and just want to lay in bed.", height=100)

if st.button("Find my Soundtrack", type="primary"):
    with st.spinner('Calculating semantic similarity...'):
        
        # Get Recommendations
        recommendations = recommender.recommend(user_text, top_k=5)
        
        # --- UI LAYOUT ---
        st.subheader("Top Matches based on Semantic Similarity")
        
        # Format the dataframe for clean display
        display_df = recommendations[['track_name', 'artists', 'track_genre', 'similarity_score', 'vibe_description']].copy()
        display_df['similarity_score'] = (display_df['similarity_score'] * 100).round(1).astype(str) + "%"
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Visualization in Acoustic Space
        st.divider()
        st.subheader("Where Your Recommendations Live")
        
        fig_scatter = px.scatter(
            recommendations, 
            x='valence', 
            y='energy', 
            hover_name='track_name', 
            hover_data=['artists', 'vibe_description'],
            color='similarity_score',
            color_continuous_scale='Viridis',
            size_max=15,
            template='plotly_dark'
        )
        fig_scatter.update_traces(marker=dict(size=14))
        fig_scatter.update_xaxes(range=[0, 1], title="Valence (Sad -> Happy)")
        fig_scatter.update_yaxes(range=[0, 1], title="Energy (Calm -> Intense)")
        
        st.plotly_chart(fig_scatter, use_container_width=True)