import streamlit as st
import pandas as pd
import plotly.express as px
import os

from src.nlp_processor import EmotionAnalyzer
from src.model import MusicRecommender

# --- PAGE CONFIG ---
st.set_page_config(page_title="MoodSync Recommender", layout="wide")
st.title("MoodSync: Emotion to Music Translation")
st.markdown("Type out how you are feeling, and we will search 114,000 songs to find the perfect acoustic match.")

# --- CACHING ---
@st.cache_resource
def load_analyzer():
    return EmotionAnalyzer()

@st.cache_resource
def load_recommender():
    # Make sure this points to your exact Kaggle csv name
    return MusicRecommender(dataset_path="data/dataset.csv") 

# Initialize 
analyzer = load_analyzer()
try:
    recommender = load_recommender()
except FileNotFoundError:
    st.error("Could not find data/dataset.csv. Please ensure your Kaggle dataset is in the data folder.")
    st.stop()

# --- USER INPUT ---
user_text = st.text_area("How are you feeling right now?", "I feel totally exhausted and a little bit sad today.", height=100)

if st.button("Find my Soundtrack", type="primary"):
    with st.spinner('Translating emotions to acoustics...'):
        
        # 1. Get Emotion Vector
        emotion_vector = analyzer.get_emotion_vector(user_text)
        
        # 2. Get Recommendations
        recommendations = recommender.recommend(emotion_vector, top_k=5)
        
        # --- UI LAYOUT ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Emotional Signature")
            emotion_df = pd.DataFrame(list(emotion_vector.items()), columns=['Emotion', 'Score'])
            fig_bar = px.bar(emotion_df, x='Emotion', y='Score', color='Emotion', template='plotly_dark')
            fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.subheader("Top Matches from 114k Tracks")
            st.dataframe(
                recommendations[['track_name', 'artists', 'track_genre', 'valence', 'energy']], 
                use_container_width=True,
                hide_index=True
            )

        # 3. Visualization in Acoustic Space
        st.divider()
        st.subheader("Where Your Mood Lives (Acoustic Space)")
        
        # Plot the recommended songs
        fig_scatter = px.scatter(
            recommendations, 
            x='valence', 
            y='energy', 
            hover_name='track_name', 
            hover_data=['artists'],
            color_discrete_sequence=['#1DB954'],
            size_max=15,
            template='plotly_dark'
        )
        fig_scatter.update_traces(marker=dict(size=12))
        
        # Add a star for the "Ideal Target" mood coordinates
        target_v = recommendations['target_valence'].iloc[0]
        target_e = recommendations['target_energy'].iloc[0]
        fig_scatter.add_scatter(x=[target_v], y=[target_e], mode='markers', 
                                marker=dict(symbol='star', size=20, color='gold'), 
                                name='Your Exact Mood Target')
        
        # Set axes to 0-1 so we see the full Spotify spectrum
        fig_scatter.update_xaxes(range=[0, 1], title="Valence (Sad -> Happy)")
        fig_scatter.update_yaxes(range=[0, 1], title="Energy (Calm -> Intense)")
        
        st.plotly_chart(fig_scatter, use_container_width=True)