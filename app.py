import streamlit as st
import pandas as pd
import plotly.express as px
import os

from src.nlp_processor import EmotionAnalyzer
from src.model import MusicRecommender

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mood-Aware Music", layout="wide")
st.title("Mood-Aware Music Discovery")
st.markdown("Enter how you are feeling, and we will find the acoustic features that match your emotional state.")

# --- CACHING HEAVY OPERATIONS ---
@st.cache_resource
def load_analyzer():
    return EmotionAnalyzer()

@st.cache_data
def load_data():
    file_path = "data/final_dataset.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

@st.cache_resource
def train_recommender(_df):
    recommender = MusicRecommender()
    recommender.train(_df)
    return recommender

# --- APP LOGIC ---
df = load_data()

if df is None:
    st.warning("Dataset not found. Please run `python main.py` first to generate the music dataset.")
    st.stop()

# Initialize NLP and Model
analyzer = load_analyzer()
recommender = train_recommender(df)

# --- USER INPUT ---
user_text = st.text_area("How are you feeling today?", "I feel empty, tired, and a bit lost in my thoughts.", height=100)

if st.button("Find my Soundtrack", type="primary"):
    with st.spinner('Analyzing emotions and searching the database...'):
        
        # 1. Get Emotion Vector
        emotion_vector = analyzer.get_emotion_vector(user_text)
        
        # 2. Get Recommendations
        recommendations = recommender.recommend(emotion_vector, top_k=5)
        
        # --- UI LAYOUT ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Emotional Signature")
            # Format emotion vector for plotting
            emotion_df = pd.DataFrame(list(emotion_vector.items()), columns=['Emotion', 'Score'])
            fig_bar = px.bar(emotion_df, x='Emotion', y='Score', color='Emotion', template='plotly_dark')
            fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.subheader("Recommended Tracks")
            st.dataframe(
                recommendations[['song', 'artist', 'valence', 'energy']], 
                use_container_width=True,
                hide_index=True
            )

        # 3. Visualization in Acoustic Space
        st.divider()
        st.subheader("Where Your Music Lives (Acoustic Space)")
        st.markdown("**Valence** (Musical Positivity) vs **Energy** (Intensity)")
        
        # Plot all songs in the background, highlight recommended ones
        df['Is_Recommended'] = df['song'].isin(recommendations['song'])
        
        fig_scatter = px.scatter(
            df, 
            x='valence', 
            y='energy', 
            hover_name='song', 
            hover_data=['artist'],
            color='Is_Recommended',
            color_discrete_map={True: '#1DB954', False: '#555555'}, # Spotify Green for recommendations
            opacity=0.8,
            template='plotly_dark'
        )
        
        # Make the recommended dots larger
        fig_scatter.update_traces(marker=dict(size=12), selector=dict(name="True"))
        fig_scatter.update_traces(marker=dict(size=6), selector=dict(name="False"))
        
        st.plotly_chart(fig_scatter, use_container_width=True)