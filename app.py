import streamlit as st
from src.model import OnlineMoodRecommender

st.set_page_config(page_title="MoodSync V2", layout="centered")
st.title("🎧 MoodSync: Live Online Engine")
st.markdown("Powered by Llama 3 & The Live Spotify Catalog")

@st.cache_resource
def load_engine():
    return OnlineMoodRecommender()

try:
    recommender = load_engine()
except Exception as e:
    st.error(f"Setup Error: {e}. Please check your .env file.")
    st.stop()

user_text = st.text_area("How are you feeling right now?", "I feel like driving down an empty highway at midnight.", height=100)

if st.button("Generate Live Playlist", type="primary"):
    with st.spinner("Consulting Llama 3 and searching the live Spotify database..."):
        
        try:
            playlist = recommender.recommend(user_text)
            
            if not playlist:
                st.warning("Could not find exact matches on Spotify. Try a different mood!")
            else:
                st.subheader("Your Custom Soundtrack")
                st.divider()
                
                # Build a beautiful UI for each track
                for track in playlist:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if track['album_art']:
                            st.image(track['album_art'], use_column_width=True)
                            
                    with col2:
                        st.markdown(f"### {track['title']}")
                        st.markdown(f"**{track['artist']}**")
                        st.write(f"*{track['reason']}*")
                        
                        if track['preview_url']:
                            st.audio(track['preview_url'], format="audio/mp3")
                        else:
                            st.caption("No audio preview available for this specific track.")
                            
                        st.markdown(f"[Listen on Spotify]({track['spotify_url']})")
                        
                    st.divider()
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")