from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

def recommend(user_text, top_k=5):
    """Recommends moods based on the user's text input."""

    # Create a client
    google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))        

    response = google_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["You are a helpful assistant.",
                    user_text]
    )