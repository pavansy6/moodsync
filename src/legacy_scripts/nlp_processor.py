from transformers import pipeline
import spacy

nlp = spacy.load("en_core_web_sm")

class EmotionAnalyzer:
    def __init__(self):
        # Using a model specifically trained for Ekman's emotions
        self.classifier = pipeline("text-classification", 
                                   model="j-hartmann/emotion-english-distilroberta-base", 
                                   top_k=None) # Returns scores for all labels

    def clean_text(self, text):
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop and token.is_alpha]
        return " ".join(tokens)

    def get_emotion_vector(self, text):
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {e: 0.0 for e in ['sadness', 'joy', 'anger', 'fear', 'neutral']}
            
        predictions = self.classifier(cleaned_text[:512])[0] # Limit to 512 tokens
        
        # Format output into a dictionary
        emotion_dict = {pred['label']: pred['score'] for pred in predictions}
        return emotion_dict