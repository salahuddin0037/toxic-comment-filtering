import joblib
import re
import emoji
import contractions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For lemmatization


class ToxicCommentClassifier:
    def __init__(self):
        """Initialize the classifier by loading models"""
        try:
            # Load model and vectorizer
            self.model = joblib.load('models/toxic_model.pkl')
            self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            
            # Define gaming slang dictionary
            self.gaming_slang = {
                "kys": "kill yourself", "stfu": "shut the fuck up", 
                "gg": "good game", "glhf": "good luck have fun",
                "noob": "new player", "rekt": "wrecked",
                "pwned": "dominated", "l2p": "learn to play",
                "git gud": "get good", "afk": "away from keyboard"
            }
            
            # Define toxicity labels
            self.labels = [
                'toxic', 'severe_toxic', 'obscene', 
                'threat', 'insult', 'identity_hate'
            ]
            
        except Exception as e:
            raise Exception(f"Failed to initialize classifier: {str(e)}")
    
    def preprocess_text(self, text):
        """Preprocess the input text"""
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Fix contractions
            text = contractions.fix(text)
            
            # Remove emojis
            text = emoji.replace_emoji(text, replace='')
            
            # Replace gaming slang
            words = text.split()
            words = [self.gaming_slang.get(word, word) for word in words]
            text = ' '.join(words)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove repeated characters (e.g., noooooob -> noob)
            text = re.sub(r'(.)\1{2,}', r'\1', text)
            
            # Remove excessive punctuation
            text = re.sub(r'[!?]{2,}', '', text)
            
            # Tokenize and remove stopwords
            tokens = text.split()
            tokens = [word for word in tokens if word not in self.stop_words]
            
            # Stem words
            tokens = [self.stemmer.stem(word) for word in tokens]
            
            return ' '.join(tokens)
        
        except Exception as e:
            raise Exception(f"Text preprocessing failed: {str(e)}")
    
    def predict(self, text):
        """Make toxicity predictions on the input text"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Vectorize the text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Make prediction
            probabilities = self.model.predict_proba(text_vector)
            
            # Format results
            results = {}
            for i, label in enumerate(self.labels):
                # Get probability for positive class (toxic)
                results[label] = probabilities[i][0][1]
            
            return results
        
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

# For testing the module directly
if __name__ == "__main__":
    classifier = ToxicCommentClassifier()
    test_text = "kys you're a worthless noob!"
    print("Testing classifier with:", test_text)
    print("Results:", classifier.predict(test_text))