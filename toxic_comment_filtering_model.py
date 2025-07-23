# save_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os

# Step 1: Load Dataset
df = pd.read_csv("train.csv")  # Ensure train.csv is in the same directory
df['comment_text'] = df['comment_text'].fillna("")

# Step 2: Select features and labels
X = df['comment_text']
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Step 5: Train the Multi-Label Classifier
model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train_tfidf, y_train)

# Step 6: Create a model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Step 7: Save model and vectorizer
joblib.dump(model, "model/toxic_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved to 'model/' folder.")
