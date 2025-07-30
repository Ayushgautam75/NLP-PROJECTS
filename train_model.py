from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Training data
texts = [
    "I feel happy today",
    "This is so sad",
    "I am full of anger",
    "I'm in love",
    "I'm scared right now",
    "Wow, that surprised me!"
]
labels = ['happy', 'sad', 'anger', 'love', 'fear', 'surprise']

# 2. TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 3. Train model (e.g. Naive Bayes)
model = MultinomialNB()
model.fit(X, labels)

# 4. Save both
joblib.dump(vectorizer, "vectorizer_tfidf.pkl")
joblib.dump(model, "emotion_model.pkl")
