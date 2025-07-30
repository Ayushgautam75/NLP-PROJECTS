import streamlit as st
import joblib


model = joblib.load("emotion_model.pkl")            # Trained classifier (e.g., Naive Bayes)
vectorizer = joblib.load("vectorizer_tfidf.pkl")    # TF-IDF vectorizer

def get_emoji(emotion):
    emojis = {
        'happy': '😊',
        'sad': '😢',
        'anger': '😠',
        'fear': '😨',
        'love': '❤️',
        'surprise': '😲',
        'neutral': '😐'
    }
    return emojis.get(str(emotion).lower(), '❓')

# Streamlit UI
st.set_page_config(page_title="NLP Emotion Detector ", page_icon="🧠", layout="centered")
st.title("🔍 NLP Emotion Detector by Ayush❤️")

text = st.text_area("Enter a message:")

if st.button("Predict"):
    if text.strip():
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        st.success(f"**Emotion:** {prediction} {get_emoji(prediction)}")
    else:
        st.warning("Please type something.")
