import streamlit as st
import joblib

# Load saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("language_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# App layout
st.set_page_config(page_title="Language Detector", layout="centered")
st.title("🌐 Language Detection App")
st.write("Type any sentence and the model will detect its language.")

# User input
user_input = st.text_area("✍️ Enter your sentence here:", height=150)

# Prediction
if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        st.success(f"🔤 Detected Language: **{prediction[0]}**")
