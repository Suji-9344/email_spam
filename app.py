import streamlit as st
import pickle
import pandas as pd
import os

# -------------------------------
# Load model & vectorizer safely
# -------------------------------
@st.cache_resource
def load_files():
    try:
        if not os.path.exists("trained_spam_classifier_model.pkl"):
            st.error("❌ trained_spam_classifier_model.pkl not found")
            st.stop()

        if not os.path.exists("vectorizer.pkl"):
            st.error("❌ vectorizer.pkl not found")
            st.stop()

        with open("trained_spam_classifier_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        st.error("❌ Failed to load model/vectorizer")
        st.exception(e)
        st.stop()

model, vectorizer = load_files()

# -------------------------------
# UI
# -------------------------------
st.title("📩 Spam Message Classifier")

# Sample data
df = pd.DataFrame({
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hi, are we meeting tomorrow?",
        "URGENT! Call this number to claim your prize"
    ]
})
st.dataframe(df)

# Input
user_input = st.text_area("✍️ Enter your message")

# Predict
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Enter a message")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")
