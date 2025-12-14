import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    with open("trained_spam_classifier_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📩 Spam Message Classifier")
st.write("Predict whether a message is *Spam* or *Not Spam*")

# -------------------------------
# Sample dataset (added inside code)
# -------------------------------
data = {
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hi, are we meeting tomorrow?",
        "URGENT! Call this num…
import streamlit as st
import pickle

# -------------------------------
# Load model & vectorizer
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("trained_spam_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# -------------------------------
# UI
# -------------------------------
st.title("📩 Spam Message Classifier")

user_input = st.text_area("Enter message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        # 🔥 FIX: Transform text before predict
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.error("🚨 SPAM message")
        else:
            st.success("✅ NOT SPAM")
