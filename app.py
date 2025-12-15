[1:08 PM, 12/15/2025] Suji🥰: import streamlit as st
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
st.write("Predict whether a message is Spam or Not Spam")

# -------------------------------
# Sample dataset (added inside code)
# -------------------------------
data = {
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hi, are we meeting tomorrow?",
        "URGENT! Call this number …
[1:58 PM, 12/15/2025] Suji🥰: import streamlit as st
import pickle

# Load model and vectorizer
with open("email_spam_classifier_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
vectorizer = data["vectorizer"]

st.title("📧 Email Spam Classifier")

user_input = st.text_area("Enter email message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an email message")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1 or prediction == "spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM (HAM)")
