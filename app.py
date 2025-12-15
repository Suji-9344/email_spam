import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load model & vectorizer
# -------------------------------
@st.cache_resource
def load_files():
    with open("trained_spam_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_files()

# -------------------------------
# UI
# -------------------------------
st.title("📩 Spam Message Classifier")
st.write("Predict whether a message is Spam or Not Spam")

# -------------------------------
# Sample dataset
# -------------------------------
df = pd.DataFrame({
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hi, are we meeting tomorrow?",
        "URGENT! Call this number to claim your prize",
        "Please review the attached document",
        "Win cash now!!! Limited offer"
    ]
})

st.subheader("📊 Sample Messages")
st.dataframe(df)

# -------------------------------
# User input
# -------------------------------
user_input = st.text_area("✍️ Enter your message")

# -------------------------------
# Prediction (FIXED)
# -------------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        input_vector = vectorizer.transform([user_input])  # ✅ REQUIRED
        prediction = model.predict(input_vector)[0]        # ✅ FIXED

        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")

st.markdown("---")
st.markdown("Developed using Streamlit & Machine Learning")
