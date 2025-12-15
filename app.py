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
st.write("Predict whether a message is Spam or Not Spam")

# -------------------------------
# Sample dataset (added inside code)
# -------------------------------
data = {
    "message": [
        "Congratulations! You won a free lottery ticket",
        "Hi, are we meeting tomorrow?",
        "URGENT! Call this number to claim your prize",
        "Please review the attached document",
        "Win cash now!!! Limited offer"
    ]
}

df = pd.DataFrame(data)

st.subheader("📊 Sample Messages Dataset")
st.dataframe(df)

# -------------------------------
# User Input
# -------------------------------
st.subheader("✍️ Enter a message to classify")
user_input = st.text_area("Type your message here")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        prediction = model.predict([user_input])[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Developed using Streamlit & Machine Learning")
