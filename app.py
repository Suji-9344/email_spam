import streamlit as st
import pickle

# App title
st.title("📧 Email Spam Detection App")
st.write("Check whether an email message is **Spam** or **Ham**")

# Load trained model and vectorizer
@st.cache_resource
def load_model():
    with open("trained_spam_classifier_model.pkl", "rb") as file:
        data = pickle.load(file)
    return data

model_data = load_model()

model = model_data["model"]
vectorizer = model_data["vectorizer"]

# User input
st.subheader("✉️ Enter Email Text")
email_text = st.text_area("Type your email message here")

# Prediction
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter an email message")
    else:
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]

        if prediction == 1:
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is NOT SPAM (Ham)")

