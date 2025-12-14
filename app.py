import streamlit as st
import pickle

# Load model and vectorizer
@st.cache_resource
def load_files():
    with open("trained_spam_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_files()

st.title("📩 Email Spam Classifier")

user_input = st.text_area("Type your message here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")
