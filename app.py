import streamlit as st
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
