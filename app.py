import streamlit as st
import pickle

# Load model
with open("trained_spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📧 Email Spam Classifier")

user_input = st.text_area("Enter email message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter email text")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1 or prediction == "spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM (HAM)")

