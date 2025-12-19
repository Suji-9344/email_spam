import streamlit as st
import pickle
import numpy as np

st.title("📧 Email Spam Classification App")

# Load Model Safely
@st.cache_resource
def load_model():
    with open("model.pkl","rb") as file:
        return pickle.load(file)

model = load_model()

st.write("Enter email text to check whether it is Spam or Not Spam:")

email_text = st.text_area("✉️ Email Content")

if st.button("Predict"):
    if email_text.strip() == "":
        st.error("Please enter an email text")
    else:
        result = model.predict([email_text])[0]

        if result == 1:
            st.error("🚨 This is a SPAM Email")
        else:
            st.success("✔️ This is NOT a Spam Email")
