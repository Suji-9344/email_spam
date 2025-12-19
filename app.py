import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam + Threat Detector")
st.title("📩 SMS Spam / Threat Message Classifier")

# -----------------------------------------
# DATASET AT TOP (VISIBLE)
# -----------------------------------------
data = {
    "label": [
        "ham","ham","spam","spam","spam","spam","ham","ham","spam","spam",
        "spam","spam","spam","ham","ham","spam","spam","spam","ham","ham",
        "spam","spam","spam","spam","spam","ham","ham","ham","spam","spam"
    ],
    "message": [
        "Hi how are you",
        "Are we meeting tomorrow?",
        "Congratulations! You won a lottery",
        "Claim your free reward now",
        "Win iPhone click link",
        "Free prize waiting for you",
        "Let's go to college",
        "Happy birthday",
        "URGENT! Claim cash now",
        "Final reminder your prize",

        # Threat / Violence
        "I will kill you",
        "You will be dead soon",
        "We will attack you",
        "I hate you",
        "Don't message me again",
        "Modiji was dead",
        "I will destroy you",
        "Bomb blast warning",

        "See you soon",
        "Good morning",

        # More spam
        "free recharge offer"
    ]
