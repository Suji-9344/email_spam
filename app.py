import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Spam Classifier")
st.title("📩 SMS Spam Classifier")
st.write("This app predicts whether a message is **SPAM** or **NOT SPAM**")

# -----------------------------------------
# Dataset
# -----------------------------------------
data = {
    "label": [
        "ham","ham","spam","ham","spam","spam","ham","spam","ham","ham",
        "spam","ham","spam","ham","ham","spam","spam","ham","ham","spam",
        "spam","ham","spam","ham","spam","ham","ham","spam","ham","spam"
    ],
    "message": [
        "Hey, are you coming tomorrow?",
        "Lets have lunch today",
        "Congratulations! You won a free lottery",
        "I will call you later",
        "URGENT!!! Claim your prize now",
        "Win a brand new iPhone click here",
        "Don't forget our meeting",
        "Free entry in 2 crore contest",
        "See you soon",
        "How are you?",
        "You have been selected for cash reward",
        "Can we talk now?",
        "Call this number to get your prize",
        "Happy birthday!",
        "Good night",
        "Hurry! Only few hours left",
        "Exclusive offer just for you",
        "Where are you now?",
        "Let's study together",
        "Claim your reward now!",
        "WIN cash now click here",
        "Meet me at 6pm",
        "Free recharge offer limited time",
        "Call me when free",
        "You won 500000 cash prize",
        "Good morning",
        "Shall we go out today?",
        "FREE vacation package claim now",
        "See you in class",
        "Final reminder – prize waiting!"
    ]
}

df = pd.DataFrame(data)

with st.expander("📊 View Dataset"):
    st.dataframe(df)

# -----------------------------------------
# Preprocess
# -----------------------------------------
df["label_num"] = df["label"].map({"ham":0, "spam":1})

X = df["message"]
y = df["label_num"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Model
model = MultinomialNB()
model.fit(X_vec, y)

# -----------------------------------------
# Model Evaluation
# -----------------------------------------
st.subheader("📈 Model Performance")
pred = model.predict(X_vec)
st.text(classification_report(y, pred))

# -----------------------------------------
# User Input
# -----------------------------------------
st.subheader("✍️ Enter Message to Check")
user_input = st.text_area("Type message here...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        text = vectorizer.transform([user_input])
        result = model.predict(text)[0]

        if result == 1:
            st.error("🚨 SPAM Message Detected")
        else:
            st.success("✅ NOT SPAM Message")
