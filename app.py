import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Spam Classifier")
st.title("📩 SMS Spam Classifier")
st.write("This app predicts whether a message is SPAM or NOT SPAM")

# ----------------------------
# Dataset (Built-In)
# ----------------------------
data = {
    "label": ["ham","ham","spam","ham","spam","spam","ham","spam","ham","ham",
              "spam","ham","spam","ham","ham","spam","spam","ham","ham","spam"],
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
        "Claim your reward now!"
    ]
}

df = pd.DataFrame(data)

with st.expander("📊 View Dataset"):
    st.dataframe(df)

# ----------------------------
# Preprocessing
# ----------------------------
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label_num"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train Model
model = LogisticRegression()
model.fit(X_vec, y)

# ----------------------------
# Evaluation
# ----------------------------
st.subheader("📈 Model Performance")
y_pred = model.predict(X_vec)
st.text(classification_report(y, y_pred))

# ----------------------------
# USER INPUT
# ----------------------------
st.subheader("✍️ Enter Message to Check")
user_input = st.text_area("Type message here...")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        user_vec = vectorizer.transform([user_input])
        result = model.predict(user_vec)[0]

        if result == 1:
            st.error("🚨 SPAM Message")
        else:
            st.success("✅ NOT SPAM Message")
