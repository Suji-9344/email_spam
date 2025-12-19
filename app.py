import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam + Threat Detector")
st.title("📩 SMS Spam / Threat Message Classifier")

# --------------------- DATASET ---------------------
st.subheader("📊 Training Dataset Used")

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
        "free recharge offer",
        "lottery ticket winner",
        "urgent call now",
        "exclusive reward waiting",
        "click here to win",
        "Meet me in class",
        "See you tomorrow",
        "Call me later",
        "Final prize claim today",
        "Hurry offer ends soon"
    ]
}

df = pd.DataFrame(data)
st.dataframe(df)

# --------------------- MODEL ---------------------
df["label_num"] = df["label"].map({"ham":0, "spam":1})

X = df["message"]
y = df["label_num"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# --------------------- THREAT RULE ---------------------
danger_words = [
    "kill","dead","murder","attack","bomb",
    "terror","die","destroy","threat","warning"
]

def is_dangerous(text):
    text = text.lower()
    return any(word in text for word in danger_words)

# --------------------- USER INPUT ---------------------
st.subheader("✍️ Enter Message")
msg = st.text_area("Type message here...")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Enter a message")
    else:
        if is_dangerous(msg):
            st.error("🚨 DANGEROUS / THREAT MESSAGE (SPAM)")
        else:
            v = vectorizer.transform([msg])
            result = model.predict(v)[0]

            if result == 1:
                st.error("🚨 SPAM MESSAGE")
            else:
                st.success("✅ NOT SPAM MESSAGE")
