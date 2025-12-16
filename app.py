import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ---------- Load intents ----------
with open("intents.json", "r") as f:
    intents = json.load(f)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        tags.append(intent["tag"])

# training the model
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(patterns)

    clf = LogisticRegression(
        max_iter=10000,
        class_weight="balanced"
    )
    clf.fit(X, tags)
    return vectorizer, clf

vectorizer, clf = train_model()

def chatbot(user_input):
    clean_input = preprocess_text(user_input)
    X = vectorizer.transform([clean_input])
    tag = clf.predict(X)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

def main():
    st.title("Hospital Chatbot using NLP")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You")

    if user_input:
        response = chatbot(user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))

        with open("chat_log.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([user_input, response,
                             datetime.datetime.now()])

        st.session_state.user_input = ""

    for sender, msg in st.session_state.history:
        if sender == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**ChatBot:** {msg}")
    
    menu = st.sidebar.radio("Menu", ["Chat", "History"])

    if menu == "Chat":
        st.subheader(" ")
    elif menu == "History":
        st.subheader("Conversation History")

        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    st.write(f"User: {row[0]}")
                    st.write(f"Bot: {row[1]}")
                    st.write(f"Time: {row[2]}")
                    st.markdown("---")
        else:
            st.info("No history found.")


if __name__ == "__main__":
    main()
