import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Hospital Chatbot Using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input in the context of a hospital setting. The chatbot uses Natural Language Processing (NLP) techniques, particularly Logistic Regression, to classify intents and extract entities from user input. The chatbot is built using **Streamlit**, a Python library for building interactive web applications.")
        st.subheader("Project Overview:")
        
        st.write(""" 
        The project is divided into two main parts:
        1. **NLP Techniques and Logistic Regression**: 
        The chatbot is trained on a labeled dataset consisting of various intents (such as greeting, appointment inquiries, etc.) and entities (such as patient names, appointment dates).
        2. **Streamlit Chatbot Interface**:
        Streamlit is used to build the web-based interface that allows users (patients and hospital staff) to interact with the chatbot. The interface accepts text input from the user and displays the chatbot's responses.
        """)
        
        st.subheader("Dataset:")
        
        st.write(""" The dataset used in this project is a collection of labeled intents and entities, which is stored in a list format. Each entry in the dataset includes:
        - **Intents**: The user’s intent (e.g., "greeting", "appointment", "query doctor availability").
        - **Entities**: Information extracted from the input, such as names, dates, and medical-related queries.
        - **Text**: Example user inputs corresponding to different intents and entities.""")
        
        st.subheader("Streamlit Chatbot Interface:")
        st.write("""
        The chatbot interface is designed using Streamlit. It includes:
        - **Text Input Box**: Users can type their questions or queries.
        - **Chat Window**: Displays the chatbot’s responses based on user input.
        - The chatbot uses the trained NLP model to generate appropriate responses in real-time.
        """)
        
        st.subheader("Conclusion:")
        
        st.write("""
        In this project, a chatbot was developed to assist patients and staff in hospitals by understanding and responding to medical-related queries. The chatbot was trained using NLP and Logistic Regression, with the interface built using Streamlit. This system could be extended by adding more intents, incorporating more sophisticated NLP models, and integrating it with hospital management systems for enhanced functionality.
        """)

if __name__ == '__main__':
    main()
