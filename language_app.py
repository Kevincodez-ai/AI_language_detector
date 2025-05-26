import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("🌍 Language Detection App")
st.write("Enter any sentence and the model will predict the language.")

# Input from user
user_input = st.text_area("Enter your sentence here:")

# Predict button
if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        # Preprocess and predict
        user_input = user_input.lower()
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]

        # Display result
        st.success(f"🗣️ Detected Language: **{prediction}**")
#CREATED BY KEVIN JAMES.
