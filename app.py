import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# These need to be defined or loaded within the Streamlit app context
# For simplicity, we're re-initializing them here. In a real-world app,
# you might load pre-trained models or pass them more elegantly.

model_name = "j-hartmann/emotion-english-distilroberta-base"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

labels = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]

sentiment_map = {
    "joy": "Positive",
    "surprise": "Positive",
    "neutral": "Neutral",
    "anger": "Negative",
    "fear": "Negative",
    "sadness": "Negative",
    "disgust": "Negative"
}

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    emotion = labels[prediction]
    sentiment = sentiment_map[emotion]

    return emotion, sentiment

st.title("Emotion and Sentiment Analysis App")
st.write("Enter a text review below to get its predicted emotion and sentiment.")

user_input = st.text_area("Enter your Bahubali Movie review here:", "I love this movie, it was fantastic!")

if st.button("Analyze Sentiment"):
    if user_input:
        emotion, sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Emotion: {emotion.capitalize()}")
        st.info(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")
