import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "text_model.joblib")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "text_tokenizer.joblib")

def load_model_and_tokenizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):

        tokenizer = joblib.load(TOKENIZER_PATH)
        model = joblib.load(MODEL_PATH)
 

    return tokenizer, model

def classify_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

def run():
    st.title("Twitter Sentiment Classification")
    st.write("Enter your tweet below:")

    tweet = st.text_area("Tweet:")

    if st.button("Classify"):
        if tweet:
            tokenizer, model = load_model_and_tokenizer()
            sentiment = classify_sentiment(tweet, tokenizer, model)
            sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
            st.write(f"Sentiment: {sentiment_labels[sentiment]}")
        else:
            st.warning("Please enter a tweet for classification.")

if __name__ == "__main__":
    run()
