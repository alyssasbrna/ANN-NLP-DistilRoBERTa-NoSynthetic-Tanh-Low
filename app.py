import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import pickle

# Load embedding model and ANN classifier
embedder = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
ann_model = load_model("ANN-DistilRoBERTa-model.keras")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🕵️ SCAM DETECTION")
text_input = st.text_area("Enter a text:")

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # Generate embedding
        embedding = embedder.encode([text_input], convert_to_numpy=True)
        scaled = scaler.transform(embedding)

        # Predict
        prediction = ann_model.predict(scaled)[0][0]
        label = "🔴 Scam" if prediction >= 0.5 else "🟢 Not Scam"
        st.success(f"{label} (Confidence: {prediction:.2f})")
