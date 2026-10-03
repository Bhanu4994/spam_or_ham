import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack, csr_matrix

MODELS_DIR = Path(__file__).parent / "models"

# Must match the notebook exactly (same order as num_cols)
NUM_COLS = ["char_count", "word_count", "digit_count", "upper_count", "special_count"]

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    tfidf = joblib.load(MODELS_DIR / "tfidf.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, tfidf, scaler


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [STEMMER.stem(w) for w in words if w not in STOP_WORDS]
    return " ".join(words)


def numeric_features(text):
    return [[
        len(text),
        len(text.split()),
        sum(c.isdigit() for c in text),
        sum(c.isupper() for c in text),
        sum(not c.isalnum() and not c.isspace() for c in text),
    ]]


def predict(message, model, tfidf, scaler):
    text_vec = tfidf.transform([clean_text(message)])
    num_vec = scaler.transform(numeric_features(message))
    features = hstack([text_vec, csr_matrix(num_vec)])

    label = int(model.predict(features)[0])

    # LinearSVC has no predict_proba; decision_function gives a distance from the boundary
    score = float(model.decision_function(features)[0]) if hasattr(model, "decision_function") else None
    return label, score


st.set_page_config(page_title="Spam or Ham", page_icon="📩")
st.title("📩 Spam or Ham Detector")
st.write("Paste a text message and the model will classify it.")

try:
    model, tfidf, scaler = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model files not found in {MODELS_DIR}. Run the notebook first to save them.\n\n{e}")
    st.stop()

message = st.text_area("Message", height=120, placeholder="e.g. Congratulations! You have won a free prize. Call now to claim.")

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        label, score = predict(message, model, tfidf, scaler)

        if label == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM (not spam)")

        if score is not None:
            st.caption(f"Model score: {score:.2f}  (above 0 leans spam, below 0 leans ham; further from 0 means more confident)")

with st.expander("Try these examples"):
    st.write("**Spam-like:** WINNER!! You have been selected for a £900 prize. Call 09061701461 now to claim.")
    st.write("**Ham-like:** Hey, are we still meeting for lunch tomorrow?")

