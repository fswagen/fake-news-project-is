import streamlit as st
import os
import json

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

st.sidebar.title("🚀 Navigation")
st.sidebar.info("Fake News Detection System")

st.title("📰 Fake News Detection System")
st.title("🧪 Test Neural Network")

MODEL_PATH = "models/nn_model.keras"
TOKENIZER_PATH = "models/tokenizer.json"

# =========================
# โหลด MODEL
# =========================
@st.cache_resource
def load_nn_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model file not found")
            return None
        return load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_nn_model()

# =========================
# โหลด TOKENIZER
# =========================
try:
    if not os.path.exists(TOKENIZER_PATH):
        st.error("❌ Tokenizer file not found")
        tokenizer = None
    else:
        with open(TOKENIZER_PATH) as f:
            data = f.read()   # ✅ สำคัญมาก
            tokenizer = tokenizer_from_json(data)
except Exception as e:
    st.error(f"❌ Error loading tokenizer: {e}")
    tokenizer = None

# =========================
# UI
# =========================
text = st.text_area("Enter news text")

if model is not None and tokenizer is not None and st.button("Predict NN"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=100)
        pred = model.predict(pad)

        if pred[0][0] > 0.5:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")