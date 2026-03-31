import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("🧠 Neural Network Prediction")

@st.cache_resource
def load_nn_model():
    try:
        return load_model("models/nn_model.keras", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_nn_model()

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

text = st.text_area("📝 Enter news text")

if model is not None and st.button("🔍 Predict NN"):
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