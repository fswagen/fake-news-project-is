import streamlit as st
import joblib

st.sidebar.title("🚀 Navigation")
st.sidebar.info("Fake News Detection System")

st.title("📰 Fake News Detection System")

st.set_page_config(page_title="ML Prediction")

st.title("🤖 Machine Learning Prediction")

model = joblib.load("models/ml_model.pkl")

text = st.text_area("📝 Enter news text here")

if st.button("🔍 Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        pred = model.predict([text])
        
        if pred[0] == 1:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")