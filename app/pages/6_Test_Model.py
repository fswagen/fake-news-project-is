import streamlit as st
import joblib
import pickle


def load_css():
    with open("app/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.sidebar.title("🚀 Navigation")
st.sidebar.info("Fake News Detection System")

st.title("🧪 Test Fake News Detection")

# Load model
model = joblib.load("models/ml_model.pkl")

# Input
text = st.text_area("Enter News Text")

if st.button("Predict"):
    prediction = model.predict([text])[0]

    if prediction == 1:
        st.error("🚨 FAKE NEWS")
    else:
        st.success("✅ REAL NEWS")

st.subheader("📂 Upload CSV")

uploaded_file = st.file_uploader("Upload file", type=["csv"])

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)

    st.write(df.head())

    if "text" in df.columns:
        preds = model.predict(df["text"])
        df["prediction"] = preds

        st.write(df)