import streamlit as st

def load_css():
    with open("app/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
st.title("📰 Fake News Detection System")


st.write("""
### 🎯 Objective
Detect fake news using Machine Learning and Neural Network

### ⚙️ Features
- ML Model (Ensemble)
- Neural Network
- Real-time Prediction

### 👨‍💻 Developed by
Your Name
""")

st.sidebar.title("🚀 Navigation")
st.sidebar.info("Fake News Detection System")


st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")

st.markdown("---")

st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=150)

st.markdown("""
## 🔍 About this project

This system detects **Fake News** using:

- 🤖 Machine Learning (Ensemble)
- 🧠 Neural Network (LSTM)

---

### 👈 Select a page from the sidebar to begin
""")