import streamlit as st
import pandas as pd

st.title("📂 Dataset")

df = pd.read_csv("data/processed.csv")

st.write("### 🔍 Sample Data")
st.dataframe(df.head())

st.write("### 📊 Statistics")
st.write(df.describe())

st.write("""
### 🧠 Features
- text: news content
- label: 0 = Real, 1 = Fake
""")