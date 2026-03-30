import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns


def load_css():
    with open("app/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# Sidebar (UI โปร)
# =========================
st.sidebar.title("🚀 Navigation")
st.sidebar.info("Fake News Detection System")

# =========================
# Title
# =========================
st.title("📊 Model Performance Dashboard")

# =========================
# Load Metrics
# =========================
with open("results/metrics.json") as f:
    data = json.load(f)

ml_acc = data["ml_accuracy"]
nn_acc = data["nn_accuracy"]
cm = data["confusion_matrix"]

# =========================
# Show Metrics
# =========================
col1, col2 = st.columns(2)

with col1:
    st.metric("ML Accuracy", f"{ml_acc*100:.2f}%")

with col2:
    st.metric("NN Accuracy", f"{nn_acc*100:.2f}%")

# =========================
# Accuracy Comparison Chart
# =========================
st.subheader("📈 Accuracy Comparison")

fig, ax = plt.subplots()
models = ["ML", "NN"]
values = [ml_acc, nn_acc]

ax.bar(models, values)
ax.set_ylabel("Accuracy")

st.pyplot(fig)

# =========================
# Confusion Matrix
# =========================
st.subheader("🔥 Confusion Matrix")

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

st.subheader("📊 Raw Metrics")
st.json(data)