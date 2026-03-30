print("🚀 Start training...")

# =========================
# 1. Import
# =========================
import pandas as pd
import re
import joblib
import pickle
import os
import json

from sklearn.metrics import accuracy_score, confusion_matrix

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# NN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# =========================
# 2. Load Data
# =========================
df = pd.read_csv("data/combined.csv")

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# 3. Clean Text
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['clean'] = df['text'].apply(clean_text)

# =========================
# 4. Split Data
# =========================
X = df['clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. ML Model (Ensemble)
# =========================
model_ml = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', VotingClassifier(estimators=[
        ('nb', MultinomialNB()),
        ('lr', LogisticRegression(max_iter=200)),
        ('rf', RandomForestClassifier())
    ], voting='hard'))
])

print("Training ML model...")
model_ml.fit(X_train, y_train)

# Evaluate ML
y_pred = model_ml.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("ML Accuracy:", acc)

# Save ML model
joblib.dump(model_ml, "models/ml_model.pkl")
print("ML model saved!")

# =========================
# 6. Neural Network (LSTM)
# =========================
print("Preparing data for NN...")

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# =========================
# 7. Build Model
# =========================
model_nn = Sequential([
    Embedding(5000, 64, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training NN model...")
model_nn.fit(X_train_pad, y_train, epochs=3)

# ✅ Evaluate NN (สำคัญมาก)
loss, nn_acc = model_nn.evaluate(X_test_pad, y_test)
print("NN Accuracy:", nn_acc)

# Save NN model
model_nn.save("models/nn_model.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("NN model saved!")

# =========================
# 8. SAVE METRICS (ต้องอยู่ท้ายสุด)
# =========================
with open("results/metrics.json", "w") as f:
    json.dump({
        "ml_accuracy": float(acc),
        "nn_accuracy": float(nn_acc),
        "confusion_matrix": cm.tolist()
    }, f)

print("📊 Metrics saved!")

print("✅ Training completed successfully!")