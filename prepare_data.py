import pandas as pd

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true])

# เลือกเฉพาะคอลัมน์ที่ใช้
df = df[["text", "label"]]

df.to_csv("data/processed.csv", index=False)

print("✅ processed.csv created")