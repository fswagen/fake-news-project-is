import pandas as pd

df = pd.read_csv("data/processed.csv")

rows_per_file = 400000  # ลดจำนวนแถวให้แต่ละไฟล์ <100 MB

for i, start in enumerate(range(0, len(df), rows_per_file), 1):
    df_chunk = df.iloc[start:start+rows_per_file]
    df_chunk.to_csv(f"data/processed_part{i}.csv", index=False)