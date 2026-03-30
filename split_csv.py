import pandas as pd
import os

# อ่านไฟล์ CSV ขนาดใหญ่
df = pd.read_csv("data/processed.csv")

# ประมาณจำนวนแถวต่อไฟล์ให้แต่ละไฟล์ <100 MB
rows_per_file = 300000  # ลดลงเพื่อให้แต่ละไฟล์เล็กกว่า 100 MB

# สร้างไฟล์ย่อย
for i, start in enumerate(range(0, len(df), rows_per_file), 1):
    df_chunk = df.iloc[start:start+rows_per_file]
    out_file = f"data/processed_part{i}.csv"
    df_chunk.to_csv(out_file, index=False)
    print(f"สร้างไฟล์ {out_file} ขนาด {os.path.getsize(out_file)/1024/1024:.2f} MB")