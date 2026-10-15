import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Loading new data...")
df = pd.read_csv('ve_tay_10_20 - data (1).csv')
print(f"Initial shape: {df.shape}")

X_raw = df.drop(columns=['label', 'writer', 'time', 'strokes'], errors='ignore').values.reshape(-1, 28, 28)
y_raw = df['label'].values

keep = []
for i in range(len(X_raw)):
    img = X_raw[i] > 30
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    if not np.any(rows):
        keep.append(True)
        continue
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    ar = w / float(h) if h > 0 else 0
    if 0.3 <= ar <= 2.0:
        keep.append(True)
    else:
        keep.append(False)

df_clean = df[keep].copy()
print(f"Clean shape: {df_clean.shape} (Removed {df.shape[0] - df_clean.shape[0]} outliers)")

X = df_clean.drop(columns=['label', 'writer', 'time', 'strokes'], errors='ignore').values.reshape(-1, 28, 28).astype(np.uint8)
y = df_clean['label'].values.astype(np.int64)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs('data/output', exist_ok=True)
np.savez_compressed('data/output/v5_new_train.npz', X=X_train, y=y_train)
np.savez_compressed('data/output/v5_new_val.npz', X=X_val, y=y_val)
print(f"Saved: Train {len(X_train)} samples, Val {len(X_val)} samples.")
