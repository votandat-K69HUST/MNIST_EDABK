import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import json

sys.path.insert(0, ".")
from model.mlp import build_model, LABEL_OFFSET

def load_models(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "model_seed*.pt")))
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = build_model(ck["config"]["model"], ck["mean"], ck["std"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models

@torch.no_grad()
def predict_proba(models, X, tta_shifts=2, bs=2048):
    shifts = [(0, 0)]
    if tta_shifts > 0:
        s = tta_shifts
        shifts += [(s, 0), (-s, 0), (0, s), (0, -s)]
    out = torch.zeros(len(X), 11)
    for m in models:
        for dy, dx in shifts:
            for i in range(0, len(X), bs):
                xb = X[i:i+bs]
                if dy or dx:
                    xb = torch.roll(xb, shifts=(dy, dx), dims=(1, 2))
                out[i:i+bs] += torch.softmax(m(xb), dim=1)
    return out / (len(models) * len(shifts))

print("Loading models...")
models = load_models("model/runs/mlp_v4_new_data")

# 1. Evaluate on Test Set
print("Analyzing test.csv...")
df_test = pd.read_csv("test.csv")
X_test = df_test.drop(columns=["id"]).values.reshape(-1, 28, 28).astype(np.float32) / 255.0
X_test = torch.from_numpy(X_test)
proba_test = predict_proba(models, X_test, tta_shifts=2)

conf_test, pred_test = proba_test.max(dim=1)
pred_test = pred_test.numpy() + LABEL_OFFSET
conf_test = conf_test.numpy()

# Find top 2 predictions for confusion analysis
top2_prob, top2_idx = torch.topk(proba_test, 2, dim=1)
top2_idx = top2_idx.numpy() + LABEL_OFFSET
top2_prob = top2_prob.numpy()

# Statistics
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
print("\n=== Confidence Distribution on Test Set ===")
for t in thresholds:
    count = (conf_test < t).sum()
    print(f"Confidence < {t}: {count} samples ({(count/len(conf_test))*100:.2f}%)")

print("\n=== Most common top-2 confusions (Confidence < 0.8) ===")
low_conf_mask = conf_test < 0.8
low_conf_idx = np.where(low_conf_mask)[0]

confusions = []
for idx in low_conf_idx:
    c1, c2 = top2_idx[idx][0], top2_idx[idx][1]
    p1, p2 = top2_prob[idx][0], top2_prob[idx][1]
    pair = tuple(sorted((c1, c2)))
    confusions.append(pair)

from collections import Counter
c = Counter(confusions)
for pair, count in c.most_common(10):
    print(f"Pair {pair}: {count} samples")

# Save low confidence samples to review
low_conf_df = df_test.iloc[low_conf_idx].copy()
low_conf_df['pred'] = pred_test[low_conf_idx]
low_conf_df['conf'] = conf_test[low_conf_idx]
low_conf_df['alt_pred'] = top2_idx[low_conf_idx, 1]
low_conf_df['alt_conf'] = top2_prob[low_conf_idx, 1]

low_conf_df.to_csv("v4/low_confidence_test.csv", index=False)
print(f"\nSaved {len(low_conf_df)} low confidence samples to v4/low_confidence_test.csv")

# 2. Evaluate on Val Set to see actual accuracy drops
print("\nAnalyzing v5_new_val.npz...")
val_data = np.load("data/output/v5_new_val.npz")
X_val = torch.from_numpy(val_data['X'].astype(np.float32) / 255.0)
y_val = val_data['y']

proba_val = predict_proba(models, X_val, tta_shifts=2)
conf_val, pred_val = proba_val.max(dim=1)
pred_val = pred_val.numpy() + LABEL_OFFSET
conf_val = conf_val.numpy()

# Accuracy vs Confidence
acc = (pred_val == y_val)
print("\n=== Val Set Accuracy by Confidence Bin ===")
bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
for i in range(len(bins)-1):
    mask = (conf_val >= bins[i]) & (conf_val < bins[i+1])
    if mask.sum() > 0:
        bin_acc = acc[mask].mean() * 100
        print(f"[{bins[i]:.2f} - {bins[i+1]:.2f}): {mask.sum():5d} samples -> Accuracy: {bin_acc:.2f}%")

