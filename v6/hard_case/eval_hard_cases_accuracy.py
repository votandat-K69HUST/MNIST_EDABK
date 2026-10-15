import os
import sys
import glob
import torch
import numpy as np
import pandas as pd

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

print("Loading v4_new models...")
models_v4 = load_models("model/runs/mlp_v4_new_data")

print("Loading model_hard_case specialist models...")
models_hard = load_models("v6/hard_case")

# 1. EVALUATION ON THE 31 ORIGINAL HAND-DRAWN HARD CASES
print("\n=======================================================")
print("1. EVALUATING ON THE 31 ORIGINAL HAND-DRAWN HARD CASES")
print("=======================================================")
df_hard_raw = pd.read_csv("v6/ve_tay_hard_cases.csv")
y_hard_raw = df_hard_raw['label'].values
X_hard_raw = df_hard_raw.drop(columns=['label', 'writer', 'time', 'strokes'], errors='ignore').values.reshape(-1, 28, 28).astype(np.float32) / 255.0
X_hard_raw_tensor = torch.from_numpy(X_hard_raw)

proba_v4_raw = predict_proba(models_v4, X_hard_raw_tensor)
conf_v4_raw, pred_v4_raw = proba_v4_raw.max(dim=1)
pred_v4_raw = pred_v4_raw.numpy() + LABEL_OFFSET
conf_v4_raw = conf_v4_raw.numpy()

proba_hard_raw = predict_proba(models_hard, X_hard_raw_tensor)
conf_hard_raw, pred_hard_raw = proba_hard_raw.max(dim=1)
pred_hard_raw = pred_hard_raw.numpy() + LABEL_OFFSET
conf_hard_raw = conf_hard_raw.numpy()

# Cascade pipeline evaluation (threshold = 0.6)
pred_cascade_raw = pred_v4_raw.copy()
low_conf_mask_raw = conf_v4_raw < 0.6
pred_cascade_raw[low_conf_mask_raw] = pred_hard_raw[low_conf_mask_raw]

acc_v4_raw = (pred_v4_raw == y_hard_raw).mean() * 100
acc_hard_raw = (pred_hard_raw == y_hard_raw).mean() * 100
acc_cascade_raw = (pred_cascade_raw == y_hard_raw).mean() * 100

print(f"v4_new Accuracy on 31 Hard Drawings:       {acc_v4_raw:.2f}% ({(pred_v4_raw == y_hard_raw).sum()}/31)")
print(f"Hard Specialist Accuracy on 31 Drawings:   {acc_hard_raw:.2f}% ({(pred_hard_raw == y_hard_raw).sum()}/31)")
print(f"CASCADE Pipeline Accuracy on 31 Drawings:  {acc_cascade_raw:.2f}% ({(pred_cascade_raw == y_hard_raw).sum()}/31)")

print("\n--- Detailed Case-by-Case Breakdown (31 Drawings) ---")
for i in range(len(y_hard_raw)):
    gt = y_hard_raw[i]
    p_v4 = pred_v4_raw[i]
    c_v4 = conf_v4_raw[i]
    p_hard = pred_hard_raw[i]
    c_hard = conf_hard_raw[i]
    p_cas = pred_cascade_raw[i]
    status_v4 = "CORRECT" if p_v4 == gt else "WRONG"
    status_cas = "CORRECT" if p_cas == gt else "WRONG"
    print(f"Sample {i+1:2d} | True: {gt:2d} | v4_new: {p_v4:2d} (conf {c_v4:.3f}) [{status_v4}] -> Cascade: {p_cas:2d} [{status_cas}]")

# 2. EVALUATION ON MINED REAL HARD CASES FROM TRAIN SET
print("\n=======================================================")
print("2. EVALUATING ON MINED REAL HARD CASES FROM REAL TRAIN SET (236 samples)")
print("=======================================================")
train_data = np.load("data/output/v5_new_train.npz")
X_train_raw = train_data['X']
y_train_raw = train_data['y']
X_train_tensor = torch.from_numpy(X_train_raw.astype(np.float32) / 255.0)

proba_v4_tr = predict_proba(models_v4, X_train_tensor)
conf_v4_tr, pred_v4_tr = proba_v4_tr.max(dim=1)
pred_v4_tr = pred_v4_tr.numpy() + LABEL_OFFSET
conf_v4_tr = conf_v4_tr.numpy()

# Mask for mined hard samples
hard_mask_tr = (pred_v4_tr != y_train_raw) | (conf_v4_tr < 0.8)
X_hard_mined = X_train_tensor[hard_mask_tr]
y_hard_mined = y_train_raw[hard_mask_tr]

proba_hard_mined = predict_proba(models_hard, X_hard_mined)
conf_hard_mined, pred_hard_mined = proba_hard_mined.max(dim=1)
pred_hard_mined = pred_hard_mined.numpy() + LABEL_OFFSET
conf_hard_mined = conf_hard_mined.numpy()

pred_v4_mined = pred_v4_tr[hard_mask_tr]
conf_v4_mined = conf_v4_tr[hard_mask_tr]

pred_cascade_mined = pred_v4_mined.copy()
low_conf_mined = conf_v4_mined < 0.6
pred_cascade_mined[low_conf_mined] = pred_hard_mined[low_conf_mined]

acc_v4_mined = (pred_v4_mined == y_hard_mined).mean() * 100
acc_hard_mined = (pred_hard_mined == y_hard_mined).mean() * 100
acc_cascade_mined = (pred_cascade_mined == y_hard_mined).mean() * 100

print(f"v4_new Accuracy on 236 Mined Real Hard Samples:      {acc_v4_mined:.2f}% ({(pred_v4_mined == y_hard_mined).sum()}/236)")
print(f"Hard Specialist Accuracy on 236 Mined Hard Samples:  {acc_hard_mined:.2f}% ({(pred_hard_mined == y_hard_mined).sum()}/236)")
print(f"CASCADE Pipeline Accuracy on 236 Mined Hard Samples: {acc_cascade_mined:.2f}% ({(pred_cascade_mined == y_hard_mined).sum()}/236)")

