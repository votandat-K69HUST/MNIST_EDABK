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
models = load_models("model/runs/mlp_v4_new_data")

print("Loading v5_new_train.npz...")
train_data = np.load("data/output/v5_new_train.npz")
X_train_raw = train_data['X']
y_train_raw = train_data['y']

X_train_tensor = torch.from_numpy(X_train_raw.astype(np.float32) / 255.0)

print("Running v4_new on v5_new_train...")
proba = predict_proba(models, X_train_tensor, tta_shifts=2)
conf, pred = proba.max(dim=1)
pred = pred.numpy() + LABEL_OFFSET
conf = conf.numpy()

# Hard condition: wrong prediction OR confidence < 0.8
wrong_mask = (pred != y_train_raw)
low_conf_mask = (conf < 0.8)
hard_mask = wrong_mask | low_conf_mask

X_hard_mined = X_train_raw[hard_mask]
y_hard_mined = y_train_raw[hard_mask]

print(f"Mined {len(X_hard_mined)} hard samples from v5_new_train (Wrong: {wrong_mask.sum()}, Low Conf < 0.8: {low_conf_mask.sum()})")

# Load user drawn hard cases
user_hard = np.load("data/output/hard_cases_augmented.npz")
X_user_hard = user_hard['X']
y_user_hard = user_hard['y']
print(f"Loaded {len(X_user_hard)} user drawn hard cases.")

# Combine mined + user drawn (repeat mined 3x to give them good weight)
X_hard_combined = np.concatenate([X_user_hard, np.tile(X_hard_mined, (3, 1, 1))], axis=0)
y_hard_combined = np.concatenate([y_user_hard, np.tile(y_hard_mined, 3)], axis=0)

print(f"Total hard training dataset size: {len(X_hard_combined)}")

os.makedirs("data/output", exist_ok=True)
np.savez_compressed("data/output/hard_cases_dataset.npz", X=X_hard_combined, y=y_hard_combined)
print("Saved data/output/hard_cases_dataset.npz successfully!")
