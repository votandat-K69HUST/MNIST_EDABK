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

models_v4 = load_models("model/runs/mlp_v4_new_data")
models_heavy = load_models("v6/hard_case")
models_light = load_models("model/runs/mlp_hard_case_lightweight")

# 1. Evaluate on 31 Hand Drawn Hard Cases
df_hard_raw = pd.read_csv("v6/ve_tay_hard_cases.csv")
y_hard_raw = df_hard_raw['label'].values
X_hard_raw = torch.from_numpy(df_hard_raw.drop(columns=['label', 'writer', 'time', 'strokes'], errors='ignore').values.reshape(-1, 28, 28).astype(np.float32) / 255.0)

proba_v4 = predict_proba(models_v4, X_hard_raw)
proba_heavy = predict_proba(models_heavy, X_hard_raw)
proba_light = predict_proba(models_light, X_hard_raw)

pred_v4 = proba_v4.max(1)[1].numpy() + LABEL_OFFSET
pred_heavy = proba_heavy.max(1)[1].numpy() + LABEL_OFFSET
pred_light = proba_light.max(1)[1].numpy() + LABEL_OFFSET

acc_v4 = (pred_v4 == y_hard_raw).mean() * 100
acc_heavy = (pred_heavy == y_hard_raw).mean() * 100
acc_light = (pred_light == y_hard_raw).mean() * 100

print("==========================================================================")
print("COMPARING HEAVY VS LIGHTWEIGHT HARD CASE MODELS ON 31 HAND-DRAWN HARD CASES")
print("==========================================================================")
print(f"v4_new Baseline (801k params):                        {acc_v4:.2f}% ({(pred_v4 == y_hard_raw).sum()}/31)")
print(f"Heavy Specialist (512-512-256, 801k params):          {acc_heavy:.2f}% ({(pred_heavy == y_hard_raw).sum()}/31)")
print(f"Lightweight Specialist (256-128, 236k params):         {acc_light:.2f}% ({(pred_light == y_hard_raw).sum()}/31)")

# 2. Evaluate on 236 Mined Real Hard Samples
train_data = np.load("data/output/v5_new_train.npz")
X_tr = torch.from_numpy(train_data['X'].astype(np.float32) / 255.0)
y_tr = train_data['y']

p_v4_tr = predict_proba(models_v4, X_tr)
c_v4_tr, pr_v4_tr = p_v4_tr.max(1)
pr_v4_tr = pr_v4_tr.numpy() + LABEL_OFFSET

hard_mask = (pr_v4_tr != y_tr) | (c_v4_tr.numpy() < 0.8)
X_mined = X_tr[hard_mask]
y_mined = y_tr[hard_mask]

pr_v4_m = pr_v4_tr[hard_mask]
pr_heavy_m = predict_proba(models_heavy, X_mined).max(1)[1].numpy() + LABEL_OFFSET
pr_light_m = predict_proba(models_light, X_mined).max(1)[1].numpy() + LABEL_OFFSET

print("\n==========================================================================")
print("COMPARING HEAVY VS LIGHTWEIGHT HARD CASE MODELS ON 236 MINED REAL HARD SAMPLES")
print("==========================================================================")
print(f"v4_new Baseline:                                      {(pr_v4_m == y_mined).mean()*100:.2f}% ({(pr_v4_m == y_mined).sum()}/236)")
print(f"Heavy Specialist (512-512-256, 801k params):          {(pr_heavy_m == y_mined).mean()*100:.2f}% ({(pr_heavy_m == y_mined).sum()}/236)")
print(f"Lightweight Specialist (256-128, 236k params):         {(pr_light_m == y_mined).mean()*100:.2f}% ({(pr_light_m == y_mined).sum()}/236)")

