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
models_hard = load_models("v6/hard_case")

df_test = pd.read_csv("test.csv")
ids = df_test["id"].values
X_test = torch.from_numpy(df_test.drop(columns=["id"]).values.reshape(-1, 28, 28).astype(np.float32) / 255.0)

proba_v4 = predict_proba(models_v4, X_test, tta_shifts=2)
conf_v4, pred_v4 = proba_v4.max(dim=1)
pred_v4 = pred_v4.numpy() + LABEL_OFFSET
conf_v4 = conf_v4.numpy()

proba_hard = predict_proba(models_hard, X_test, tta_shifts=2)
conf_hard, pred_hard = proba_hard.max(dim=1)
pred_hard = pred_hard.numpy() + LABEL_OFFSET
conf_hard = conf_hard.numpy()

# Check overrides at different thresholds: 0.3, 0.4, 0.5, 0.55, 0.6
print("=================================================================")
print("CASCADE PIPELINE OVERRIDE DIAGNOSIS AT DIFFERENT THRESHOLDS")
print("=================================================================")
for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    mask = conf_v4 < th
    overrides = (pred_v4[mask] != pred_hard[mask]).sum()
    print(f"Threshold < {th:.2f}: {mask.sum():2d} candidates -> {overrides:2d} prediction changes")

print("\n--- Detailed Analysis of the 17 Overridden Samples (Threshold 0.6) ---")
override_mask = (conf_v4 < 0.6) & (pred_v4 != pred_hard)
override_idx = np.where(override_mask)[0]

analysis_list = []
for idx in override_idx:
    sid = ids[idx]
    v4_p = pred_v4[idx]
    v4_c = conf_v4[idx]
    hard_p = pred_hard[idx]
    hard_c = conf_hard[idx]
    
    # Top 2 from v4
    v4_probs = proba_v4[idx].numpy()
    top2_v4_idx = np.argsort(v4_probs)[::-1][:2] + LABEL_OFFSET
    top2_v4_p = v4_probs[np.argsort(v4_probs)[::-1][:2]]
    
    analysis_list.append({
        "id": sid,
        "v4_pred": v4_p,
        "v4_conf": np.round(v4_c, 4),
        "hard_pred": hard_p,
        "hard_conf": np.round(hard_c, 4),
        "v4_alt_pred": top2_v4_idx[1],
        "v4_alt_conf": np.round(top2_v4_p[1], 4)
    })

df_analysis = pd.DataFrame(analysis_list)
print(df_analysis.to_string(index=False))

# Also compare with submission_v4_new.csv vs submission_v6_cascade.csv directly
sub_v4 = pd.read_csv("v4/submission_v4_new.csv")
sub_cas = pd.read_csv("v6/submission_v6_cascade.csv")

diff_mask = sub_v4['label'] != sub_cas['label']
print(f"\nTotal differences between submission_v4_new and submission_v6_cascade: {diff_mask.sum()}")
