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

print("Loading models...")
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

thresholds = [0.55, 0.50, 0.45, 0.40, 0.35, 0.30]

print("\n==========================================================================")
print("GENERATING CASCADE SUBMISSION FILES FOR DECREASING THRESHOLDS")
print("==========================================================================")

summary_list = []

for th in thresholds:
    mask = conf_v4 < th
    final_labels = pred_v4.copy()
    final_labels[mask] = pred_hard[mask]
    
    diff_mask = (pred_v4 != final_labels)
    diff_count = diff_mask.sum()
    
    out_file = f"v6/submission_cascade_th{th:.2f}.csv"
    sub = pd.DataFrame({"id": ids, "label": final_labels.astype(int)})
    sub.to_csv(out_file, index=False)
    
    diff_ids = ids[diff_mask]
    changes_str = ", ".join([f"ID {sid} ({pred_v4[ids==sid][0]}->{final_labels[ids==sid][0]})" for sid in diff_ids])
    
    summary_list.append({
        "Threshold": f"{th:.2f}",
        "Candidates (< Th)": mask.sum(),
        "Modified Count": diff_count,
        "Output File": out_file,
        "Modified Samples": changes_str if changes_str else "None"
    })

df_summary = pd.DataFrame(summary_list)
print(df_summary[["Threshold", "Candidates (< Th)", "Modified Count", "Output File"]].to_string(index=False))

print("\n--- Detailed Changes Per Threshold ---")
for row in summary_list:
    print(f"\n Threshold {row['Threshold']}: {row['Modified Count']} modified sample(s)")
    print(f" File: {row['Output File']}")
    print(f" Samples: {row['Modified Samples']}")

