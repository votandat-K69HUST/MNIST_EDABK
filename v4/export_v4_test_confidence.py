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
models = load_models("model/runs/mlp_v4_new_data")

print("Reading test.csv...")
df_test = pd.read_csv("test.csv")
ids = df_test["id"].values
X_test = df_test.drop(columns=["id"]).values.reshape(-1, 28, 28).astype(np.float32) / 255.0
X_test = torch.from_numpy(X_test)

print("Running inference...")
proba = predict_proba(models, X_test, tta_shifts=2).numpy()

# Top 2 predictions
top2_idx = np.argsort(proba, axis=1)[:, ::-1][:, :2]
top2_prob = np.take_along_axis(proba, top2_idx, axis=1)

pred_label = top2_idx[:, 0] + LABEL_OFFSET
conf1 = top2_prob[:, 0]

second_label = top2_idx[:, 1] + LABEL_OFFSET
conf2 = top2_prob[:, 1]

margin = conf1 - conf2

res_df = pd.DataFrame({
    "id": ids,
    "pred_label": pred_label,
    "confidence": np.round(conf1, 4),
    "second_label": second_label,
    "second_confidence": np.round(conf2, 4),
    "margin": np.round(margin, 4)
})

# Add columns for all class probabilities
for idx in range(11):
    label_num = idx + LABEL_OFFSET
    res_df[f"p_{label_num}"] = np.round(proba[:, idx], 4)

# Sort by confidence ascending so low confidence samples are at top
res_df_sorted = res_df.sort_values(by="confidence", ascending=True)

out_path = "v4/v4_new_test_confidence.csv"
res_df.to_csv(out_path, index=False)
print(f"Exported all 550 samples to {out_path}")

out_sorted_path = "v4/v4_new_test_confidence_sorted.csv"
res_df_sorted.to_csv(out_sorted_path, index=False)
print(f"Exported sorted samples (low conf first) to {out_sorted_path}")

print("\nFirst 10 lowest confidence samples:")
print(res_df_sorted[["id", "pred_label", "confidence", "second_label", "second_confidence", "margin"]].head(10).to_string(index=False))
