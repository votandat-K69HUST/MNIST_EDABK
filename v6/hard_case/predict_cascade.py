"""
Cascade Inference Pipeline:
1. Predict using main model ensemble (mlp_v4_new_data).
2. For samples with confidence < threshold (default 0.6), override prediction using specialist model ensemble (mlp_hard_case_specialist).
3. Output final submission CSV.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_run", default="model/runs/mlp_v4_new_data", help="Directory for v4_new main models")
    parser.add_argument("--hard_run", default="model/runs/mlp_hard_case_specialist", help="Directory for hard case specialist models")
    parser.add_argument("--test", default="test.csv", help="Path to test.csv")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold below which hard model overrides (default: 0.6)")
    parser.add_argument("--output", default="v6/submission_v6_cascade.csv", help="Output submission CSV path")
    parser.add_argument("--tta", type=int, default=2, help="TTA shifts")
    args = parser.parse_args()

    print(f"=== Running Cascade Pipeline (Threshold = {args.threshold}) ===")
    
    # Load test data
    df_test = pd.read_csv(args.test)
    ids = df_test["id"].values
    X_test = torch.from_numpy(df_test.drop(columns=["id"]).values.reshape(-1, 28, 28).astype(np.float32) / 255.0)

    # 1. Main Model Predictions
    print(f"Loading main models from {args.main_run}...")
    main_models = load_models(args.main_run)
    proba_main = predict_proba(main_models, X_test, tta_shifts=args.tta)
    conf_main, pred_main = proba_main.max(dim=1)
    pred_main = pred_main.numpy() + LABEL_OFFSET
    conf_main = conf_main.numpy()

    # 2. Hard Case Specialist Predictions
    print(f"Loading hard case models from {args.hard_run}...")
    hard_models = load_models(args.hard_run)
    proba_hard = predict_proba(hard_models, X_test, tta_shifts=args.tta)
    conf_hard, pred_hard = proba_hard.max(dim=1)
    pred_hard = pred_hard.numpy() + LABEL_OFFSET
    conf_hard = conf_hard.numpy()

    # 3. Cascade Selection
    low_conf_mask = (conf_main < args.threshold)
    override_indices = np.where(low_conf_mask)[0]

    final_labels = pred_main.copy()
    final_labels[low_conf_mask] = pred_hard[low_conf_mask]

    print(f"\nTotal test samples: {len(ids)}")
    print(f"Samples with main confidence < {args.threshold}: {len(override_indices)} ({len(override_indices)/len(ids)*100:.2f}%)")

    changes_count = 0
    print("\n--- Detailed Override Decisions ---")
    for idx in override_indices:
        sample_id = ids[idx]
        old_l = pred_main[idx]
        new_l = pred_hard[idx]
        c_main = conf_main[idx]
        c_hard = conf_hard[idx]
        status = "CHANGED" if old_l != new_l else "SAME"
        if old_l != new_l:
            changes_count += 1
        print(f"ID {sample_id:3d} | Main: {old_l} (conf: {c_main:.4f}) -> Specialist: {new_l} (conf: {c_hard:.4f}) | [{status}]")

    print(f"\nTotal predictions modified by Specialist Model: {changes_count}/{len(ids)}")

    # Save final submission
    sub = pd.DataFrame({"id": ids, "label": final_labels.astype(int)})
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f"Saved cascade submission to: {args.output}")

if __name__ == "__main__":
    main()
