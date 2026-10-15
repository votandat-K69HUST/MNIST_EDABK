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
    if not paths:
        raise FileNotFoundError(f"No model_seed*.pt files found in {run_dir}")
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
    print("=======================================================")
    print("    EVALUATING MODEL V7 ON HANDWRITTEN HARD CASES    ")
    print("=======================================================\n")
    
    print("Loading v4_new models...")
    models_v4 = load_models("model/runs/mlp_v4_new_data")
    print(f"Loaded {len(models_v4)} v4_new seed models.")
    
    print("Loading v7_best_tuned models...")
    models_v7 = load_models("model/runs/mlp_v7_best_tuned")
    print(f"Loaded {len(models_v7)} v7 seed models.")
    
    # ---------------------------------------------------------
    # 1. EVALUATION ON THE 31 ORIGINAL HAND-DRAWN HARD CASES
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------")
    print("1. EVALUATION ON 31 HAND-DRAWN HARD CASES (v6/ve_tay_hard_cases.csv)")
    print("-------------------------------------------------------")
    
    df_hard_raw = pd.read_csv("v6/ve_tay_hard_cases.csv")
    y_hard_raw = df_hard_raw['label'].values
    X_hard_raw = df_hard_raw.drop(columns=['label', 'writer', 'time', 'strokes'], errors='ignore').values.reshape(-1, 28, 28).astype(np.float32) / 255.0
    X_hard_tensor = torch.from_numpy(X_hard_raw)
    
    # v4_new predictions
    proba_v4 = predict_proba(models_v4, X_hard_tensor)
    conf_v4, pred_v4 = proba_v4.max(dim=1)
    pred_v4 = pred_v4.numpy() + LABEL_OFFSET
    conf_v4 = conf_v4.numpy()
    
    # v7 predictions
    proba_v7 = predict_proba(models_v7, X_hard_tensor)
    conf_v7, pred_v7 = proba_v7.max(dim=1)
    pred_v7 = pred_v7.numpy() + LABEL_OFFSET
    conf_v7 = conf_v7.numpy()
    
    acc_v4 = (pred_v4 == y_hard_raw).mean() * 100
    acc_v7 = (pred_v7 == y_hard_raw).mean() * 100
    
    correct_v4 = (pred_v4 == y_hard_raw).sum()
    correct_v7 = (pred_v7 == y_hard_raw).sum()
    
    print(f"v4_new Accuracy on 31 Hand-drawn Hard Cases: {acc_v4:.2f}% ({correct_v4}/31)")
    print(f"v7     Accuracy on 31 Hand-drawn Hard Cases: {acc_v7:.2f}% ({correct_v7}/31)")
    print(f"Chênh lệch: +{correct_v7 - correct_v4} mẫu đúng (Accuracy {(acc_v7 - acc_v4):+.2f}%)")
    
    print("\n--- Chi tiết từng mẫu vẽ tay (31 Samples Breakdown) ---")
    print(f"{'No':<3} | {'True':<4} | {'v4_new Pred':<11} | {'v4 Conf':<7} | {'v7 Pred':<11} | {'v7 Conf':<7} | {'Status v7':<10}")
    print("-" * 75)
    
    for i in range(len(y_hard_raw)):
        gt = y_hard_raw[i]
        p4 = pred_v4[i]
        c4 = conf_v4[i]
        p7 = pred_v7[i]
        c7 = conf_v7[i]
        
        status_v7 = "CORRECT" if p7 == gt else "WRONG"
        print(f"{i+1:2d}  | {gt:<4} | {p4:<11} | {c4:<7.3f} | {p7:<11} | {c7:<7.3f} | {status_v7:<10}")

    # ---------------------------------------------------------
    # 2. EVALUATION ON REAL MINED HARD CASES IN REAL TRAIN & VAL
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------")
    print("2. EVALUATION ON MINED HARD CASES FROM REAL DATASET (v5_new_train.npz & v5_new_val.npz)")
    print("-------------------------------------------------------")
    
    val_data = np.load("data/output/v5_new_val.npz")
    X_val = torch.from_numpy(val_data['X'].astype(np.float32) / 255.0)
    y_val = val_data['y']
    
    proba_v4_val = predict_proba(models_v4, X_val)
    conf_v4_val, pred_v4_val = proba_v4_val.max(dim=1)
    pred_v4_val = pred_v4_val.numpy() + LABEL_OFFSET
    
    proba_v7_val = predict_proba(models_v7, X_val)
    conf_v7_val, pred_v7_val = proba_v7_val.max(dim=1)
    pred_v7_val = pred_v7_val.numpy() + LABEL_OFFSET
    
    # Mined hard cases in val set (where v4_new was wrong or low confidence < 0.8)
    hard_mask_val = (pred_v4_val != y_val) | (conf_v4_val.numpy() < 0.8)
    
    acc_v4_val_all = (pred_v4_val == y_val).mean() * 100
    acc_v7_val_all = (pred_v7_val == y_val).mean() * 100
    
    acc_v4_val_hard = (pred_v4_val[hard_mask_val] == y_val[hard_mask_val]).mean() * 100
    acc_v7_val_hard = (pred_v7_val[hard_mask_val] == y_val[hard_mask_val]).mean() * 100
    
    print(f"Tập Validation Thực Tế (868 mẫu):")
    print(f"  - v4_new Total Accuracy: {acc_v4_val_all:.2f}% ({(pred_v4_val == y_val).sum()}/868)")
    print(f"  - v7     Total Accuracy: {acc_v7_val_all:.2f}% ({(pred_v7_val == y_val).sum()}/868)")
    print(f"  - v4_new Hard Cases Accuracy ({hard_mask_val.sum()} mẫu): {acc_v4_val_hard:.2f}%")
    print(f"  - v7     Hard Cases Accuracy ({hard_mask_val.sum()} mẫu): {acc_v7_val_hard:.2f}%")

if __name__ == "__main__":
    main()
