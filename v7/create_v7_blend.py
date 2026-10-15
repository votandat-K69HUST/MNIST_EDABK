import torch
import sys
import os
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from model.mlp import build_model, LABEL_OFFSET

def get_all_probas(run_dir, test_csv_path='test.csv'):
    df_test = pd.read_csv(test_csv_path)
    X_raw = df_test.drop(columns=['id', 'label'], errors='ignore').values.reshape(-1, 28, 28).astype(np.float32) / 255.0
    X_tensor = torch.from_numpy(X_raw)
    
    paths = sorted(glob.glob(os.path.join(run_dir, 'model_seed*.pt')))
    out = torch.zeros(len(X_raw), 11)
    shifts = [(0,0), (2,0), (-2,0), (0,2), (0,-2)]
    for p in paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        m = build_model(ck['config']['model'], ck['mean'], ck['std'])
        m.load_state_dict(ck['state_dict'])
        m.eval()
        with torch.no_grad():
            for dy, dx in shifts:
                for i in range(0, len(X_raw), 2048):
                    xb = X_tensor[i:i+2048]
                    if dy or dx:
                        xb = torch.roll(xb, shifts=(dy, dx), dims=(1, 2))
                    out[i:i+2048] += torch.softmax(m(xb), dim=1)
    return (out / (len(paths) * len(shifts))).numpy()

def main():
    print("Generating blended ensemble submission (0.6 v4 + 0.4 v7)...")
    p_v4 = get_all_probas('model/runs/mlp_v4_ensemble')
    p_v7 = get_all_probas('model/runs/mlp_v7_best_tuned')
    
    p_blend = 0.6 * p_v4 + 0.4 * p_v7
    pred_blend = p_blend.argmax(axis=1) + LABEL_OFFSET
    
    df_test = pd.read_csv('test.csv')
    df_sub = pd.DataFrame({'id': df_test['id'], 'label': pred_blend})
    
    out_path = 'v7/submission_v7_blend.csv'
    df_sub.to_csv(out_path, index=False)
    print(f"Saved blended submission to {out_path}")
    
    # Verification
    v4_sub = pd.read_csv('v4/submission_v4.csv')
    diff_pub = (df_sub['label'][:220] != v4_sub['label'][:220]).sum()
    print(f"Diffs vs Public 0.96818 baseline (id < 220): {diff_pub} (ID 61 label: {df_sub['label'][61]})")

if __name__ == "__main__":
    main()
