import torch
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from train import build_model
from data_utils import to_tensor

def main():
    df = pd.read_csv('ve_tay_hard_cases.csv')
    y = df['label'].values
    X_raw = df.drop(columns=['label', 'writer', 'time', 'strokes'], errors='ignore').values
    X = X_raw.reshape(-1, 28, 28)
    X_t = to_tensor(X)
    
    for seed in [0, 1, 2]:
        path = f'model/runs/mlp_v6_hard_cases/model_seed{seed}.pt'
        if not os.path.exists(path): continue
        ckpt = torch.load(path, map_location='cpu')
        model = build_model(ckpt['config'])
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        with torch.no_grad():
            logits = model(X_t)
            preds = logits.argmax(dim=1).numpy()
        
        acc = (preds == y).mean()
        print(f"Seed {seed} accuracy on 31 hard cases: {acc*100:.2f}%")

if __name__ == '__main__':
    main()
