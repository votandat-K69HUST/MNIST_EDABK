import os
import sys
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from augment_torch import augment

def main():
    print("Doc file hard cases...")
    df = pd.read_csv('ve_tay_hard_cases.csv')
    y_raw = df['label'].values
    X_raw = df.drop(columns=['label', 'writer', 'time', 'strokes'], errors='ignore').values
    
    # Reshape va scale ve 0..1 cho tensor
    X_raw = X_raw.reshape(-1, 28, 28)
    X_t = torch.from_numpy(X_raw).float() / 255.0
    
    cfg = {
        "enabled": True,
        "rotate": 12,
        "shear": 15,
        "shift": 3,
        "noise_std": 0.05,
        "thickness_p": 0.5
    }
    
    gen = torch.Generator().manual_seed(42)
    
    N_COPIES = 100
    all_X = [X_raw]  # Giu lai nguyen ban
    all_y = [y_raw]
    
    print(f"Bat dau tao augmentation ({N_COPIES} copies cho moi mau)...")
    for i in range(N_COPIES):
        # Augment
        X_aug = augment(X_t, cfg, gen)
        # Convert back to uint8 [0, 255]
        X_aug_np = (X_aug.numpy() * 255).clip(0, 255).astype(np.uint8)
        all_X.append(X_aug_np)
        all_y.append(y_raw)
        
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    
    print(f"Tong cong sinh ra {len(final_X)} mau tu {len(X_raw)} mau goc.")
    
    out_path = 'data/output/hard_cases_augmented.npz'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=final_X, y=final_y)
    print(f"Da luu thanh cong vao {out_path}")

if __name__ == '__main__':
    main()
