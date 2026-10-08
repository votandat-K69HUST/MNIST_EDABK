import pandas as pd
import numpy as np

def analyze():
    print("Loading ve_tay_10_20 - data.csv...")
    df = pd.read_csv('ve_tay_10_20 - data.csv')
    pixel_cols = [f'pixel{i}' for i in range(784)]
    X = df[pixel_cols].values
    
    means = X.mean(axis=1)
    active_pixels = (X > 128).sum(axis=1)
    
    X_2d = X.reshape(-1, 28, 28)
    aspect_ratios = []
    for img in X_2d:
        rows = np.any(img > 50, axis=1)
        cols = np.any(img > 50, axis=0)
        if not np.any(rows) or not np.any(cols):
            aspect_ratios.append(0)
            continue
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        aspect_ratios.append(w / h)
    aspect_ratios = np.array(aspect_ratios)
    
    print(f"Total samples: {len(df)}")
    
    # Thresholds
    inverted = means > 80
    empty = active_pixels < 20
    too_thick = active_pixels > 350
    crazy_aspect = (aspect_ratios > 2.0) | (aspect_ratios < 0.3)
    
    bad_idx = inverted | empty | too_thick | crazy_aspect
    
    print(f"Detected noisy samples: {bad_idx.sum()}")
    print(f"  - Inverted/Too bright: {inverted.sum()}")
    print(f"  - Empty/Too faint: {empty.sum()}")
    print(f"  - Too thick: {too_thick.sum()}")
    print(f"  - Crazy aspect ratio (Too tilted/wide/narrow): {crazy_aspect.sum()}")
    
    good_active = active_pixels[~bad_idx]
    print(f"\nGood samples stats:")
    print(f"  - Avg active pixels (thickness): {good_active.mean():.1f} (std: {good_active.std():.1f})")
    print(f"  - Avg mean intensity: {means[~bad_idx].mean():.1f}")
    
    df_clean = df[~bad_idx]
    df_clean.to_csv('ve_tay_10_20_cleaned.csv', index=False)
    print(f"\nSaved {len(df_clean)} clean samples to ve_tay_10_20_cleaned.csv")

if __name__ == '__main__':
    analyze()
