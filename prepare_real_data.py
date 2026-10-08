import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def main():
    print("Loading ve_tay_10_20_cleaned.csv...")
    df = pd.read_csv('ve_tay_10_20_cleaned.csv')
    
    # Extract X and y
    pixel_cols = [f'pixel{i}' for i in range(784)]
    X = df[pixel_cols].values
    
    # Reshape X to (N, 28, 28) and ensure uint8
    X = X.reshape(-1, 28, 28).astype(np.uint8)
    
    # Labels
    y = df['label'].values
    
    # Writers (map to integer IDs)
    writers = df['writer'].astype('category').cat.codes.values
    
    # Split into train and val (80/20 stratified by label)
    # Using stratify=y ensures both train and val have balanced classes
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, writers, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save NPZ files
    out_dir = 'data/output'
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, 'self_numbers_train.npz')
    np.savez_compressed(train_path, X=X_train, y=y_train, writer=w_train)
    print(f"Saved {train_path} - shape: {X_train.shape}, classes: {np.bincount(y_train)[10:]}")
    
    val_path = os.path.join(out_dir, 'self_numbers_val.npz')
    np.savez_compressed(val_path, X=X_val, y=y_val, writer=w_val)
    print(f"Saved {val_path} - shape: {X_val.shape}, classes: {np.bincount(y_val)[10:]}")

if __name__ == '__main__':
    main()
