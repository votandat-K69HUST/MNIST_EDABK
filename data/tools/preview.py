"""Xuat luoi anh mau (PNG) tu mot file npz do TU SINH (train/val) de kiem tra mat thuong.

CHI dung cho du lieu tu tao. KHONG dung cho test.csv (luat cuoc thi cam xem anh test).
Chay: python tools/preview.py output/m04_ligature_strokes/train.npz [--per-class 12] [--out preview.png]
"""
import argparse
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    z = np.load(a.npz)
    X, y = z["X"], z["y"]
    rows = []
    for lab in range(10, 21):
        idx = np.flatnonzero(y == lab)[:a.per_class]
        row = np.zeros((28, 28 * a.per_class), dtype=np.uint8)
        for k, i in enumerate(idx):
            row[:, k * 28:(k + 1) * 28] = X[i]
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    im = Image.fromarray(grid).resize((grid.shape[1] * a.scale, grid.shape[0] * a.scale), Image.NEAREST)
    out = a.out or os.path.splitext(a.npz)[0] + "_preview.png"
    im.save(out)
    print("da luu", out)


if __name__ == "__main__":
    main()
