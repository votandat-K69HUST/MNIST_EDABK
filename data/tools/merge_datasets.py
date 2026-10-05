"""Tron nhieu file npz (X, y) thanh 1 file huan luyen, co trong so / gioi han so mau tung nguon.

Vi du: 60% ghep cung writer, 25% co net noi, 15% tu viet:
  python tools/merge_datasets.py --out output/final_train.npz \
      output/m03_qmnist_same_writer/train.npz:0.6 \
      output/m04_ligature_strokes/train.npz:0.25 \
      output/self_numbers_train.npz:0.15 --total 60000

Ngoai ra ghi them mang 'source' (chi so nguon) de phan tich loi theo nguon.
Neu nguon co it mau hon so duoc yeu cau thi lap lai (replace=True) mau cua nguon do.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.io_utils import save_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="file.npz[:trong_so]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, help="tong so mau (mac dinh: tong so mau cua cac nguon nhan trong so)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    srcs = []
    for s in a.inputs:
        path, w = s, 1.0
        head, sep, tail = s.rpartition(":")
        if sep:
            try:  # "file.npz:0.6"; duong dan Windows "C:\x.npz" khong parse duoc thanh so nen giu nguyen
                w, path = float(tail), head
            except ValueError:
                pass
        z = np.load(path)
        srcs.append((path, z["X"], z["y"], w))
    wsum = sum(s[3] for s in srcs)
    total = a.total or int(sum(len(s[1]) for s in srcs))
    rng = np.random.default_rng(a.seed)
    Xs, ys, ss = [], [], []
    for k, (path, X, y, w) in enumerate(srcs):
        n = int(round(total * w / wsum))
        idx = rng.choice(len(X), size=n, replace=n > len(X))
        Xs.append(X[idx])
        ys.append(y[idx])
        ss.append(np.full(n, k, dtype=np.int8))
        print(f"{path}: {n} mau (nguon co {len(X)})")
    X, y, s = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
    p = rng.permutation(len(X))
    save_dataset(a.out, X[p], y[p], source=s[p])
    print(f"da luu {a.out}: {len(X)} mau; nguon: {[x[0] for x in srcs]}")


if __name__ == "__main__":
    main()
