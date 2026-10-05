"""Augmentation offline mot file npz (X, y) -> file npz moi.

Vi du:
  python augment_dataset.py --in ../output/m03_qmnist_same_writer/train.npz --factor 3 --preset medium
  -> ../output/m03_qmnist_same_writer/train_aug_medium_x3.npz  (gom ban goc + 3 ban augment)

Chi augment tap TRAIN. Khong augment tap val (de do dung kha nang tong quat).
Cho huan luyen online, import: from common.augment import augment_batch
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common.augment import augment_batch  # noqa: E402
from common.io_utils import save_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--factor", type=int, default=3, help="so ban augment moi anh goc")
    ap.add_argument("--no-original", action="store_true", help="khong giu ban goc")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)[a.preset]
    z = np.load(a.inp)
    X, y = z["X"], z["y"]
    rng = np.random.default_rng(a.seed)
    Xs = [] if a.no_original else [X]
    ys = [] if a.no_original else [y]
    for k in range(a.factor):
        Xs.append(augment_batch(X, rng, cfg))
        ys.append(y)
        print(f"ban augment {k + 1}/{a.factor} xong")
    Xo, yo = np.concatenate(Xs), np.concatenate(ys)
    perm = rng.permutation(len(Xo))
    out = a.out or os.path.splitext(a.inp)[0] + f"_aug_{a.preset}_x{a.factor}.npz"
    save_dataset(out, Xo[perm], yo[perm])
    print(f"da luu {out}: {len(Xo)} anh")


if __name__ == "__main__":
    main()
