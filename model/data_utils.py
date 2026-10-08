"""Nap du lieu train/val tu cac file npz do data/ sinh ra, tron theo trong so."""
import glob
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(path, base):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def _load(path):
    z = np.load(path)
    return z["X"], z["y"]


def load_train(cfg, base, seed):
    """Tron cac nguon train theo trong so. Tra ve (X uint8 (N,28,28), y int (N,), mo ta nguon)."""
    dcfg = cfg["data"]
    srcs = []
    for s in dcfg["train"]:
        paths = sorted(glob.glob(resolve(s["path"], base)))
        if not paths:
            print(f"[canh bao] khong tim thay {s['path']} - bo qua")
            continue
        for p in paths:
            X, y = _load(p)
            srcs.append((p, X, y, float(s.get("weight", 1.0)) / len(paths)))
    if not srcs:
        raise FileNotFoundError("Khong co nguon train nao. Chay: python data/tools/run_all.py")
    wsum = sum(s[3] for s in srcs)
    total = int(dcfg.get("total_train") or sum(len(s[1]) for s in srcs))
    rng = np.random.default_rng(seed)
    Xs, ys, desc = [], [], []
    for p, X, y, w in srcs:
        n = int(round(total * w / wsum))
        idx = rng.permutation(len(X))[:n] if n <= len(X) else rng.choice(len(X), n, replace=True)
        Xs.append(X[idx])
        ys.append(y[idx])
        desc.append(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}:{n}")
    X, y = np.concatenate(Xs), np.concatenate(ys)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], desc


def load_val(cfg, base):
    """dict ten -> (X uint8, y). Nhom 'real' (tu viet that) neu co; nhom 'synthetic' tu data/."""
    out = {}
    vcfg = cfg["data"].get("val", {})
    for group in ("synthetic", "real"):
        for pat in vcfg.get(group, []):
            for p in sorted(glob.glob(resolve(pat, base))):
                name = f"{group}:{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
                out[name] = _load(p)
    return out


def to_tensor(X):
    return torch.from_numpy(X.astype(np.float32) / 255.0)
