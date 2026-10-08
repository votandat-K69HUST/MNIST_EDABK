"""Tim hyperparameter bang random search (chon theo diem validation, KHONG dung test).

  python tune.py --trials 20 --epochs 12                       # khong gian mac dinh
  python tune.py --trials 30 --epochs 15 --space my_space.json # khong gian tuy chinh
  python tune.py --trials 20 --subsample 60000                 # train tren 60k mau cho nhanh

Khong gian tim kiem: dict 'duong.dan.tham.so' -> {"choice": [...]} | {"uniform": [lo,hi]} | {"loguniform": [lo,hi]}
Ket qua: runs/tune_<name>/results.csv (moi lan thu), best_config.json (cau hinh tot nhat; copy de huan luyen lai).
Luu y: chon theo val 'real' (tu viet) neu co, neu khong thi val synthetic (co the lac quan hon thuc te).
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_utils import load_train, load_val, to_tensor  # noqa: E402
from mlp import LABEL_OFFSET  # noqa: E402
from train import apply_overrides, load_config, train_one  # noqa: E402

DEFAULT_SPACE = {
    "model.hidden": {"choice": [[256], [512, 256], [1024, 512], [1024, 512, 256], [512, 512, 256], [2048, 1024, 512]]},
    "model.dropout": {"choice": [0.0, 0.1, 0.2, 0.3, 0.4]},
    "model.batchnorm": {"choice": [True, False]},
    "model.activation": {"choice": ["relu", "gelu", "silu"]},
    "train.lr": {"loguniform": [3e-4, 5e-3]},
    "train.weight_decay": {"loguniform": [1e-6, 1e-2]},
    "train.batch_size": {"choice": [128, 256, 512]},
    "train.label_smoothing": {"choice": [0.0, 0.05, 0.1]},
    "train.optimizer": {"choice": ["adamw", "sgd"]},
    "augment.rotate": {"choice": [4, 8, 12, 15]},
    "augment.shear": {"choice": [5, 10, 15]},
    "augment.shift": {"choice": [1, 2, 3]},
    "augment.noise_std": {"choice": [0.0, 0.03, 0.06]},
    "augment.thickness_p": {"choice": [0.0, 0.3, 0.5]},
}


def sample(space, rng):
    out = {}
    for k, spec in space.items():
        if "choice" in spec:
            v = spec["choice"][rng.integers(len(spec["choice"]))]
        elif "uniform" in spec:
            v = float(rng.uniform(*spec["uniform"]))
        elif "loguniform" in spec:
            lo, hi = spec["loguniform"]
            v = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            raise ValueError(k)
        out[k] = v.item() if isinstance(v, np.generic) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--space", help="file JSON khong gian tim kiem")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=12, help="so epoch moi lan thu (ngan hon train that)")
    ap.add_argument("--subsample", type=int, default=0, help="so mau train moi lan thu (0 = tat ca)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", help="ghi de co dinh: data.total_train=50000")
    a = ap.parse_args()

    cfg0 = apply_overrides(load_config(a.config), a.set)
    base = os.path.dirname(os.path.abspath(a.config))
    space = json.load(open(a.space, encoding="utf-8")) if a.space else DEFAULT_SPACE
    if a.subsample:
        cfg0["data"]["total_train"] = a.subsample
    cfg0["train"]["epochs"] = a.epochs
    cfg0["train"]["patience"] = max(3, a.epochs // 3)
    out_dir = os.path.join(HERE, "runs", f"tune_{cfg0.get('name', 'run')}")
    os.makedirs(out_dir, exist_ok=True)

    # nap du lieu 1 lan, dung chung cho moi lan thu
    Xtr, ytr, desc = load_train(cfg0, base, a.seed)
    val = {k: (to_tensor(X), torch.from_numpy(y - LABEL_OFFSET).long()) for k, (X, y) in load_val(cfg0, base).items()}
    cache = (Xtr, ytr, desc, val)
    print(f"train {len(Xtr)} mau; val {list(val)}")

    rng = np.random.default_rng(a.seed)
    rows, best = [], None
    fields = ["trial", "score", "best_epoch", "params", "seconds"] + list(space) + ["val_detail"]
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in range(a.trials):
            hp = sample(space, rng)
            cfg = copy.deepcopy(cfg0)
            for k, v in hp.items():
                d = cfg
                ks = k.split(".")
                for kk in ks[:-1]:
                    d = d[kk]
                d[ks[-1]] = v
            t0 = time.time()
            r = train_one(cfg, a.seed, os.path.join(out_dir, f"trial{t:03d}"), base, verbose=False, data_cache=cache)
            row = {"trial": t, "score": round(r["score"], 4), "best_epoch": r["best_epoch"], "params": r["params"],
                   "seconds": round(time.time() - t0), **{k: json.dumps(v) for k, v in hp.items()},
                   "val_detail": json.dumps({k: round(v, 4) for k, v in r["val"].items()})}
            w.writerow(row)
            f.flush()
            print(f"[{t + 1}/{a.trials}] score {r['score']:.4f} ({row['seconds']}s) {hp}")
            if best is None or r["score"] > best[0]:
                best = (r["score"], cfg)
                bc = copy.deepcopy(cfg)
                bc["base_dir"] = base  # de train.py/predict.py giai quyet duong dan tuong doi dung
                with open(os.path.join(out_dir, "best_config.json"), "w", encoding="utf-8") as g:
                    json.dump(bc, g, indent=2, ensure_ascii=False)
    print(f"\nTOT NHAT: score {best[0]:.4f}\n  -> {os.path.join(out_dir, 'best_config.json')}")
    print("Huan luyen lai day du:  python train.py --config runs/tune_%s/best_config.json --name <ten> --set train.epochs=40"
          % cfg0.get("name", "run"))


if __name__ == "__main__":
    main()
