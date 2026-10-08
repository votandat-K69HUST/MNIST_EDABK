"""Chay inference tren test.csv bang ensemble MLP da train -> output.csv (id,label) dung dinh dang nop bai.

  python predict.py                                  # dung runs/<name>/model_seed*.pt theo config.json
  python predict.py --run runs/mlp_v1 --test ../test.csv --out ../output.csv

TUAN THU LUAT: test.csv chi duoc dung de chay inference tu dong. Script khong ve / hien thi anh test,
chi in thong ke tong hop (so dong, phan bo nhan du doan).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mlp import LABEL_OFFSET, build_model  # noqa: E402


def load_models(paths):
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = build_model(ck["config"]["model"], ck["mean"], ck["std"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def predict_proba(models, X, tta_shifts=0, bs=2048):
    """X: (N,28,28) float 0..1. Tra ve xac suat trung binh (N,11) cua cac model (+ TTA dich +-tta_shifts px)."""
    shifts = [(0, 0)]
    if tta_shifts > 0:
        s = tta_shifts
        shifts += [(s, 0), (-s, 0), (0, s), (0, -s)]
    out = torch.zeros(len(X), 11)
    for m in models:
        for dy, dx in shifts:
            for i in range(0, len(X), bs):
                xb = X[i:i + bs]
                if dy or dx:
                    xb = torch.roll(xb, shifts=(dy, dx), dims=(1, 2))
                out[i:i + bs] += torch.softmax(m(xb), dim=1)
    return out / (len(models) * len(shifts))


def read_test(path):
    df = pd.read_csv(path)
    ids = df["id"].values
    X = df.drop(columns=["id"]).values.reshape(-1, 28, 28).astype(np.float32) / 255.0
    return ids, torch.from_numpy(X)


def write_submission(ids, labels, path):
    sub = pd.DataFrame({"id": ids, "label": labels.astype(int)})
    assert len(sub) == len(ids) and sub["id"].is_unique, "id phai xuat hien dung mot lan"
    assert sub["label"].between(10, 20).all(), "label phai trong 10..20"
    sub.to_csv(path, index=False)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--run", help="thu muc runs/<name> chua model_seed*.pt")
    ap.add_argument("--test")
    ap.add_argument("--out")
    ap.add_argument("--tta", type=int, help="dich +-N px khi du doan (0 = tat)")
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_dir") or os.path.dirname(os.path.abspath(a.config))
    pcfg = cfg.get("predict", {})
    run = a.run or os.path.join(HERE, "runs", cfg.get("name", "run"))
    test = os.path.abspath(a.test or os.path.join(base, pcfg.get("test_csv", "../test.csv")))
    out = os.path.abspath(a.out or os.path.join(base, pcfg.get("output", "../output.csv")))
    tta = pcfg.get("tta_shifts", 0) if a.tta is None else a.tta

    paths = sorted(glob.glob(os.path.join(run, "model_seed*.pt")))
    if not paths:
        sys.exit(f"khong thay model trong {run}. Chay: python train.py")
    models = load_models(paths)
    ids, X = read_test(test)
    proba = predict_proba(models, X, tta)
    labels = proba.argmax(1).numpy() + LABEL_OFFSET
    sub = write_submission(ids, labels, out)
    print(f"{len(models)} model, TTA={tta}; {len(sub)} dong -> {out}")
    print("phan bo nhan du doan (ky vong ~50 moi lop):", {int(k): int(v) for k, v in sub["label"].value_counts().sort_index().items()})
    conf = proba.max(1).values
    print(f"do tin cay trung binh {conf.mean():.3f}; ty le mau < 0.6: {(conf < 0.6).float().mean():.3f}")


if __name__ == "__main__":
    main()
