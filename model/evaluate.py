"""Danh gia ensemble MLP tren mot tap CO NHAN (csv hoac npz), vd tap tu ve.

  python evaluate.py --data "../ve_tay_10_20 - data.csv" --run runs/mlp_v2_simple
  python evaluate.py --data ../data/output/self_numbers_val.npz --run runs/mlp_v2_simple

CSV: cot 'label' (10..20), cot pixel0..pixel783, tuy chon 'writer'. KHONG dung cho test.csv (khong co nhan).
Ghi ra runs/<run>/eval_<ten_tap>.json va eval_<ten_tap>_confusion.csv
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

from predict import load_models, predict_proba  # noqa: E402


def load_labeled(path):
    if path.endswith(".npz"):
        z = np.load(path)
        X, y = z["X"].astype(np.float32) / 255.0, z["y"]
        w = z["writer"] if "writer" in z.files else None
        return X, y.astype(int), w
    df = pd.read_csv(path)
    px = [c for c in df.columns if c.startswith("pixel")]
    X = df[px].values.reshape(-1, 28, 28).astype(np.float32) / 255.0
    w = df["writer"].values if "writer" in df.columns else None
    return X, df["label"].values.astype(int), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tta", type=int, default=0)
    a = ap.parse_args()
    run = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    paths = sorted(glob.glob(os.path.join(run, "model_seed*.pt")))
    models = load_models(paths)
    X, y, writer = load_labeled(a.data)
    Xt = torch.from_numpy(X)
    P = predict_proba(models, Xt, a.tta).numpy()
    pred = P.argmax(1) + 10
    out = {"data": a.data, "run": run, "n": int(len(y)), "ensemble_accuracy": float((pred == y).mean())}
    out["per_seed_accuracy"] = {os.path.basename(p): float((predict_proba([m], Xt, a.tta).argmax(1).numpy() + 10 == y).mean())
                                for p, m in zip(paths, models)}
    out["per_class_accuracy"] = {int(c): float((pred[y == c] == c).mean()) for c in range(10, 21)}
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).reindex(
        index=range(10, 21), columns=range(10, 21), fill_value=0)
    cm_path = os.path.join(run, f"eval_{os.path.splitext(os.path.basename(a.data))[0].replace(' ', '_')}_confusion.csv")
    cm.to_csv(cm_path)
    conf = P.max(1)
    out["mean_confidence"] = float(conf.mean())
    out["acc_when_conf>=0.9"] = float((pred == y)[conf >= 0.9].mean()) if (conf >= 0.9).any() else None
    out["frac_conf>=0.9"] = float((conf >= 0.9).mean())
    errs = [(int(t), int(p)) for t, p in zip(y[pred != y], pred[pred != y])]
    top = pd.Series([f"{t}->{p}" for t, p in errs]).value_counts().head(12)
    out["top_confusions"] = {k: int(v) for k, v in top.items()}
    if writer is not None:
        ok = pd.DataFrame({"w": writer, "ok": pred == y}).groupby("w")["ok"].agg(["mean", "size"])
        out["n_writers"] = int(len(ok))
        out["writer_accuracy_mean"] = float(ok["mean"].mean())
        out["writer_accuracy_min"] = float(ok["mean"].min())
        out["writers_below_80"] = {str(k): [round(float(r["mean"]), 3), int(r["size"])] for k, r in ok.iterrows() if r["mean"] < 0.8}
    name = os.path.splitext(os.path.basename(a.data))[0].replace(" ", "_")
    with open(os.path.join(run, f"eval_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nConfusion matrix (hang = dung, cot = du doan):")
    print(cm.to_string())


if __name__ == "__main__":
    main()
