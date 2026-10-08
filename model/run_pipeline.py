"""Pipeline tu dau den cuoi: (sinh du lieu neu thieu) -> [tune] -> train ensemble -> predict -> output.csv

  python run_pipeline.py                      # train theo config.json roi du doan
  python run_pipeline.py --tune 20            # random search 20 lan truoc, train lai bang config tot nhat
  python run_pipeline.py --skip-train         # chi du doan bang model da co

Moi buoc cung chay doc lap duoc: train.py, tune.py, predict.py (xem README.md).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--tune", type=int, default=0, help="so lan thu random search (0 = bo qua)")
    ap.add_argument("--tune-epochs", type=int, default=12)
    ap.add_argument("--tune-subsample", type=int, default=60000)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--out", help="duong dan output.csv")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    cfg_path = a.config
    if not a.skip_data and not os.path.exists(os.path.join(ROOT, "data", "output", "m04_ligature_strokes", "train.npz")):
        run([PY, os.path.join(ROOT, "data", "tools", "run_all.py"), "--workers", "4"])
    if a.tune and not a.skip_train:
        run([PY, "tune.py", "--config", a.config, "--trials", str(a.tune), "--epochs", str(a.tune_epochs),
             "--subsample", str(a.tune_subsample)])
        best = os.path.join(HERE, "runs", f"tune_{cfg.get('name', 'run')}", "best_config.json")
        b = json.load(open(best, encoding="utf-8"))
        b["train"]["epochs"] = cfg["train"]["epochs"]
        b["train"]["patience"] = cfg["train"].get("patience")
        b["data"]["total_train"] = cfg["data"].get("total_train", 0)
        b["name"] = cfg.get("name", "run") + "_tuned"
        b["ensemble"] = cfg.get("ensemble", {"seeds": [0, 1, 2]})
        b["predict"] = cfg.get("predict", {})
        cfg_path = os.path.join(HERE, "runs", f"tune_{cfg.get('name', 'run')}", "final_config.json")
        json.dump(b, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if not a.skip_train:
        run([PY, "train.py", "--config", cfg_path])
    cmd = [PY, "predict.py", "--config", cfg_path]
    if a.out:
        cmd += ["--out", a.out]
    run(cmd)


if __name__ == "__main__":
    main()
