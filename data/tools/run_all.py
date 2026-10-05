"""Tai du lieu goc va sinh lai TOAN BO cac bo du lieu tu dong (01..04, va 05 neu da co self_digits.npz).

  python tools/run_all.py                    # kich thuoc mac dinh trong tung config.json
  python tools/run_all.py --n-train 11000 --n-val 1100 --workers 4   # ban nho de thu nhanh

Moi bo duoc ghi vao data/output/<ten>/{train,val}.npz. Sau do dung tools/merge_datasets.py de tron.
"""
import argparse
import os
import subprocess
import sys

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["01_mnist_random_concat", "02_mnist_style_matched", "03_qmnist_same_writer", "04_ligature_strokes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    a = ap.parse_args()

    subprocess.check_call([sys.executable, os.path.join(DATA, "tools", "download_data.py")])
    methods = list(a.methods)
    if os.path.exists(os.path.join(DATA, "output", "self_digits.npz")) and "05_self_handwritten" not in methods:
        methods.append("05_self_handwritten")
    for m in methods:
        cmd = [sys.executable, os.path.join(DATA, m, "generate.py")]
        for flag, val in (("--n-train", a.n_train), ("--n-val", a.n_val), ("--workers", a.workers), ("--seed", a.seed)):
            if val is not None:
                cmd += [flag, str(val)]
        print("\n=== ", m)
        subprocess.check_call(cmd, cwd=os.path.join(DATA, m))


if __name__ == "__main__":
    main()
