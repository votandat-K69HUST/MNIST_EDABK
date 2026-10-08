"""Chuyen anh PNG tu viet (collect.py) thanh file npz kieu MNIST 28x28.

  --mode numbers : so hai chu so (nhan 10..20) -> output/self_numbers.npz (X, y, writer)
                   dung lam tap VALIDATION thuc te (hoac train them)
  --mode digits  : tung chu so 0..9 -> output/self_digits.npz (X, digit, writer)
                   dung lam nguon cho pairing same_writer (source.kind = "self")

Bo cuc cua numbers: --layout aspect|stretch (giong common/layout.py), --box 20.. cho 'aspect'.
Chia theo nguoi viet: --val-writers a,b -> self_numbers_val.npz va self_numbers_train.npz
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common.features import crop  # noqa: E402
from common.imgops import mnist_normalize, paste_center, resize, to_uint8  # noqa: E402
from common.io_utils import DEFAULT_OUT_DIR  # noqa: E402


def load_png(path):
    # canvas nen den, net trang; thu nho bang BOX (trung binh dien tich) nhu pipeline sinh du lieu
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def number_to_28(img, layout, box):
    c = crop(img, thr=0.05)
    h, w = c.shape
    if layout == "aspect":
        s = box / max(h, w)
        out = paste_center(resize(c, round(h * s), round(w * s)), 28, "com")
    else:  # stretch
        out = paste_center(resize(c, 24, 24), 28, "bbox")
    m = out.max()
    return out / m if m > 0 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["numbers", "digits"], required=True)
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--layout", choices=["aspect", "stretch"], default="aspect")
    ap.add_argument("--box", type=float, default=20.0)
    ap.add_argument("--val-writers", default="", help="danh sach writer cho tap val, cach nhau dau phay")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    a = ap.parse_args()

    sub = "number" if a.mode == "numbers" else "digit"
    paths = sorted(glob.glob(os.path.join(a.samples, "*", sub, "*.png")))
    if not paths:
        sys.exit(f"khong tim thay PNG trong {a.samples}/<writer>/{sub}/")
    writer_of = lambda p: os.path.normpath(p).split(os.sep)[-3]  # noqa: E731
    writers = sorted({writer_of(p) for p in paths})
    wid = {w: i for i, w in enumerate(writers)}
    X, lab, wr = [], [], []
    for p in paths:
        img = load_png(p)
        if img.max() <= 0:
            continue
        x = number_to_28(img, a.layout, a.box) if a.mode == "numbers" else mnist_normalize(img, 20)
        X.append(to_uint8(x))
        lab.append(int(os.path.basename(p).split("_")[0]))
        wr.append(wid[writer_of(p)])
    X, lab, wr = np.stack(X), np.array(lab), np.array(wr)
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"{len(X)} anh, writers: {wid}")
    if a.mode == "digits":
        np.savez_compressed(os.path.join(a.out_dir, "self_digits.npz"), X=X, digit=lab, writer=wr)
        print("da luu self_digits.npz (nguon cho source.kind='self')")
        return
    np.savez_compressed(os.path.join(a.out_dir, "self_numbers.npz"), X=X, y=lab, writer=wr)
    val_set = [wid[w] for w in a.val_writers.split(",") if w]
    if val_set:
        m = np.isin(wr, val_set)
        np.savez_compressed(os.path.join(a.out_dir, "self_numbers_val.npz"), X=X[m], y=lab[m], writer=wr[m])
        np.savez_compressed(os.path.join(a.out_dir, "self_numbers_train.npz"), X=X[~m], y=lab[~m], writer=wr[~m])
        print(f"val: {m.sum()} anh, train: {(~m).sum()} anh")
    print("da luu self_numbers.npz")


if __name__ == "__main__":
    main()
