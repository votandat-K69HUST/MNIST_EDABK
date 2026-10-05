"""Tai du lieu goc vao data/raw/ (bo qua file da co).

  python tools/download_data.py                 # MNIST + QMNIST (~ 100 MB)
  python tools/download_data.py --nist          # them xnist (toan bo chu so NIST SD19 co writer id, rat lon)

Nguon: MNIST (cvdf-datasets tren Google Storage), QMNIST (github.com/facebookresearch/qmnist).
QMNIST: 120.000 anh (train 60k + test 60k), 1074 writer, writer id o cot 2 cua file nhan.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import io_utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=io_utils.DEFAULT_RAW_DIR)
    ap.add_argument("--nist", action="store_true", help="tai them xnist (lon)")
    ap.add_argument("--skip-mnist", action="store_true")
    ap.add_argument("--skip-qmnist", action="store_true")
    a = ap.parse_args()
    if not a.skip_mnist:
        io_utils.download_mnist(a.raw_dir)
    if not a.skip_qmnist:
        io_utils.download_qmnist(a.raw_dir, with_nist=a.nist)
    print("xong ->", a.raw_dir)


if __name__ == "__main__":
    main()
