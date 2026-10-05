"""Doc/ghi du lieu: tai file goc (MNIST, QMNIST), parse IDX, doc/ghi npz.

Quy uoc du lieu sau khi load (dict):
    X       uint8 (N, 28, 28)   nen den (0), net trang (255)
    digit   int   (N,)          chu so 0..9
    writer  int   (N,)          id nguoi viet, -1 neu khong co thong tin
"""
import gzip
import hashlib
import lzma
import os
import struct
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)  # .../data
DEFAULT_RAW_DIR = os.path.join(DATA_DIR, "raw")
DEFAULT_OUT_DIR = os.path.join(DATA_DIR, "output")

MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

QMNIST_BASE = "https://raw.githubusercontent.com/facebookresearch/qmnist/master/"
# (ten file, md5) - md5 lay tu torchvision.datasets.QMNIST
QMNIST_FILES = [
    ("qmnist-train-images-idx3-ubyte.gz", "ed72d4157d28c017586c42bc6afe6370"),
    ("qmnist-train-labels-idx2-int.gz", "0058f8dd561b90ffdd0f734c6a30e5e4"),
    ("qmnist-test-images-idx3-ubyte.gz", "1394631089c404de565df7b7aeaf9412"),
    ("qmnist-test-labels-idx2-int.gz", "5b5b05890a5e13444e108efe57b788aa"),
]
# Tuy chon (lon): toan bo chu so NIST SD19 co writer id (~400k anh)
NIST_FILES = [
    ("xnist-images-idx3-ubyte.xz", "7f124b3b8ab81486c9d8c2749c17f834"),
    ("xnist-labels-idx2-int.xz", "5ed0e788978e45d4a8bd4b7caec3d79d"),
]

# Cot trong file nhan QMNIST (8 cot/anh): 0=lop, 1=HSF partition, 2=writer id, ...
QMNIST_COL_CLASS = 0
QMNIST_COL_WRITER = 2


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expect_md5=None):
    """Tai url -> dest (bo qua neu da co va md5 khop)."""
    if os.path.exists(dest) and (expect_md5 is None or md5(dest) == expect_md5):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[download] {url}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    if expect_md5 is not None and md5(dest) != expect_md5:
        raise RuntimeError(f"md5 khong khop cho {dest}")
    return dest


def download_mnist(raw_dir=DEFAULT_RAW_DIR):
    d = os.path.join(raw_dir, "mnist")
    for fn in MNIST_FILES:
        download(MNIST_BASE + fn, os.path.join(d, fn))


def download_qmnist(raw_dir=DEFAULT_RAW_DIR, with_nist=False):
    d = os.path.join(raw_dir, "qmnist")
    files = QMNIST_FILES + (NIST_FILES if with_nist else [])
    for fn, h in files:
        download(QMNIST_BASE + fn, os.path.join(d, fn), h)


def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".xz"):
        return lzma.open(path, "rb")
    return open(path, "rb")


def read_idx(path):
    """Doc file IDX (ubyte hoac int32) -> ndarray."""
    with _open(path) as f:
        data = f.read()
    _, _, dtype_code, ndim = struct.unpack(">BBBB", data[:4])
    dims = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    dtype = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4",
             0x0D: ">f4", 0x0E: ">f8"}[dtype_code]
    arr = np.frombuffer(data, dtype=dtype, offset=4 + 4 * ndim)
    return arr.reshape(dims).copy()


def load_mnist(raw_dir=DEFAULT_RAW_DIR):
    """MNIST 70k anh (train 60k + test 10k). Khong co writer id."""
    d = os.path.join(raw_dir, "mnist")
    xs, ys = [], []
    for split in ("train", "t10k"):
        xs.append(read_idx(os.path.join(d, f"{split}-images-idx3-ubyte.gz")))
        ys.append(read_idx(os.path.join(d, f"{split}-labels-idx1-ubyte.gz")))
    X = np.concatenate(xs).astype(np.uint8)
    y = np.concatenate(ys).astype(np.int64)
    return {"X": X, "digit": y, "writer": np.full(len(y), -1, dtype=np.int64)}


def load_qmnist(raw_dir=DEFAULT_RAW_DIR, include_nist=False):
    """QMNIST: train 60k + test 60k, co writer id. include_nist=True dung ca xnist (~400k)."""
    d = os.path.join(raw_dir, "qmnist")
    if include_nist:
        names = [("xnist-images-idx3-ubyte.xz", "xnist-labels-idx2-int.xz")]
    else:
        names = [("qmnist-train-images-idx3-ubyte.gz", "qmnist-train-labels-idx2-int.gz"),
                 ("qmnist-test-images-idx3-ubyte.gz", "qmnist-test-labels-idx2-int.gz")]
    xs, ds, ws = [], [], []
    for img_fn, lab_fn in names:
        xs.append(read_idx(os.path.join(d, img_fn)))
        lab = read_idx(os.path.join(d, lab_fn))
        ds.append(lab[:, QMNIST_COL_CLASS])
        ws.append(lab[:, QMNIST_COL_WRITER])
    X = np.concatenate(xs).astype(np.uint8)
    return {"X": X,
            "digit": np.concatenate(ds).astype(np.int64),
            "writer": np.concatenate(ws).astype(np.int64)}


def load_self(path=None):
    """Chu so tu viet tay (tao boi 05_self_handwritten/build_digit_pool.py)."""
    path = path or os.path.join(DEFAULT_OUT_DIR, "self_digits.npz")
    z = np.load(path)
    return {"X": z["X"].astype(np.uint8), "digit": z["digit"].astype(np.int64),
            "writer": z["writer"].astype(np.int64)}


def load_source(kind, raw_dir=DEFAULT_RAW_DIR, self_path=None):
    if kind == "mnist":
        return load_mnist(raw_dir)
    if kind == "qmnist":
        return load_qmnist(raw_dir)
    if kind == "nist":
        return load_qmnist(raw_dir, include_nist=True)
    if kind == "self":
        return load_self(self_path)
    raise ValueError(f"source khong hop le: {kind}")


def save_dataset(path, X, y, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=np.asarray(X, dtype=np.uint8),
                        y=np.asarray(y, dtype=np.int64), **meta)


def load_dataset(path):
    """Tra ve (X uint8 (N,28,28), y int (N,)) tu file npz."""
    z = np.load(path)
    return z["X"], z["y"]
