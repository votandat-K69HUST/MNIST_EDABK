"""Chon cap (chu so hang chuc, chu so hang don vi) de ghep thanh so.

Nhan 10..19 = "1" + d (d = nhan - 10); nhan 20 = "2" + "0".

2 che do:
  random      : chon ngau nhien trong pool (nguon bat ky, vd MNIST)
  same_writer : ca hai chu cung mot writer id (can nguon co writer: qmnist / nist / self)
"""
import numpy as np


def label_to_digits(label):
    if label == 20:
        return 2, 0
    if 10 <= label <= 19:
        return 1, label - 10
    raise ValueError(label)


class Pool:
    """Tap cac chi so anh thuoc mot split (train / val) cua nguon."""

    def __init__(self, source, idx):
        self.source = source
        self.idx = np.asarray(idx)
        digit = source["digit"][self.idx]
        writer = source["writer"][self.idx]
        self.by_digit = {d: self.idx[digit == d] for d in range(10)}
        self._ws_cache = {}
        self.by_writer = {}
        if (writer >= 0).any():
            for w in np.unique(writer[writer >= 0]):
                m = writer == w
                self.by_writer[int(w)] = {d: self.idx[m & (digit == d)] for d in range(10)}

    def writers_with(self, digits):
        return [w for w, dd in self.by_writer.items() if all(len(dd[d]) > 0 for d in digits)]


def split_pools(source, val_fraction, split_seed):
    """Chia nguon thanh (train_pool, val_pool). Neu co writer id: chia THEO WRITER (khong ro ri),
    nguoc lai chia theo chi so anh."""
    rng = np.random.default_rng(split_seed)
    N = len(source["digit"])
    writer = source["writer"]
    if (writer >= 0).all():
        ws = np.unique(writer)
        rng.shuffle(ws)
        n_val = max(int(round(len(ws) * val_fraction)), 1) if val_fraction > 0 else 0
        val_w = set(ws[:n_val].tolist())
        is_val = np.array([w in val_w for w in writer])
    else:
        is_val = np.zeros(N, dtype=bool)
        perm = rng.permutation(N)
        is_val[perm[:int(round(N * val_fraction))]] = True
    return Pool(source, np.flatnonzero(~is_val)), Pool(source, np.flatnonzero(is_val))


def sample_pair(pool, label, rng, pcfg):
    """Tra ve (idx_trai, idx_phai) la chi so vao source."""
    t, u = label_to_digits(label)
    mode = pcfg.get("mode", "random")
    if mode == "random":
        return rng.choice(pool.by_digit[t]), rng.choice(pool.by_digit[u])
    if mode == "same_writer":
        ws = pool._ws_cache.get((t, u))
        if ws is None:
            ws = pool.writers_with((t, u))
            pool._ws_cache[(t, u)] = ws
        w = ws[rng.integers(len(ws))]
        dd = pool.by_writer[w]
        return rng.choice(dd[t]), rng.choice(dd[u])
    raise ValueError(f"pairing mode khong hop le: {mode}")
