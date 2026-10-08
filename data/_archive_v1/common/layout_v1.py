"""Dat so hai chu so da ghep (canvas phan giai cao) vao anh 28x28 kieu MNIST.

De bai chi noi "chuan hoa theo format tuong tu MNIST" nen KHONG biet chinh xac to chuc dat 2 chu so
vao 28x28. Vi vay co nhieu che do bo cuc, tron theo trong so trong config:

  aspect : giu ti le, vua khung box x box (box ngau nhien trong khoang), can tam theo 'com' hoac 'bbox'
           (giong cach MNIST xu ly: box=20, can theo trong tam)
  stretch: keo gian doc lap ngang/doc de lap day anh (so cao & hep)

Khi da co thong ke tong hop cua test.csv (tools/stats.py), chon lai weights cho hop ly.
"""
import numpy as np

from .features import crop
from .imgops import paste_center, resize


def apply_layout(canvas, layouts_cfg, rng):
    """canvas: float 0..1 (nen den). Tra ve (anh 28x28 float 0..1, ten layout)."""
    c = crop(canvas, thr=0.05)
    names = list(layouts_cfg.keys())
    weights = np.array([layouts_cfg[n].get("weight", 1.0) for n in names], dtype=np.float64)
    name = names[rng.choice(len(names), p=weights / weights.sum())]
    cfg = layouts_cfg[name]
    h, w = c.shape
    if name == "aspect":
        box = rng.uniform(*cfg.get("box", [20, 24]))
        sq = cfg.get("squeeze")  # ep ngang ca so (viet "hep"): nhan chieu rong voi he so ngau nhien
        if sq:
            w = max(int(round(w * rng.uniform(*sq))), 1)
            c = resize(c, h, w)
        s = box / max(h, w)
        patch = resize(c, round(h * s), round(w * s))
    elif name == "stretch":
        tw = rng.uniform(*cfg.get("w", [22, 26]))
        th = rng.uniform(*cfg.get("h", [18, 26]))
        patch = resize(c, round(th), round(tw))
    else:
        raise ValueError(f"layout khong hop le: {name}")
    shift_max = int(cfg.get("shift", 1))
    shift = (rng.integers(-shift_max, shift_max + 1), rng.integers(-shift_max, shift_max + 1))
    out = paste_center(patch, 28, cfg.get("center", "com"), shift, cfg.get("subpixel", False))
    m = out.max()
    if m > 0:
        out = out / m
    return out, name
