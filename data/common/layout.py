"""Dat so hai chu so (canvas phan giai cao) vao anh 28x28 theo quy uoc chuan cua MNIST:
cat sat net, giu ti le de vua khung box x box (mac dinh 20x20), can tam theo trong tam (center of mass).
"""
from .features import crop
from .imgops import paste_center, resize


def apply_layout(canvas, lcfg):
    """canvas: float 0..1 (nen den). lcfg: {"box": 20, "center": "com"}. Tra ve anh 28x28 float 0..1."""
    c = crop(canvas, thr=0.05)
    h, w = c.shape
    s = lcfg.get("box", 20) / max(h, w)
    patch = resize(c, round(h * s), round(w * s))
    out = paste_center(patch, 28, lcfg.get("center", "com"))
    m = out.max()
    return out / m if m > 0 else out
