"""Dac trung phong cach cua mot anh chu so 28x28: do nghieng, do day net, kich thuoc.

Dung de ghep cac chu so "giong phong cach" (pairing = style_matched).
"""
import numpy as np


def bbox(img, thr=0.1):
    """(r0, r1, c0, c1) bao quanh pixel > thr (r1, c1 la chi so ket thuc, khong gom)."""
    m = img > thr
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if len(rows) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def crop(img, thr=0.1):
    r0, r1, c0, c1 = bbox(img, thr)
    return img[r0:r1, c0:c1]


def slant(img):
    """He so nghieng t = mu11/mu02 (cung cach deskew cua MNIST). t>0: nghieng sang phai."""
    img = img.astype(np.float64)
    tot = img.sum()
    if tot <= 0:
        return 0.0
    rr, cc = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    r_bar = (img * rr).sum() / tot
    c_bar = (img * cc).sum() / tot
    mu02 = (img * (rr - r_bar) ** 2).sum() / tot
    mu11 = (img * (rr - r_bar) * (cc - c_bar)).sum() / tot
    if mu02 < 1e-6:
        return 0.0
    # trong toa do anh (hang tang xuong duoi), net nghieng phai co mu11 < 0
    return float(-mu11 / mu02)


def thickness(img, thr=0.5):
    """Uoc luong do day net (pixel): 2 * dien tich / chu vi."""
    m = img > thr
    area = m.sum()
    if area == 0:
        return 0.0
    p = np.pad(m, 1)
    interior = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    boundary = (m & ~interior).sum()
    return float(2.0 * area / max(boundary, 1))


def style_features(img):
    """Vector dac trung [slant, thickness, aspect(w/h)] cho anh float 0..1."""
    r0, r1, c0, c1 = bbox(img)
    h, w = r1 - r0, c1 - c0
    return np.array([slant(img), thickness(img), w / max(h, 1)], dtype=np.float64)


def style_features_batch(X):
    """X uint8 (N,28,28) -> (N,3)."""
    return np.stack([style_features(x.astype(np.float32) / 255.0) for x in X])
