"""Augmentation tren anh 28x28 (numpy/scipy) - dung offline (06_augmentation/augment_dataset.py)
hoac online trong vong huan luyen: augment_batch(X, rng, cfg).

Anh vao: uint8 hoac float 0..1, nen den. Anh ra: cung dtype voi dau vao.
Vi dung MLP (khong bat bien dich chuyen nhu CNN) nen augmentation hinh hoc rat quan trong.
"""
import numpy as np
from scipy import ndimage

_CROSS = ndimage.generate_binary_structure(2, 1)


def _rand(rng, lohi):
    return rng.uniform(lohi[0], lohi[1])


def random_affine(img, rng, cfg):
    """Xoay, scale (rieng ngang/doc), shear, dich chuyen quanh tam anh."""
    ang = np.deg2rad(_rand(rng, cfg.get("rotate", [-10, 10])))
    sx = _rand(rng, cfg.get("scale", [0.9, 1.1]))
    sy = sx * _rand(rng, cfg.get("aspect_jitter", [0.92, 1.08]))
    sh = np.tan(np.deg2rad(_rand(rng, cfg.get("shear", [-12, 12]))))
    ty = _rand(rng, cfg.get("shift", [-2, 2]))
    tx = _rand(rng, cfg.get("shift", [-2, 2]))
    ca, sa = np.cos(ang), np.sin(ang)
    rot = np.array([[ca, -sa], [sa, ca]])
    shear = np.array([[1.0, 0.0], [sh, 1.0]])  # (row, col): col += sh * row  (nghieng ngang)
    scale = np.diag([1.0 / sy, 1.0 / sx])
    m = rot @ shear @ scale
    center = np.array([13.5, 13.5])
    offset = center - m @ center + np.array([ty, tx])
    return ndimage.affine_transform(img, m, offset=offset, order=1, mode="constant", cval=0.0)


def elastic(img, rng, cfg):
    alpha, sigma = cfg.get("alpha", 3.0), cfg.get("sigma", 3.0)
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha * sigma
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha * sigma
    rr, cc = np.mgrid[0:28, 0:28]
    return ndimage.map_coordinates(img, [rr + dy, cc + dx], order=1, mode="constant")


def thickness(img, rng, cfg):
    """Day / manh net duoi muc pixel: tron anh goc voi ban dilate/erode (cross 3x3)."""
    lam = _rand(rng, cfg.get("strength", [0.3, 0.8]))
    if rng.random() < 0.5:
        alt = ndimage.grey_dilation(img, footprint=_CROSS)
    else:
        alt = ndimage.grey_erosion(img, footprint=_CROSS)
    return (1 - lam) * img + lam * alt


def augment_one(img, rng, cfg):
    """img float32 0..1 (28x28). cfg: dict cac khoi, moi khoi co 'p' (xac suat) + tham so."""
    x = img
    for name, fn in (("affine", random_affine), ("elastic", elastic), ("thickness", thickness)):
        c = cfg.get(name)
        if c and rng.random() < c.get("p", 1.0):
            x = fn(x, rng, c)
    c = cfg.get("gamma")
    if c and rng.random() < c.get("p", 1.0):
        x = np.clip(x, 0, 1) ** _rand(rng, c.get("range", [0.8, 1.25]))
    c = cfg.get("blur")
    if c and rng.random() < c.get("p", 1.0):
        x = ndimage.gaussian_filter(x, _rand(rng, c.get("sigma", [0.3, 0.8])))
    c = cfg.get("noise")
    if c and rng.random() < c.get("p", 1.0):
        std = _rand(rng, c.get("std", [0.02, 0.08]))
        x = x + rng.normal(0, std, x.shape) * (rng.random(x.shape) < c.get("density", 0.3))
    c = cfg.get("cutout")
    if c and rng.random() < c.get("p", 0.0):
        s = int(rng.integers(c.get("size", [2, 5])[0], c.get("size", [2, 5])[1] + 1))
        r, q = rng.integers(0, 28 - s + 1, size=2)
        x = x.copy()
        x[r:r + s, q:q + s] = 0
    x = np.clip(x, 0, 1)
    m = x.max()
    if cfg.get("renormalize", True) and m > 0:
        x = x / m
    return x.astype(np.float32)


def augment_batch(X, rng, cfg):
    """X: (N,28,28) uint8 hoac float 0..1 -> cung kieu."""
    is_u8 = X.dtype == np.uint8
    out = np.empty(X.shape, dtype=np.float32)
    for i in range(len(X)):
        xi = X[i].astype(np.float32) / (255.0 if is_u8 else 1.0)
        out[i] = augment_one(xi, rng, cfg)
    return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8) if is_u8 else out
