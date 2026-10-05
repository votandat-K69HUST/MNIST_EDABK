"""Cac phep xu ly anh co ban (numpy/scipy/PIL) dung chung."""
import numpy as np
from PIL import Image
from scipy import ndimage

from .features import crop


def resize(a, new_h, new_w):
    """Resize anh float32 0..1. Thu nho dung BOX (trung binh dien tich), phong to dung BILINEAR."""
    new_h, new_w = max(int(new_h), 1), max(int(new_w), 1)
    h, w = a.shape
    im = Image.fromarray(a.astype(np.float32), mode="F")
    method = Image.BOX if (new_h < h or new_w < w) else Image.BILINEAR
    return np.asarray(im.resize((new_w, new_h), method), dtype=np.float32)


def shear_x(a, deg):
    """Nghieng ngang quanh tam (deg > 0: dinh chu nghieng sang phai). Mo rong canvas de khong cat."""
    t = np.tan(np.deg2rad(deg))
    if abs(t) < 1e-4:
        return a
    h, w = a.shape
    pad = int(np.ceil(abs(t) * h / 2)) + 2
    p = np.pad(a, ((0, 0), (pad, pad)))
    rc = (h - 1) / 2.0
    # anh vao (r, c + t*(r-rc)) -> dinh (r nho) lech sang phai khi t > 0
    mat = np.array([[1.0, 0.0], [t, 1.0]])
    off = np.array([0.0, -t * rc])
    return ndimage.affine_transform(p, mat, offset=off, order=1, mode="constant").astype(np.float32)


def center_of_mass(a):
    tot = a.sum()
    if tot <= 0:
        return (a.shape[0] - 1) / 2.0, (a.shape[1] - 1) / 2.0
    rr, cc = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return (a * rr).sum() / tot, (a * cc).sum() / tot


def paste_center(patch, size=28, center="com", shift=(0, 0), subpixel=False):
    """Dat patch vao canvas size x size sao cho tam (com | bbox) nam o giua (+ shift), co cat bien.

    subpixel=True: can tam voi do chinh xac duoi pixel (noi suy tuyen tinh), tam = (size-1)/2 chinh xac.
    """
    h, w = patch.shape
    if center == "com":
        cr, cc = center_of_mass(patch)
    else:
        cr, cc = (h - 1) / 2.0, (w - 1) / 2.0
    tr = (size - 1) / 2.0 - cr + shift[0]
    tc = (size - 1) / 2.0 - cc + shift[1]
    if subpixel:
        r0, c0 = int(np.floor(tr)), int(np.floor(tc))
        fr, fc = tr - r0, tc - c0
    else:
        r0, c0 = int(round(tr)), int(round(tc))
        fr = fc = 0.0
    out = np.zeros((size, size), dtype=np.float32)
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + h, size), min(c0 + w, size)
    if re > rs and ce > cs:
        out[rs:re, cs:ce] = patch[rs - r0:re - r0, cs - c0:ce - c0]
    if subpixel and (fr > 1e-6 or fc > 1e-6):
        out = ndimage.shift(out, (fr, fc), order=1, mode="constant").astype(np.float32)
    return out


def mnist_normalize(img, box=20, size=28):
    """Chuan hoa kieu MNIST: cat sat net, giu ti le vao khung box x box, can tam theo trong tam.

    img: float 0..1 (nen den). Dung cho anh tu viet / canvas.
    """
    c = crop(img, thr=0.05)
    if c.size == 0 or c.max() <= 0:
        return np.zeros((size, size), dtype=np.float32)
    h, w = c.shape
    s = box / max(h, w)
    c = resize(c, round(h * s), round(w * s))
    out = paste_center(c, size, "com")
    m = out.max()
    return out / m if m > 0 else out


def to_uint8(a):
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
