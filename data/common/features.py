"""Cat sat net cua anh chu so (bounding box)."""
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
