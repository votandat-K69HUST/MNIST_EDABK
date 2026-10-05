"""Mo phong net noi (ligature) giua hai chu so khi nguoi viet keo net lien tu chu nay sang chu kia.

Net noi: duong cong Bezier bac 2 tu "diem ra" cua chu dau (thuong o phan duoi ben phai, vi du chan so 1)
den "diem vao" cua chu sau (thuong o phan tren ben trai). Net manh hon o hai dau va MO / MONG o giua
(but nhac nhanh khi keo net), dung nhu quan sat khi viet that.

Moi toa do trong module nay la pixel cua canvas phan giai cao (hr).
"""
import numpy as np


def _ink_points(a, thr=0.3):
    rr, cc = np.nonzero(a > thr)
    return rr, cc


def exit_point(left_hr, rng, lower_frac=0.45):
    """Diem ra: pixel muc cao nhat ben phai trong phan duoi cua chu dau (toa do cuc bo cua left_hr)."""
    h = left_hr.shape[0]
    rr, cc = _ink_points(left_hr)
    if len(rr) == 0:
        return None
    sel = rr >= h * (1 - lower_frac)
    if not sel.any():
        sel = np.ones_like(rr, dtype=bool)
    rr, cc = rr[sel], cc[sel]
    # lay nhom pixel co c lon nhat (top 10%) roi chon ngau nhien
    thr_c = np.quantile(cc, 0.9)
    pick = np.flatnonzero(cc >= thr_c)
    k = rng.choice(pick)
    return float(rr[k]), float(cc[k])


def entry_point(right_hr, rng, upper_frac=0.5):
    """Diem vao: pixel ben trai nhat trong phan tren cua chu sau."""
    h = right_hr.shape[0]
    rr, cc = _ink_points(right_hr)
    if len(rr) == 0:
        return None
    sel = rr <= h * upper_frac
    if not sel.any():
        sel = np.ones_like(rr, dtype=bool)
    rr, cc = rr[sel], cc[sel]
    thr_c = np.quantile(cc, 0.1)
    pick = np.flatnonzero(cc <= thr_c)
    k = rng.choice(pick)
    return float(rr[k]), float(cc[k])


def _stamp_max(canvas, r, c, radius, alpha):
    """Dong dau dia tron (mem) vao canvas bang max."""
    R = int(np.ceil(radius)) + 1
    r0, c0 = int(round(r)), int(round(c))
    ra, rb = max(r0 - R, 0), min(r0 + R + 1, canvas.shape[0])
    ca, cb = max(c0 - R, 0), min(c0 + R + 1, canvas.shape[1])
    if rb <= ra or cb <= ca:
        return
    yy, xx = np.mgrid[ra:rb, ca:cb]
    d = np.sqrt((yy - r) ** 2 + (xx - c) ** 2)
    disk = np.clip(radius + 0.5 - d, 0.0, 1.0) * alpha
    canvas[ra:rb, ca:cb] = np.maximum(canvas[ra:rb, ca:cb], disk)


def draw_ligature(canvas, p0, p1, width_hr, rng, cfg):
    """Ve net noi tu p0 den p1 (r, c) len canvas (float 0..1, sua truc tiep).

    cfg: {"curvature": [lo, hi]  # do cong so voi do dai doan noi
          "mid_alpha": [lo, hi]  # do dam o giua net (1 = nhu net chinh, nho = mo)
          "width_scale": [lo, hi]  # do day so voi net chu so
          "fade_power": 2.0}
    """
    (r0, c0), (r1, c1) = p0, p1
    length = float(np.hypot(r1 - r0, c1 - c0))
    if length < 1.0:
        return False
    # diem dieu khien: giua doan noi lech vuong goc
    mid = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
    d = np.array([r1 - r0, c1 - c0]) / length
    normal = np.array([-d[1], d[0]])
    curv = rng.uniform(*cfg.get("curvature", [-0.3, 0.3])) * length
    ctrl = mid + normal * curv
    wscale = rng.uniform(*cfg.get("width_scale", [0.5, 0.9]))
    radius = max(width_hr * wscale / 2.0, 0.8)
    n = max(int(length / (0.6 * radius)), 8)  # buoc dong dau ~ 0.6 ban kinh
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * np.array([r0, c0]) + 2 * (1 - t) * t * ctrl + t ** 2 * np.array([r1, c1])
    mid_alpha = rng.uniform(*cfg.get("mid_alpha", [0.35, 0.9]))
    power = cfg.get("fade_power", 2.0)
    for (r, c), tt in zip(pts, t[:, 0]):
        edge = abs(2 * tt - 1) ** power  # 1 o hai dau, 0 o giua
        alpha = mid_alpha + (1 - mid_alpha) * edge
        _stamp_max(canvas, r, c, radius, alpha)
    return True
