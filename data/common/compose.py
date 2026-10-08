"""Ghep hai anh chu so 28x28 thanh MOT anh so hai chu so 28x28 (ban toi gian).

Ghep o canvas phan giai cao 'hr' roi thu nho MOT LAN ve 28x28 (xem layout.py):
  1. cat sat net tung chu so, phong to hr lan, lech kich co giua hai chu (size_ratio_std)
  2. nghieng tung chu so doc lap (shear_individual_std, do)
  3. dat canh nhau voi khoang cach 'gap' (don vi: chieu cao chu 20px; am = chong lan) va lech duong co so (baseline_std)
  4. bo cuc ve 28x28 theo quy uoc MNIST

cfg["compose"] = {"hr": 4, "size_ratio_std": 0.06, "shear_individual_std": 3.0, "gap": [-0.05, 0.35], "baseline_std": 0.04}
cfg["layout"]  = {"box": 20, "center": "com"}
"""
import numpy as np

from .features import crop
from .imgops import resize, shear_x, to_uint8
from .layout import apply_layout


def _prep_digit(img, scale, shear_deg):
    """img float 0..1 28x28 -> chu so cat sat, phong to 'scale' lan, nghieng, cat sat lai."""
    c = crop(img, thr=0.1)
    h, w = c.shape
    c = resize(c, round(h * scale), round(w * scale))
    c = shear_x(c, shear_deg)
    return crop(c, thr=0.05)


def compose_number(left, right, rng, cfg):
    """left, right: float32 0..1 (28x28). Tra ve anh float 0..1 (28x28)."""
    cc = cfg["compose"]
    hr = int(cc.get("hr", 4))

    # --- 1 & 2: kich co + do nghieng cua tung chu ---
    size_ratio = np.exp(rng.normal(0, cc.get("size_ratio_std", 0.06)))
    shear_std = cc.get("shear_individual_std", 3.0)
    left_hr = _prep_digit(left, hr, rng.normal(0, shear_std))
    right_hr = _prep_digit(right, hr * size_ratio, rng.normal(0, shear_std))
    hL, wL = left_hr.shape
    hR, wR = right_hr.shape

    # --- 3: dat canh nhau ---
    base = 20 * hr
    gap = rng.uniform(*cc.get("gap", [-0.05, 0.35])) * base
    gap = max(gap, -0.6 * min(wL, wR))
    dy = rng.normal(0, cc.get("baseline_std", 0.04)) * base
    gap_i, dy_i = int(round(gap)), int(round(dy))
    xL, xR = 0, wL + gap_i
    yL_bot, yR_bot = 0, dy_i  # day duoi (bottom aligned), chu phai lech dy
    top = min(yL_bot - hL, yR_bot - hR)
    bot = max(yL_bot, yR_bot)
    margin = 4 * hr
    H = (bot - top) + 2 * margin
    W = max(xL + wL, xR + wR) - min(xL, xR) + 2 * margin
    ox = margin - min(xL, xR)
    oy = margin - top
    canvas = np.zeros((H, W), dtype=np.float32)
    yl, xl = oy + yL_bot - hL, ox + xL
    yr, xr = oy + yR_bot - hR, ox + xR
    canvas[yl:yl + hL, xl:xl + wL] = np.maximum(canvas[yl:yl + hL, xl:xl + wL], left_hr)
    canvas[yr:yr + hR, xr:xr + wR] = np.maximum(canvas[yr:yr + hR, xr:xr + wR], right_hr)

    # --- 4: bo cuc MNIST ---
    return apply_layout(np.clip(canvas, 0, 1), cfg.get("layout", {})), {"gap": gap / base}


def compose_uint8(left_u8, right_u8, rng, cfg):
    img, info = compose_number(left_u8.astype(np.float32) / 255.0,
                               right_u8.astype(np.float32) / 255.0, rng, cfg)
    return to_uint8(img), info
