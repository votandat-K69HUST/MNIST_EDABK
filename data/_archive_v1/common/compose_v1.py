"""Ghep hai anh chu so 28x28 thanh MOT anh so hai chu so 28x28.

Quy trinh (tat ca o canvas phan giai cao 'hr' roi moi thu nho MOT LAN ve 28x28, de vung noi bi mo
tu nhien giong anh that bi resize, thay vi moi chu bi resize rieng roi dan lai):

  1. cat sat net tung chu so, phong to hr lan, thay doi kich co (chung + lech nho giua 2 chu)
  2. nghieng (shear): do nghieng chung cho ca so + lech rieng tung chu
  3. dat canh nhau voi khoang cach (am = chong lan / cham nhau), lech duong co so (baseline)
  4. thay doi do day net (dilate / erode) tren ca canvas
  5. (tuy chon) ve net noi (ligature) giua hai chu so
  6. bo cuc ve 28x28 (layout.py) + hau xu ly (gamma, lam mo, nhieu)

Tat ca tham so doc tu cfg["compose"], xem config.json trong tung thu muc phuong phap.
"""
import numpy as np
from scipy import ndimage

from . import ligature as lig
from .features import crop, thickness
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
    """left, right: float32 0..1 (28x28). cfg: cfg['compose'] (+ 'layouts', 'post'). Tra ve (float 0..1, info)."""
    cc = cfg["compose"]
    hr = int(cc.get("hr", 4))

    # --- 1 & 2: kich co + do nghieng ---
    size_common = np.exp(rng.normal(0, cc.get("size_common_std", 0.05)))
    size_ratio = np.exp(rng.normal(0, cc.get("size_ratio_std", 0.06)))
    shear_common = rng.normal(0, cc.get("shear_common_std", 6.0))
    shear_ind = cc.get("shear_individual_std", 3.0)
    left_hr = _prep_digit(left, hr * size_common, shear_common + rng.normal(0, shear_ind))
    right_hr = _prep_digit(right, hr * size_common * size_ratio, shear_common + rng.normal(0, shear_ind))
    hL, wL = left_hr.shape
    hR, wR = right_hr.shape

    # --- 3: dat canh nhau ---
    base = 20 * hr
    gap = rng.uniform(*cc.get("gap", [-0.05, 0.35])) * base
    gap = max(gap, -0.6 * min(wL, wR))
    dy = rng.normal(0, cc.get("baseline_std", 0.04)) * base
    gap_i, dy_i = int(round(gap)), int(round(dy))
    xL, xR = 0, wL + gap_i
    yL_bot, yR_bot = 0, dy_i  # toa do day duoi (bottom aligned) tuong doi
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

    # --- 4: do day net ---
    tcfg = cc.get("thickness_hr", {"values": [0], "probs": [1.0]})
    t = int(rng.choice(tcfg["values"], p=np.array(tcfg["probs"]) / np.sum(tcfg["probs"])))
    if t > 0:
        canvas = ndimage.grey_dilation(canvas, size=(2 * t + 1, 2 * t + 1))
    elif t < 0:
        canvas = ndimage.grey_erosion(canvas, size=(2 * -t + 1, 2 * -t + 1))
    canvas = np.clip(canvas, 0, 1)

    # --- 5: net noi ---
    lig_mode = 0
    lcfg = cc.get("ligature", {})
    if lcfg and rng.random() < lcfg.get("prob", 0.0):
        p0 = lig.exit_point(left_hr, rng, lcfg.get("exit_lower_frac", 0.45))
        p1 = lig.entry_point(right_hr, rng, lcfg.get("entry_upper_frac", 0.5))
        if p0 is not None and p1 is not None:
            p0 = (p0[0] + yl, p0[1] + xl)
            p1 = (p1[0] + yr, p1[1] + xr)
            width = max(thickness(left_hr, 0.5), 1.5 * hr / 2)
            dist = p1[1] - p0[1]
            if rng.random() < lcfg.get("tail_prob", 0.25):
                # duoi but: net ngan huong sang phai, khong cham chu sau
                p1 = (p0[0] - rng.uniform(0, 0.3) * hL, p0[1] + max(dist, 3 * hr) * rng.uniform(0.3, 0.8))
                if lig.draw_ligature(canvas, p0, p1, width, rng, lcfg):
                    lig_mode = 2
            elif dist > 2 * hr:
                if lig.draw_ligature(canvas, p0, p1, width, rng, lcfg):
                    lig_mode = 1

    # --- 6: bo cuc + hau xu ly ---
    img, layout_name = apply_layout(canvas, cfg["layouts"], rng)
    img = postprocess(img, cfg.get("post", {}), rng)
    info = {"gap": gap / base, "thick": t, "lig": lig_mode, "layout": layout_name}
    return img, info


def postprocess(img, pcfg, rng):
    """Hau xu ly tren anh 28x28 float 0..1: gamma, lam mo, nhieu. Renormalize max = 1."""
    if pcfg.get("gamma"):
        img = img ** rng.uniform(*pcfg["gamma"])
    if pcfg.get("blur_sigma"):
        s = rng.uniform(*pcfg["blur_sigma"])
        if s > 0.05:
            img = ndimage.gaussian_filter(img, s)
    if pcfg.get("contrast"):  # lam net sac hon: keo gian muc xam quanh 0.5
        k = rng.uniform(*pcfg["contrast"])
        img = np.clip((img - 0.5) * k + 0.5, 0.0, 1.0)
    m = img.max()
    if m > 0:
        img = img / m
    ns = pcfg.get("noise_std", 0.0)
    if ns > 0:
        img = np.clip(img + rng.normal(0, ns, img.shape) * (rng.random(img.shape) < 0.3), 0, 1)
    return img.astype(np.float32)


def compose_uint8(left_u8, right_u8, rng, cfg):
    img, info = compose_number(left_u8.astype(np.float32) / 255.0,
                               right_u8.astype(np.float32) / 255.0, rng, cfg)
    return to_uint8(img), info
