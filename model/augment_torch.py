"""Augmentation online tren batch (torch, chay nhanh tren CPU): affine, day/manh net, gamma, nhieu.

Vi dung MLP (khong bat bien dich chuyen nhu CNN) nen augmentation hinh hoc la rat quan trong.
Dau vao/ra: tensor (B,28,28) hoac (B,1,28,28) float 0..1, nen den.
"""
import math

import torch
import torch.nn.functional as F


def _u(lo, hi, n, g):
    return lo + (hi - lo) * torch.rand(n, generator=g)


def augment(x, cfg, g):
    """x: (B,28,28) float 0..1. cfg: dict (xem config.json -> 'augment'). g: torch.Generator."""
    if not cfg or not cfg.get("enabled", True):
        return x
    B = x.shape[0]
    x = x.unsqueeze(1)

    # --- affine ---
    rot = math.radians(cfg.get("rotate", 8))
    ang = _u(-rot, rot, B, g)
    s_lo, s_hi = cfg.get("scale", [0.9, 1.1])
    sx = _u(s_lo, s_hi, B, g)
    sy = sx * _u(*cfg.get("aspect_jitter", [0.93, 1.07]), B, g)
    sh = torch.tan(torch.deg2rad(_u(-cfg.get("shear", 10), cfg.get("shear", 10), B, g)))
    shift = cfg.get("shift", 2) / 14.0
    tx, ty = _u(-shift, shift, B, g), _u(-shift, shift, B, g)
    cos, sin = torch.cos(ang), torch.sin(ang)
    # theta = R @ [[1/sx, sh],[0, 1/sy]]
    a11, a12 = cos / sx, cos * sh - sin / sy
    a21, a22 = sin / sx, sin * sh + cos / sy
    theta = torch.stack([torch.stack([a11, a12, tx], 1), torch.stack([a21, a22, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # --- do day net: tron anh voi ban dilate / erode ---
    p = cfg.get("thickness_p", 0.3)
    if p > 0:
        lam = _u(*cfg.get("thickness_strength", [0.3, 0.8]), B, g).view(B, 1, 1, 1)
        use = (torch.rand(B, generator=g) < p).view(B, 1, 1, 1).float()
        dil = F.max_pool2d(x, 3, 1, 1)
        ero = -F.max_pool2d(-x, 3, 1, 1)
        alt = torch.where(torch.rand(B, 1, 1, 1, generator=g) < 0.5, dil, ero)
        x = x + use * lam * (alt - x)

    # --- gamma ---
    gp = cfg.get("gamma_p", 0.3)
    if gp > 0:
        gam = _u(*cfg.get("gamma", [0.8, 1.25]), B, g).view(B, 1, 1, 1)
        use = (torch.rand(B, generator=g) < gp).view(B, 1, 1, 1)
        x = torch.where(use, x.clamp(0, 1) ** gam, x)

    # --- nhieu thua ---
    ns = cfg.get("noise_std", 0.03)
    if ns > 0:
        noise = torch.randn(x.shape, generator=g) * ns * (torch.rand(x.shape, generator=g) < 0.3)
        x = x + noise

    x = x.clamp(0, 1)
    m = x.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x / m).squeeze(1)
