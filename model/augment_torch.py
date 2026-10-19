"""Augmentation online tren batch (torch, chay nhanh tren CPU): affine, day/manh net, gamma, nhieu.

Vi dung MLP (khong bat bien dich chuyen nhu CNN) nen augmentation hinh hoc la rat quan trong.
Dau vao/ra: tensor (B,28,28) hoac (B,1,28,28) float 0..1, nen den.
"""
import math

import torch
import torch.nn.functional as F


def _u(lo, hi, n, g, device=None, dtype=torch.float32):
    return lo + (hi - lo) * torch.rand(n, generator=g, device=device, dtype=dtype)


def augment(x, cfg, g):
    """x: (B,28,28) float 0..1. cfg: dict (xem config.json -> 'augment'). g: torch.Generator."""
    if not cfg or not cfg.get("enabled", True):
        return x
    device = x.device
    dtype = x.dtype
    gen = g
    if gen is not None and getattr(gen, "device", None) != device:
        gen = None

    B = x.shape[0]
    x = x.unsqueeze(1)

    # --- affine ---
    rot = math.radians(cfg.get("rotate", 8))
    ang = _u(-rot, rot, B, gen, device=device, dtype=dtype)
    s_lo, s_hi = cfg.get("scale", [0.9, 1.1])
    sx = _u(s_lo, s_hi, B, gen, device=device, dtype=dtype)
    sy = sx * _u(*cfg.get("aspect_jitter", [0.93, 1.07]), B, gen, device=device, dtype=dtype)
    sh = torch.tan(torch.deg2rad(_u(-cfg.get("shear", 10), cfg.get("shear", 10), B, gen, device=device, dtype=dtype)))
    shift = cfg.get("shift", 2) / 14.0
    tx, ty = _u(-shift, shift, B, gen, device=device, dtype=dtype), _u(-shift, shift, B, gen, device=device, dtype=dtype)
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
        lam = _u(*cfg.get("thickness_strength", [0.3, 0.8]), B, gen, device=device, dtype=dtype).view(B, 1, 1, 1)
        use = (torch.rand(B, generator=gen, device=device) < p).view(B, 1, 1, 1).float()
        dil = F.max_pool2d(x, 3, 1, 1)
        ero = -F.max_pool2d(-x, 3, 1, 1)
        alt = torch.where(torch.rand(B, 1, 1, 1, generator=gen, device=device) < 0.5, dil, ero)
        x = x + use * lam * (alt - x)

    # --- gamma ---
    gp = cfg.get("gamma_p", 0.3)
    if gp > 0:
        gam = _u(*cfg.get("gamma", [0.8, 1.25]), B, gen, device=device, dtype=dtype).view(B, 1, 1, 1)
        use = (torch.rand(B, generator=gen, device=device) < gp).view(B, 1, 1, 1)
        x = torch.where(use, x.clamp(0, 1) ** gam, x)

    # --- nhieu thua ---
    ns = cfg.get("noise_std", 0.03)
    if ns > 0:
        noise = torch.randn(x.shape, generator=gen, device=device, dtype=dtype) * ns * (torch.rand(x.shape, generator=gen, device=device) < 0.3)
        x = x + noise

    x = x.clamp(0, 1)
    m = x.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x / m).squeeze(1)
