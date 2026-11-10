"""Legitimate Adversarial Patches — Tan et al., ACM MM 2021. https://doi.org/10.1145/3474085.3475653"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of [3,H,W] or [1,3,H,W], returned [H,W]."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    C = x.shape[1]
    gx = F.conv2d(x, kx.repeat(C, 1, 1, 1), padding=1, groups=C)
    gy = F.conv2d(x, kx.transpose(2, 3).repeat(C, 1, 1, 1), padding=1, groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).mean(1).squeeze(0)


def edge_mask(reference: torch.Tensor, theta: float = 0.1) -> torch.Tensor:
    """Bool [H,W] of the reference's strong edges — the Grad term of Eq 5."""
    g = gradient_magnitude(reference)
    return (g / g.max().clamp(min=1e-8)) > theta


def logit_seed(pixels: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Inverse sigmoid, so sigmoid(param) == pixels at step 0."""
    p = pixels.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def rationality_loss(patch, reference, mask: Optional[torch.Tensor] = None):
    """L_rat, Eq 7: sqrt( sum_ij (q_ij - c_ij)^2 )"""
    d2 = (patch - reference) ** 2
    if mask is not None:
        d2 = d2 * mask.unsqueeze(0).float()
    return torch.sqrt(d2.sum() + 1e-12)


def tv_loss(patch, mask: Optional[torch.Tensor] = None):
    """L_tv, Eq 8: sum_ij sqrt( (q_ij - q_i+1,j)^2 + (q_ij - q_i,j+1)^2 )"""
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)
    dh = patch[:, :, 1:, :-1] - patch[:, :, :-1, :-1]
    dw = patch[:, :, :-1, 1:] - patch[:, :, :-1, :-1]
    per_pixel = torch.sqrt(dh ** 2 + dw ** 2 + 1e-12)
    if mask is not None:
        valid = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:]
        per_pixel = per_pixel * valid.unsqueeze(0).unsqueeze(0).float()
    return per_pixel.sum()


_PRINTABLE = torch.tensor([
    [0.10, 0.10, 0.10], [0.90, 0.90, 0.90], [0.50, 0.50, 0.50],
    [0.75, 0.15, 0.15], [0.15, 0.55, 0.20], [0.15, 0.20, 0.65],
    [0.85, 0.75, 0.15], [0.80, 0.45, 0.15], [0.55, 0.20, 0.55],
    [0.20, 0.65, 0.70], [0.95, 0.55, 0.60], [0.35, 0.30, 0.20],
    [0.65, 0.80, 0.40], [0.25, 0.35, 0.45], [0.90, 0.85, 0.70],
], dtype=torch.float32)


def nps_loss(patch, printable: Optional[torch.Tensor] = None,
             mask: Optional[torch.Tensor] = None):
    """L_nps, Eq 9: sum over pixels of min L2 distance to any printable colour."""
    if printable is None:
        printable = _PRINTABLE.to(patch.device)
    px = patch.permute(1, 2, 0).reshape(-1, 3)
    d = torch.cdist(px, printable).min(dim=1).values
    if mask is not None:
        d = d * mask.reshape(-1).float()
    return d.sum()


def _hwc(patch: torch.Tensor) -> np.ndarray:
    return patch.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def asi(patch: torch.Tensor) -> float:
    """ASI, Eq 1 — mean HSL saturation over non-black pixels."""
    x = _hwc(patch)
    mx, mn = x.max(axis=2), x.min(axis=2)
    L = (mx + mn) / 2.0
    denom = np.where(L < 0.5, mx + mn, 2.0 - mx - mn)
    S = np.where(np.abs(mx - mn) < 1e-8, 0.0,
                 (mx - mn) / np.clip(denom, 1e-8, None))
    valid = mx > 0
    return float(S[valid].sum() / max(valid.sum(), 1))


def agi(patch: torch.Tensor, theta: float = 0.1) -> float:
    """AGI, Eq 2 — mean gradient magnitude over pixels above theta."""
    g = gradient_magnitude(patch)
    g = (g / g.max().clamp(min=1e-8)).detach().cpu().numpy()
    sel = g > theta
    return float(g[sel].mean()) if sel.any() else 0.0


def _glcm(gq: np.ndarray, dy: int, dx: int, levels: int) -> np.ndarray:
    H, W = gq.shape
    y0, y1 = max(0, -dy), min(H, H - dy)
    x0, x1 = max(0, -dx), min(W, W - dx)
    a = gq[y0:y1, x0:x1].ravel()
    b = gq[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
    G = np.bincount(a * levels + b, minlength=levels * levels)
    G = G.reshape(levels, levels).astype(np.float64)
    s = G.sum()
    return G / s if s > 0 else G


def ade(patch: torch.Tensor, levels: int = 8, distance: int = 1) -> float:
    """ADE, Eq 3 — mean GLCM energy over 4 directions (0, 45, 90, 135 deg), where energy_k = sqrt(sum_ij (G^k_ij)^2)."""
    x = _hwc(patch)
    gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    gq = np.clip((gray * levels).astype(np.int64), 0, levels - 1)
    offs = [(0, distance), (-distance, distance),
            (-distance, 0), (-distance, -distance)]
    return float(np.mean([np.sqrt((_glcm(gq, dy, dx, levels) ** 2).sum())
                          for dy, dx in offs]))


def rationality_report(patch, reference=None, theta: float = 0.1, log=print):
    """All three indicators, optionally against the reference."""
    out = {"ASI": asi(patch), "AGI": agi(patch, theta), "ADE": ade(patch)}
    log("\n  Rationality indicators (Tan et al. 2021):")
    log(f"    ASI (saturation,  LOWER = natural): {out['ASI']:.4f}")
    log(f"    AGI (edge clarity, HIGHER = natural): {out['AGI']:.4f}")
    log(f"    ADE (texture,      HIGHER = natural): {out['ADE']:.4f}")
    if reference is not None:
        ref = {"ASI": asi(reference), "AGI": agi(reference, theta),
               "ADE": ade(reference)}
        out["reference"] = ref
        out["L2_to_reference"] = float(
            torch.sqrt(((patch - reference) ** 2).sum()).item())
        log(f"    -- reference: ASI {ref['ASI']:.4f}  AGI {ref['AGI']:.4f}  "
            f"ADE {ref['ADE']:.4f}")
        log(f"    L2 to reference: {out['L2_to_reference']:.2f}")
    return out


def magnitude_report(adv_value, patch, reference, active, cfg, log=print):
    """Magnitudes of EVERY LAP term — including zero-weighted ones — plus the weights that would put each at 10% / 50% of the adversarial term."""
    with torch.no_grad():
        raw = {"rat": float(rationality_loss(patch, reference, mask=active)),
               "tv": float(tv_loss(patch, mask=active)),
               "nps": float(nps_loss(patch, mask=active))}
    adv = abs(float(adv_value))

    log("\n[lap] term magnitudes — use these to set stage-2 weights")
    log(f"      |L_adv| = {adv:.4e}")
    if active is not None:
        log(f"      scored over the ACTIVE set: {int(active.sum()):,} / "
            f"{active.numel():,} px ({100*active.float().mean():.1f}%)")
    log(f"      {'term':<5s}{'raw':>13s}{'weight':>11s}{'weighted':>13s}"
        f"   suggested w (10% / 50%)")
    for k, w in (("rat", cfg.lap_alpha), ("tv", cfg.lap_beta),
                 ("nps", cfg.lap_gamma)):
        r = raw[k]
        sug = (f"{0.10*adv/r:.2e} / {0.50*adv/r:.2e}"
               if r > 0 and adv > 0 else "n/a")
        log(f"      {k:<5s}{r:13.4e}{w:11g}{r*w:13.4e}   {sug}")

    tot = cfg.lap_alpha * raw["rat"] + cfg.lap_beta * raw["tv"] \
        + cfg.lap_gamma * raw["nps"]
    log(f"      TOTAL weighted extra = {tot:.4e}")
    if tot > adv:
        log("      WARNING: rationality terms EXCEED |L_adv|. The patch will "
            "collapse toward the reference and the attack will fail.")
    elif tot == 0.0:
        log("      (stage 1: no rationality constraint — equivalent to raw "
            "but seeded from the reference image)")
    return raw
