r"""
Legitimate Adversarial Patches — Tan et al., ACM MM 2021.
https://doi.org/10.1145/3474085.3475653

WHY THIS AND NOT THE BigGAN PATCH
---------------------------------
gan mode constrains the patch to a 120-dim BigGAN latent — a HARD manifold
constraint. In our runs the optimiser found nothing adversarial inside it
(SegFormer: 0.71% flip vs 99.63% for unconstrained raw), across two
architectures and both objectives.

LAP keeps the FULL pixel parameterisation and adds a SOFT pull toward a
reference image:

    L = L_adv + alpha*L_rat + beta*L_tv + gamma*L_nps

alpha is a continuous knob between unconstrained (alpha=0, i.e. raw) and "looks
exactly like the reference". The naturalism/strength tradeoff becomes a
measurable curve rather than a binary.

TWO-STAGE TRAINING (their Fig 4, Sec 4.3)
-----------------------------------------
Stage 1 : init from the reference, alpha=0  -> "transition patch"
Stage 2 : init from the transition patch, alpha>0 -> LAP

They report this beats single-stage: transition-init ASI 0.2938 vs
cartoon-init 0.7301, a large rationality gain at comparable attack strength.

WEIGHT CALIBRATION — READ BEFORE SETTING alpha/beta
---------------------------------------------------
Their alpha=1e-4 / beta=0.2 are tuned to THEIR loss scales: L_obj is a SUM over
YOLO box confidences and the patch is 300x300. Our adversarial losses are
per-pixel MEANS of order 0.03 (cospgd) to 20 (ipatch), while L_rat and L_tv
here are SUMS over a 128x128 patch (order 1e2-1e4). Copying their weights lets
the rationality terms swamp the attack by 3-5 orders of magnitude.

Defaults are therefore 0. Run stage 1, read magnitude_report(), set the weights
from the printed suggestions.

PORTED / DROPPED
----------------
  PORTED  L_rat (Eq 7), L_tv (Eq 8, ISOTROPIC), edge mask (Eq 5 Grad term),
          ASI / AGI / ADE (Eqs 1-3)
  DROPPED L_nps (Eq 9) is implemented but off by default — it only matters for
          physical printing, and this threat model is digital.
          Bg(p) lives in shape.py rather than here.
  ADAPTED Their YOLOv2 objectness L_obj becomes our segmentation loss.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
#  Reference and edge mask
# ═════════════════════════════════════════════════════════════════════════════

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
    """
    Bool [H,W] of the reference's strong edges — the Grad term of Eq 5.

    Freezing the outline stops the optimiser erasing what makes the image
    recognisable. theta=0.1 in their Fig 2.
    """
    g = gradient_magnitude(reference)
    return (g / g.max().clamp(min=1e-8)) > theta


def logit_seed(pixels: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Inverse sigmoid, so sigmoid(param) == pixels at step 0.

    Clamped away from 0 and 1: logit(0) = -inf saturates the sigmoid and kills
    the gradient. Verified round-trip error ~1e-16.
    """
    p = pixels.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


# ═════════════════════════════════════════════════════════════════════════════
#  Losses
# ═════════════════════════════════════════════════════════════════════════════

def rationality_loss(patch, reference, mask: Optional[torch.Tensor] = None):
    """
    L_rat, Eq 7:  sqrt( sum_ij (q_ij - c_ij)^2 )

    Euclidean, NOT squared. The sqrt matters: the gradient scales as 1/||.||,
    so the pull weakens as the patch approaches the reference rather than
    vanishing quadratically.

    mask : optional [H,W] bool — score only these pixels. Use the interior when
           edges are hard-frozen, since frozen pixels only add a constant.
    """
    d2 = (patch - reference) ** 2
    if mask is not None:
        d2 = d2 * mask.unsqueeze(0).float()
    return torch.sqrt(d2.sum() + 1e-12)


def tv_loss(patch, mask: Optional[torch.Tensor] = None):
    r"""
    L_tv, Eq 8:  sum_ij sqrt( (q_ij - q_i+1,j)^2 + (q_ij - q_i,j+1)^2 )

    ISOTROPIC. Differs from the anisotropic mean(|dx|)+mean(|dy|) used
    elsewhere; this one matches the paper, and its rotational symmetry
    preserves diagonal edges better. Note it is a SUM, so its magnitude is
    orders above a mean-based TV.

    mask : [H,W] bool — the ACTIVE set, i.e. the support of q = M(p).

    WHY THE MASK MATTERS MORE HERE THAN ANYWHERE ELSE
    -------------------------------------------------
    A difference term is kept only where the centre AND both of its neighbours
    are active. Include the silhouette boundary and TV penalises the object's
    OUTLINE — the single feature LAP exists to preserve, and the thing Eq 5's
    Grad term separately goes to the trouble of freezing.

    Measured on a synthetic disc-on-background: 95% of the unmasked TV comes
    from the outline alone. Unmasked, beta would spend almost all of its budget
    smoothing away the shape.
    """
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
    """
    L_nps, Eq 9: sum over pixels of min L2 distance to any printable colour.

    mask : [H,W] bool active set. Background pixels are never printed, so
           scoring them adds a constant that inflates the calibration reading.

    OFF BY DEFAULT — irrelevant to a digital threat model, and it only costs
    attack strength. Kept for a future physical-world extension.
    """
    if printable is None:
        printable = _PRINTABLE.to(patch.device)
    px = patch.permute(1, 2, 0).reshape(-1, 3)
    d = torch.cdist(px, printable).min(dim=1).values
    if mask is not None:
        d = d * mask.reshape(-1).float()
    return d.sum()


# ═════════════════════════════════════════════════════════════════════════════
#  Rationality indicators (evaluation only — numpy, not differentiable)
# ═════════════════════════════════════════════════════════════════════════════

def _hwc(patch: torch.Tensor) -> np.ndarray:
    return patch.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def asi(patch: torch.Tensor) -> float:
    """
    ASI, Eq 1 — mean HSL saturation over non-black pixels. LOWER = more
    natural. Their TAP 0.6388, cartoon 0.3704, LAP 0.4449.
    """
    x = _hwc(patch)
    mx, mn = x.max(axis=2), x.min(axis=2)
    L = (mx + mn) / 2.0
    denom = np.where(L < 0.5, mx + mn, 2.0 - mx - mn)
    S = np.where(np.abs(mx - mn) < 1e-8, 0.0,
                 (mx - mn) / np.clip(denom, 1e-8, None))
    valid = mx > 0
    return float(S[valid].sum() / max(valid.sum(), 1))


def agi(patch: torch.Tensor, theta: float = 0.1) -> float:
    """
    AGI, Eq 2 — mean gradient magnitude over pixels above theta. HIGHER = more
    natural: deliberate edges rather than uniform high-frequency noise.
    Their TAP 0.2183, cartoon 0.3378, LAP 0.2850.
    """
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
    """
    ADE, Eq 3 — mean GLCM energy over 4 directions (0, 45, 90, 135 deg), where
    energy_k = sqrt(sum_ij (G^k_ij)^2). HIGHER = more natural: regular, stable
    texture rather than static. Their TAP 0.0499, cartoon 0.4056, LAP 0.4072 —
    this is the metric that separates structured imagery from adversarial noise
    most sharply.

    Verified against extremes: uniform image -> 1.0, uniform noise -> ~1/levels.
    """
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
    r"""
    Magnitudes of EVERY LAP term — including zero-weighted ones — plus the
    weights that would put each at 10% / 50% of the adversarial term.

    `active` is the support of q = M(p): the SAME mask the training loop uses.
    Passing it explicitly (rather than recomputing an "interior" here) is what
    keeps calibration and optimisation consistent. The earlier version scored
    the full rectangle, so with a ~55% silhouette roughly 45% of the reported
    L_rat came from pixels that can never affect the attack — and the suggested
    alpha was correspondingly too small.

    Deliberately RECOMPUTES rather than reading the training-loop values, which
    skip zero-weighted terms for speed. Without that, a stage-1 run
    (alpha=beta=gamma=0) reports all zeros and gives you nothing to calibrate
    against — which is the entire point of stage 1.
    """
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
