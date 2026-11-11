"""Did the defence FIND the patch?"""
from __future__ import annotations

from typing import Optional

import torch

NAN = float("nan")

DETECT_COVERAGE = 0.5


def _as_bool_bchw(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """[B,H,W] or [B,1,H,W] or [H,W], any dtype -> bool [B,1,H,W]."""
    if m is None:
        return None
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    if m.dim() != 4:
        raise ValueError(f"expected a mask of 2-4 dims, got {tuple(m.shape)}")
    return m > 0.5 if torch.is_floating_point(m) else m.bool()


def mask_detection(pred: torch.Tensor,
                   true: Optional[torch.Tensor] = None) -> dict:
    """Score a predicted patch mask against the true footprint."""
    p = _as_bool_bchw(pred)
    t = _as_bool_bchw(true)
    B = p.shape[0]
    npx = float(p.shape[-1] * p.shape[-2])
    pa = p.flatten(1).sum(1).double()

    if t is None:
        return {"coverage": NAN, "precision": NAN, "mask_iou": NAN,
                "fp_area_frac": float((pa / npx).mean()),
                "pred_area_frac": float((pa / npx).mean()),
                "true_area_frac": 0.0, "detected": NAN}

    if t.shape[0] == 1 and B > 1:
        t = t.expand(B, -1, -1, -1)
    ta = t.flatten(1).sum(1).double()
    inter = (p & t).flatten(1).sum(1).double()
    union = (p | t).flatten(1).sum(1).double()
    fp = (p & ~t).flatten(1).sum(1).double()

    cov = torch.where(ta > 0, inter / ta.clamp(min=1), torch.full_like(ta, NAN))
    prec = torch.where(pa > 0, inter / pa.clamp(min=1), torch.full_like(pa, NAN))
    iou = torch.where(union > 0, inter / union.clamp(min=1),
                      torch.full_like(union, NAN))
    det = torch.where(ta > 0, (cov >= DETECT_COVERAGE).double(),
                      torch.full_like(ta, NAN))

    def nm(x):
        return float(x.nanmean()) if bool((~x.isnan()).any()) else NAN

    return {"coverage": nm(cov), "precision": nm(prec), "mask_iou": nm(iou),
            "fp_area_frac": float((fp / npx).mean()),
            "pred_area_frac": float((pa / npx).mean()),
            "true_area_frac": float((ta / npx).mean()),
            "detected": nm(det)}



def aggregate_detection(rows) -> dict:
    """nanmean each field over a list of mask_detection() dicts."""
    rows = [r for r in rows if r]
    if not rows:
        return {}
    out = {}
    for k in rows[0]:
        v = torch.tensor([float(r.get(k, NAN)) for r in rows])
        out[k] = float(v.nanmean()) if bool((~v.isnan()).any()) else NAN
        out[f"{k}_n"] = int((~v.isnan()).sum())
    return out


def score_stats(prob, true=None) -> dict:
    """What the detector SCORED, before any threshold."""
    if prob is None:
        return {}
    p = prob.flatten(1).double() if prob.dim() == 4 else prob.flatten().double()
    out = {"prob_max": float(p.max()), "prob_mean": float(p.mean()),
           "prob_q999": float(p.flatten().quantile(0.999))}
    if true is None:
        return out
    t = _as_bool_bchw(true)
    if t.shape[0] == 1 and prob.shape[0] > 1:
        t = t.expand(prob.shape[0], -1, -1, -1)
    pm = prob if prob.dim() == 4 else prob.view(1, 1, *prob.shape[-2:])
    inside, outside = pm[t.expand_as(pm)], pm[~t.expand_as(pm)]
    if inside.numel():
        out.update(in_max=float(inside.max()), in_mean=float(inside.mean()))
    if outside.numel():
        out.update(out_max=float(outside.max()), out_mean=float(outside.mean()))
    if inside.numel() and outside.numel():
        out["in_out_gap"] = out["in_mean"] - out["out_mean"]
    return out
