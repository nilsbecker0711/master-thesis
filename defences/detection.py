r"""
Did the defence FIND the patch? Predicted mask against the true footprint.

WHY THIS IS NOT REDUNDANT WITH mIoU RECOVERY. A defence can recover mIoU while
localising nothing — it blanked enough of the frame that the patch went with it
— and it can localise perfectly while recovering nothing. Those are different
failures with different fixes, and a recovery number alone cannot tell them
apart. Worse, the one that looks best on recovery is often the one that
destroys the most clean image.

WE HAVE THE GROUND TRUTH AND THE PAPERS DID NOT. Patch.apply() returns the exact
footprint it composited, so the predicted mask has a pixel-accurate label beside
it at no cost. SAC and Jedi were evaluated on detection and classification
benchmarks where that label had to be annotated by hand (SAC published the
APRICOT-Mask dataset precisely to get it). Here it is free, which makes
localisation a measurable axis rather than a claim.

THE ASYMMETRY THAT MATTERS. For removing a patch, COVERAGE is the quantity with
teeth: a mask that covers 100% of the footprint removes the patch whatever else
it does, and SAC's shape completion is designed to guarantee exactly that. IoU
penalises over-covering, which is the right instinct for a segmenter and the
wrong one here — a defence is allowed to be generous, it just has to pay for it
on the clean image. So report coverage and the clean-image false-positive area
TOGETHER, and never quote IoU alone.

CLEAN IMAGES HAVE NO FOOTPRINT. coverage and IoU are then undefined and come
back NaN by construction; `fp_area_frac` is still defined and is the number
that matters there — it is the fraction of a patch-free frame the defence
erased. nanmean over images, as everywhere else in this package.
"""
from __future__ import annotations

from typing import Optional

import torch

NAN = float("nan")

#: coverage at or above which the patch counts as located.
#: Any threshold is arbitrary, so it is a module constant rather than an
#: argument — one definition for every table. 0.5 is the loosest defensible
#: reading of "found it": half the patch still showing is not removal, but it
#: is localisation.
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
    r"""
    Score a predicted patch mask against the true footprint. One image or a
    batch; a batch is reduced by mean over images (nanmean for the ratios).

    pred   the defence's final mask. 1 = "this is patch".
    true   the footprint from Patch.apply(), or None for a clean image.

    Returns, all floats:

      coverage        |pred & true| / |true|    1.0 = the patch is fully removed
      precision       |pred & true| / |pred|    how much of the hole was useful
      mask_iou        |pred & true| / |pred | true|
      fp_area_frac    |pred & ~true| / (H*W)    frame erased for no reason
      pred_area_frac  |pred| / (H*W)
      true_area_frac  |true| / (H*W)
      detected        coverage >= DETECT_COVERAGE, as a 0/1 rate over the batch

    `detected` thresholds coverage at DETECT_COVERAGE.
    """
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
    r"""
    nanmean each field over a list of mask_detection() dicts.

    Separate from SegMetric because detection does NOT pool: a confusion matrix
    over all images would let one big false positive dominate, and the question
    "on what fraction of images was the patch located" is per image by
    definition. This is the NmIoU-style aggregation, deliberately.
    """
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
    r"""
    What the detector SCORED, before any threshold. The diagnostic for an empty
    mask.

    A zero mask has two causes with opposite conclusions:

      in_max near 0      the detector is out of domain on this input. No
                         threshold, no shape prior and no operating point
                         rescues it; only retraining or matching the scale it
                         was trained at.
      in_max near 0.5    it is in domain and losing to the cut. A threshold
                         sweep or an --sac_image_scale change moves it.

    `in_` is inside the true footprint, `out_` outside it. Reporting both is the
    point: a detector that scores 0.4 inside and 0.4 everywhere else has not
    found anything, it is just uniformly uncertain, and only the CONTRAST
    between the two says whether there is signal to threshold at all.
    """
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
        # The only number that says whether a threshold could ever work here.
        out["in_out_gap"] = out["in_mean"] - out["out_mean"]
    return out
