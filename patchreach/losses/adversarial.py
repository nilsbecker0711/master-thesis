"""Adversarial objectives."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _reduce(per_pixel: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked mean with a guarded denominator."""
    n = valid.sum().clamp(min=1)
    return (per_pixel * valid.float()).sum() / n


def ce_loss(logits, labels, support: Optional[torch.Tensor] = None):
    """Untargeted CE."""
    valid = labels != 255
    if support is not None:
        valid = valid & support
    per_pixel = F.cross_entropy(logits, labels, reduction="none",
                                ignore_index=255)
    return -_reduce(per_pixel, valid)


def cospgd_loss(logits, labels, support: Optional[torch.Tensor] = None):
    """CosPGD (Agnihotri et al., ICML 2024)"""
    C = logits.shape[1]
    valid = labels != 255
    if support is not None:
        valid = valid & support

    safe = labels.clone()
    safe[labels == 255] = 0
    ref = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()

    cos_w = F.cosine_similarity(F.softmax(logits, 1), ref, dim=1).detach()
    ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    return -_reduce(cos_w * ce, valid)


def ce_target_loss(logits, target_class: int,
                   footprint: Optional[torch.Tensor] = None,
                   support: Optional[torch.Tensor] = None):
    """PLAIN TARGETED CE, per pixel"""
    B, C, H, W = logits.shape
    target = torch.full((B, H, W), target_class, dtype=torch.long,
                        device=logits.device)
    if footprint is not None:
        target[footprint] = 255

    valid = target != 255
    if support is not None:
        valid = valid & support
    if valid.sum() == 0:
        raise ValueError("ce_target loss has no valid pixels — check the "
                         "footprint and reach masks do not exclude everything")

    per_pixel = F.cross_entropy(logits, target, reduction="none",
                                ignore_index=255)
    return _reduce(per_pixel, valid)


def ipatch_cospgd_loss(logits, target_class: int,
                       footprint: Optional[torch.Tensor] = None,
                       support: Optional[torch.Tensor] = None):
    """TARGETED COSPGD — Agnihotri et al., ICML 2024, Eq. 8"""
    B, C, H, W = logits.shape
    target = torch.full((B, H, W), target_class, dtype=torch.long,
                        device=logits.device)
    if footprint is not None:
        target[footprint] = 255

    valid = target != 255
    if support is not None:
        valid = valid & support
    if valid.sum() == 0:
        raise ValueError("ipatch loss has no valid pixels — check the "
                         "footprint and reach masks do not exclude everything")

    safe = target.clone()
    safe[~valid] = 0
    ref = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()

    weight = 1.0 - F.cosine_similarity(F.softmax(logits, 1), ref, dim=1).detach()
    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=255)
    return _reduce(weight * ce, valid)


def margin_loss(logits, ref, support: Optional[torch.Tensor] = None,
                kappa: float = 5.0):
    """CW margin against the CLEAN PREDICTION"""
    if logits.shape[1] < 2:
        raise ValueError("margin loss needs at least two classes")
    valid = ref != 255
    if support is not None:
        valid = valid & support

    safe = ref.clone()
    safe[~valid] = 0
    idx = safe.unsqueeze(1)
    z_ref = logits.gather(1, idx).squeeze(1)
    z_other = logits.scatter(1, idx, float("-inf")).amax(1)
    return _reduce(F.relu(z_ref - z_other + kappa), valid)


def cos_margin_loss(logits, ref, support: Optional[torch.Tensor] = None,
                    kappa: float = 5.0):
    """CosPGD's weight on the CW hinge, both against the CLEAN PREDICTION"""
    if logits.shape[1] < 2:
        raise ValueError("cos_margin loss needs at least two classes")
    C = logits.shape[1]
    valid = ref != 255
    if support is not None:
        valid = valid & support

    safe = ref.clone()
    safe[~valid] = 0
    idx = safe.unsqueeze(1)
    z_ref = logits.gather(1, idx).squeeze(1)
    z_other = logits.scatter(1, idx, float("-inf")).amax(1)

    onehot = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()
    w = F.cosine_similarity(F.softmax(logits, 1), onehot, dim=1).detach()
    return _reduce(w * F.relu(z_ref - z_other + kappa), valid)


def _margin_unbound(*_):
    raise ValueError(
        "loss_fn='margin'/'cos_margin' needs the clean prediction bound at "
        "build time — build(loss_fn, margin_ref=clean_argmax). It is wired "
        "through "
        "patch/optimise.attack_image only; placement=gradcam with "
        "--cam_objective attack cannot use it.")


def build(loss_fn: str, target_class: int = 8,
          tsallis_q: float = 0.0, tsallis_schedule: str = "const",
          tsallis_q_start: float = -2.0, tsallis_q_end: float = 1.0,
          tsallis_total_steps: int = 1,
          margin_ref: Optional[torch.Tensor] = None,
          margin_kappa: float = 5.0):
    """Returns f(logits, labels, footprint, support) -> scalar."""
    if loss_fn in ("margin", "cos_margin"):
        if margin_ref is None:
            return _margin_unbound
        ref = margin_ref
        fn = margin_loss if loss_fn == "margin" else cos_margin_loss
        return lambda lg, lb, fp, sp: fn(lg, ref, sp, margin_kappa)
    if loss_fn == "tsallis":
        from .tsallis import TsallisCELoss
        return TsallisCELoss(q=tsallis_q, schedule=tsallis_schedule,
                             q_start=tsallis_q_start, q_end=tsallis_q_end,
                             total_steps=tsallis_total_steps)
    if loss_fn == "ce_target":
        return lambda lg, lb, fp, sp: ce_target_loss(lg, target_class, fp, sp)
    if loss_fn == "ipatch_cospgd":
        return lambda lg, lb, fp, sp: ipatch_cospgd_loss(lg, target_class,
                                                         fp, sp)
    if loss_fn == "cospgd":
        return lambda lg, lb, fp, sp: cospgd_loss(lg, lb, sp)
    if loss_fn == "ce":
        return lambda lg, lb, fp, sp: ce_loss(lg, lb, sp)
    raise ValueError(f"unknown loss_fn {loss_fn!r}")
