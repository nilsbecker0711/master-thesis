"""SEMANTIC diagnostics — within the reachable zone, WHICH pixels can flip."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.cityscapes import class_name, upsample_to
from ..losses.reach import centroid, distance_map
from ..metrics.miou import SegMetric

RINGS = [(0, 150), (150, 300), (300, 450), (450, 650), (650, 900), (900, 1200)]


@torch.no_grad()
def per_class_iou(logits, labels, K: int, log=print):
    """Clean per-class IoU."""
    m = SegMetric(K, device=logits.device)
    m.update(upsample_to(logits, labels.shape[-2:]).argmax(1), labels)
    iou = m.per_class()
    present = sorted(labels[labels != 255].unique().tolist())
    log("\n[semantic] clean per-class IoU (classes present in GT):")
    out = {}
    for c in present:
        if c < K:
            out[class_name(c)] = iou[c].item()
            log(f"    {c:2d} {class_name(c):10s}: {iou[c]:6.2f}")
    return out


@torch.no_grad()
def class_margin(logits, target_class: int, footprint, K: int, log=print):
    """Mean logit margin to target_class per distance ring"""
    H, W = logits.shape[-2:]
    lg = logits[0, :K]
    tgt = lg[target_class]
    other = lg.clone()
    other[target_class] = -1e9
    margin = (tgt - other.max(0).values)

    dist = distance_map(H, W, *centroid(footprint), logits.device)
    outside = ~footprint[0]

    log(f"\n[semantic] margin to class {target_class} "
        f"({class_name(target_class)}) by ring (higher = easier to flip):")
    out = []
    for lo, hi in RINGS:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() == 0:
            continue
        v = margin[ring].mean().item()
        out.append((lo, hi, v))
        log(f"    {lo:5d}-{hi:<5d}px : {v:+7.2f}")
    return out


@torch.no_grad()
def contestability(logits, footprint, K: int, min_dist: float = 300.0,
                   tol: float = 3.0, log=print):
    """Per class: % of FAR-FIELD pixels where that class is within `tol` logits of the top-1 — i.e. how attackable it is at distance."""
    H, W = logits.shape[-2:]
    lg = logits[0, :K]
    top1 = lg.max(0).values

    dist = distance_map(H, W, *centroid(footprint), logits.device)
    far = (~footprint[0]) & (dist >= min_dist)
    if far.sum() == 0:
        log(f"\n[semantic] no pixels beyond {min_dist}px — image too small")
        return {}

    scores = {c: ((lg[c][far] > top1[far] - tol).float().mean().item() * 100)
              for c in range(K)}
    log(f"\n[semantic] far-field contestability (dist >= {min_dist:.0f}px, "
        f"within {tol} logits of top-1):")
    log("            HIGH = attackable far away; LOW = the network resists")
    for c, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        if v > 0.05:
            log(f"    {c:2d} {class_name(c):10s}: {v:5.1f}%")
    return {class_name(c): v for c, v in scores.items()}


@torch.no_grad()
def target_probability(clean_logits, adv_logits, target_class: int, footprint,
                       log=print):
    """P(target) per ring, clean vs patched."""
    H, W = clean_logits.shape[-2:]
    pc = F.softmax(clean_logits, 1)[0, target_class]
    pa = F.softmax(upsample_to(adv_logits, (H, W)), 1)[0, target_class]
    dist = distance_map(H, W, *centroid(footprint), clean_logits.device)
    outside = ~footprint[0]

    log(f"\n[semantic] mean P(class {target_class}) by ring:")
    log("      ring(px)     clean    patched      delta")
    out = []
    for lo, hi in RINGS:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() == 0:
            continue
        c, a = pc[ring].mean().item(), pa[ring].mean().item()
        out.append((lo, hi, c, a))
        log(f"    {lo:5d}-{hi:<5d}    {c:6.3f}    {a:6.3f}    {a-c:+7.3f}")
    return out
