r"""
Reach curves — attack effect as a function of distance from the patch.

Two variants, because targeted and untargeted attacks succeed differently:

  targeted   : % of ring pixels predicted as target_class
  untargeted : % of ring pixels whose argmax changed AT ALL

Both exclude the footprint and derive the centre from the footprint itself, so
they remain correct under non-central placement.

INTERPRETING THE PLATEAU: its height equals the fraction of non-target pixels
the SCENE leaves contestable, not the attack's raw power. Road plateauing at
50% means "every nearly-road pixel became road", not "everything became road" —
the other half were high-margin obstacles the attack correctly failed to claim.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from ..data.cityscapes import upsample_to
from ..losses.reach import centroid, distance_map

RINGS = [(0, 150), (150, 300), (300, 450), (450, 650), (650, 900),
         (900, 1200), (1200, 1600)]


def _rings(H, W, footprint, device, n_bins=None):
    cy, cx = centroid(footprint)
    dist = distance_map(H, W, cy, cx, device)
    outside = ~footprint[0]
    if n_bins is None:
        return dist, outside, RINGS
    max_d = dist[outside].max().item()
    edges = torch.linspace(0, max_d, n_bins + 1).tolist()
    return dist, outside, list(zip(edges[:-1], edges[1:]))


@torch.no_grad()
def targeted_reach(logits, target_class: int, footprint, n_bins: int = 12):
    """(distances, rates) — % of each ring predicted as target_class."""
    H, W = logits.shape[-2:]
    preds = logits.argmax(1)[0]
    dist, outside, bins = _rings(H, W, footprint, logits.device, n_bins)
    hit = preds == target_class

    d, r = [], []
    for lo, hi in bins:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() > 0:
            d.append((lo + hi) / 2)
            r.append(hit[ring].float().mean().item() * 100.0)
    return d, r


@torch.no_grad()
def untargeted_reach(clean_logits, adv_logits, footprint, n_bins: int = 12):
    """(distances, rates) — % of each ring whose argmax changed."""
    H, W = clean_logits.shape[-2:]
    pc = clean_logits.argmax(1)[0]
    pa = upsample_to(adv_logits, (H, W)).argmax(1)[0]
    dist, outside, bins = _rings(H, W, footprint, clean_logits.device, n_bins)
    changed = pc != pa

    d, r = [], []
    for lo, hi in bins:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() > 0:
            d.append((lo + hi) / 2)
            r.append(changed[ring].float().mean().item() * 100.0)
    return d, r


def print_curve(distances: List[float], rates: List[float], title: str,
                log=print):
    log(f"\n  {title}")
    for d, r in zip(distances, rates):
        log(f"    {d:6.0f} px : {r:5.1f}%  {'#' * int(r / 2)}")


def collapse_point(distances, rates, thresh: float = 5.0):
    """First ring centre where the rate falls below `thresh`, or None."""
    for d, r in zip(distances, rates):
        if r < thresh:
            return d
    return None
