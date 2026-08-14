r"""
UNTARGETED diagnostics — "where does the patch cause confusion, and what kind?"

Targeted diagnostics ask "can the patch impose class t here?". These ask a
different question and need different instruments.

THE FINDING THESE EXIST TO SURFACE: an untargeted loss has no class objective,
yet it converges on ONE confusion channel. Measured here: ~97% of all remote
flips went road -> car, with car IoU dropping ~68 points in every image. The
loss is an IMPLICIT CLASS SELECTOR, not a broad disruptor — it finds the most
exploitable boundary and exploits only that. The confusion matrix is what makes
that visible; aggregate mIoU hides it completely.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.cityscapes import class_name, upsample_to
from ..losses.reach import centroid, distance_map

RINGS = [(0, 150), (150, 300), (300, 450), (450, 650), (650, 900)]


@torch.no_grad()
def winner_margin(clean_logits, adv_logits, footprint, K: int, log=print):
    """
    max_logit - second_max_logit, clean vs patched.

    This is the most SENSITIVE reach measurement available. The argmax-flip map
    shows completed flips; the margin delta shows pixels the patch has pushed
    toward a flip WITHOUT completing one. The blue region in the delta reliably
    extends past the flip zone — the attack's influence is wider than its
    effect, which is what motivated testing untargeted attacks in the first
    place.

    topk is capped at K: heads emitting 150 channels would otherwise compute
    the runner-up from inert channels.
    """
    H, W = clean_logits.shape[-2:]

    def margin(lg):
        t2 = lg[0, :K].topk(2, dim=0).values
        return t2[0] - t2[1]

    mc = margin(clean_logits)
    ma = margin(upsample_to(adv_logits, (H, W)))
    delta = ma - mc

    dist = distance_map(H, W, *centroid(footprint), clean_logits.device)
    outside = ~footprint[0]

    log("\n[untargeted] winner margin by ring "
        "(negative delta = confidence eroded):")
    log("      ring(px)     clean    patched      delta")
    out = []
    for lo, hi in RINGS:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() == 0:
            continue
        c, a = mc[ring].mean().item(), ma[ring].mean().item()
        out.append((lo, hi, c, a))
        log(f"    {lo:5d}-{hi:<5d}    {c:6.2f}    {a:6.2f}    {a-c:+7.2f}")
    return mc.cpu(), ma.cpu(), delta.cpu(), out


@torch.no_grad()
def confusion(clean_logits, adv_logits, labels, footprint, K: int, log=print):
    """
    What did flipped pixels become?

    Returns per-class flip rate plus the source->destination flow table. A
    STRUCTURED matrix (a few flows dominating) means the untargeted loss is
    behaving like a weak targeted attack; a DIFFUSE one means genuine
    broad-spectrum confusion. These have completely different security
    implications and mIoU cannot tell them apart.
    """
    hw = labels.shape[-2:]
    pc = upsample_to(clean_logits, hw).argmax(1)[0]
    pa = upsample_to(adv_logits, hw).argmax(1)[0]
    remote = (labels[0] != 255) & (~footprint[0])
    changed = (pc != pa) & remote

    if changed.sum() == 0:
        log("\n[untargeted] confusion: no remote pixels flipped")
        return {}, []

    src = pc[changed].clamp(0, K - 1)
    dst = pa[changed].clamp(0, K - 1)
    cm = torch.zeros(K, K, dtype=torch.long, device=pc.device)
    cm.view(-1).scatter_add_(0, src * K + dst, torch.ones_like(src))

    log("\n[untargeted] per-class flip rate (% of that class's remote px):")
    rates = {}
    for c in range(K):
        total = int(((pc == c) & remote).sum())
        if total < 50:
            continue
        flipped = int(cm[c].sum())
        r = 100.0 * flipped / total
        rates[class_name(c)] = r
        log(f"    {c:2d} {class_name(c):10s}: {r:5.1f}%  "
            f"({flipped:,}/{total:,})  {'#' * int(r / 5)}")

    flows = sorted(((int(cm[s, d]), s, d) for s in range(K) for d in range(K)
                    if s != d and cm[s, d] > 0), reverse=True)
    total_changed = int(changed.sum())
    log(f"\n    top source->destination flows "
        f"({total_changed:,} flipped px total):")
    for n, s, d in flows[:5]:
        log(f"    {class_name(s):10s} -> {class_name(d):10s} : {n:8,} px "
            f"({100*n/total_changed:4.1f}%)")
    if flows and flows[0][0] / total_changed > 0.5:
        log(f"    -> STRUCTURED: one channel carries "
            f"{100*flows[0][0]/total_changed:.0f}% of all flips. The "
            f"untargeted loss is acting as an implicit class selector.")
    return rates, [(n, class_name(s), class_name(d)) for n, s, d in flows[:10]]


@torch.no_grad()
def entropy(clean_logits, adv_logits, footprint, K: int, log=print):
    """
    Normalised Shannon entropy, clean vs patched.

    Complements the winner margin: margin measures the top-2 gap, entropy
    measures uncertainty across ALL classes. They can disagree — a patch can
    narrow the top-2 gap while leaving the rest of the distribution untouched.
    """
    H, W = clean_logits.shape[-2:]

    def ent(lg):
        p = F.softmax(lg[:, :K], 1)[0]
        return -(p * (p + 1e-10).log()).sum(0) / np.log(K)

    ec = ent(clean_logits)
    ea = ent(upsample_to(adv_logits, (H, W)))
    dist = distance_map(H, W, *centroid(footprint), clean_logits.device)
    outside = ~footprint[0]

    log("\n[untargeted] prediction entropy by ring:")
    out = []
    for lo, hi in RINGS:
        ring = outside & (dist >= lo) & (dist < hi)
        if ring.sum() == 0:
            continue
        c, a = ec[ring].mean().item(), ea[ring].mean().item()
        out.append((lo, hi, c, a))
        log(f"    {lo:5d}-{hi:<5d}    {c:6.3f}    {a:6.3f}    {a-c:+7.3f}")
    return ec.cpu(), ea.cpu(), out


@torch.no_grad()
def reach_by_source_class(clean_logits, adv_logits, labels, footprint, K: int,
                          n_bins: int = 8, log=print):
    """
    Flip rate vs distance, split by what was predicted BEFORE the attack.

    If high-confidence classes (road, sky) flip as readily as low-confidence
    ones, the attack overcomes the network. If only naturally-uncertain classes
    flip, it merely exploits pre-existing uncertainty. That distinction decides
    how the result should be framed.
    """
    hw = labels.shape[-2:]
    pc = upsample_to(clean_logits, hw).argmax(1)[0]
    pa = upsample_to(adv_logits, hw).argmax(1)[0]
    remote = (labels[0] != 255) & (~footprint[0])
    changed = (pc != pa) & remote

    H, W = hw
    dist = distance_map(H, W, *centroid(footprint), pc.device)
    max_d = dist[remote].max().item()
    edges = torch.linspace(0, max_d, n_bins + 1).tolist()

    log("\n[untargeted] flip rate vs distance, by SOURCE class:")
    out = {}
    for c in range(K):
        cr = remote & (pc == c)
        if cr.sum() < 50:
            continue
        pts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            ring = cr & (dist >= lo) & (dist < hi)
            if ring.sum() < 10:
                continue
            pts.append(((lo + hi) / 2,
                        (changed & ring).float().sum().item()
                        / ring.float().sum().item() * 100))
        if pts:
            out[class_name(c)] = pts
            log(f"    {c:2d} {class_name(c):10s}: "
                + "  ".join(f"{r:.0f}%@{d:.0f}px" for d, r in pts))
    return out