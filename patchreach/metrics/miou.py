r"""
mIoU and attack-effect metrics.

REMOTE vs ALL — the distinction the whole thesis rests on. drop_all conflates
occlusion (the patch physically covering pixels) with genuine adversarial
influence; drop_remote excludes the footprint and measures only the latter.
Report remote.

PER-IMAGE vs DATASET mIoU — these are NOT comparable. Per-image mIoU averages
IoU over classes present in ONE image, so a rare class covering a few hundred
pixels scores near zero and drags the mean down hard; 50 per-image is normal
for a model that scores 76 on the dataset. Dataset mIoU accumulates one
confusion matrix across many images. Published numbers are always dataset mIoU.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..data.cityscapes import upsample_to


class SegMetric:
    """Confusion-matrix mIoU. iou_c = TP/(TP+FP+FN); NaN if the class is absent."""

    def __init__(self, K: int, ignore_index: int = 255, device="cpu"):
        self.K, self.ig = K, ignore_index
        self.cm = torch.zeros(K, K, dtype=torch.long, device=device)

    def reset(self):
        self.cm.zero_()

    @torch.no_grad()
    def update(self, preds, labels, exclude: Optional[torch.Tensor] = None):
        valid = labels != self.ig
        if exclude is not None:
            valid = valid & (~exclude)
        p, l = preds[valid].long(), labels[valid].long()
        self.cm.view(-1).scatter_add_(0, l * self.K + p, torch.ones_like(l))

    @torch.no_grad()
    def per_class(self) -> torch.Tensor:
        cm = self.cm.float()
        tp = cm.diagonal()
        denom = cm.sum(0) + cm.sum(1) - tp
        return torch.where(denom > 0, tp / denom,
                           torch.full_like(tp, float("nan"))) * 100.0

    @torch.no_grad()
    def compute(self) -> float:
        return self.per_class().nanmean().item()


@torch.no_grad()
def single_image_miou(logits, labels, K: int,
                      exclude: Optional[torch.Tensor] = None) -> float:
    m = SegMetric(K, device=logits.device)
    m.update(upsample_to(logits, labels.shape[-2:]).argmax(1), labels, exclude)
    return m.compute()


@torch.no_grad()
def attack_rates(clean_logits, adv_logits, labels, footprint,
                 target_class: Optional[int] = None) -> dict:
    """
    any_flip_rate   : % of remote pixels whose argmax changed at all.
    target_hit_rate : % of NON-target remote pixels flipped TO target_class.

    The hit-rate denominator excludes pixels already predicted target on the
    CLEAN image — they cannot be flipped TO a class they already are. Without
    that exclusion, attacking a scene-dominant class (road, ~50% prior) looks
    like a strong attack when the scene simply handed you the win. This is the
    metric that separates "network overpowered" from "network agreed".
    """
    hw = labels.shape[-2:]
    pc = upsample_to(clean_logits, hw).argmax(1)
    pa = upsample_to(adv_logits, hw).argmax(1)
    remote = (labels != 255) & (~footprint)

    n = int(remote.sum())
    out = {"any_flip_rate": 100.0 * int((remote & (pc != pa)).sum()) / max(n, 1)}

    if target_class is not None:
        not_tgt = remote & (pc != target_class)
        d = int(not_tgt.sum())
        out["target_hit_rate"] = (
            100.0 * int((not_tgt & (pa == target_class)).sum()) / max(d, 1))
    return out
