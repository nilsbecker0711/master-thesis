"""mIoU and attack-effect metrics."""
from __future__ import annotations

from typing import Optional

import torch

from ..data.cityscapes import upsample_to


CLASS_SETS = ("gt", "union")


class SegMetric:
    """Confusion-matrix mIoU. iou_c = TP/(TP+FP+FN)."""

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
    def per_class(self, classes: str = "gt") -> torch.Tensor:
        if classes not in CLASS_SETS:
            raise ValueError(f"classes must be one of {CLASS_SETS}, got {classes!r}")
        cm = self.cm.float()
        tp = cm.diagonal()
        gt = cm.sum(1)
        denom = cm.sum(0) + gt - tp
        keep = (gt > 0) if classes == "gt" else (denom > 0)
        return torch.where(keep, tp / denom.clamp(min=1.0),
                           torch.full_like(tp, float("nan"))) * 100.0

    @torch.no_grad()
    def per_class_acc(self) -> torch.Tensor:
        """Per-class RECALL, tp / gt — the quantity averaged into mAcc."""
        cm = self.cm.float()
        tp, gt = cm.diagonal(), cm.sum(1)
        return torch.where(gt > 0, tp / gt.clamp(min=1.0),
                           torch.full_like(tp, float("nan"))) * 100.0

    @torch.no_grad()
    def compute(self, classes: str = "gt") -> float:
        return self.per_class(classes).nanmean().item()

    @torch.no_grad()
    def compute_acc(self) -> float:
        return self.per_class_acc().nanmean().item()

    @torch.no_grad()
    def n_counted(self, classes: str = "gt") -> int:
        return int((~torch.isnan(self.per_class(classes))).sum())


@torch.no_grad()
def single_image_miou(logits, labels, K: int,
                      exclude: Optional[torch.Tensor] = None,
                      classes: str = "gt") -> float:
    m = SegMetric(K, device=logits.device)
    m.update(upsample_to(logits, labels.shape[-2:]).argmax(1), labels, exclude)
    return m.compute(classes)


@torch.no_grad()
def compare(clean: SegMetric, adv: SegMetric, prefix: str = "") -> dict:
    """Clean vs attacked under BOTH class sets, plus the class counts that make any divergence between them legible."""
    out = {}
    for cs in CLASS_SETS:
        suffix = "" if cs == "gt" else "_union"
        c, a = clean.compute(cs), adv.compute(cs)
        out[f"{prefix}clean{suffix}"] = c
        out[f"{prefix}adv{suffix}"] = a
        out[f"{prefix}drop{suffix}"] = c - a
    ca, aa = clean.compute_acc(), adv.compute_acc()
    out[f"{prefix}clean_acc"] = ca
    out[f"{prefix}adv_acc"] = aa
    out[f"{prefix}drop_acc"] = ca - aa
    out[f"{prefix}n_classes_gt"] = clean.n_counted("gt")
    out[f"{prefix}n_classes_clean_union"] = clean.n_counted("union")
    out[f"{prefix}n_classes_adv_union"] = adv.n_counted("union")
    return out


@torch.no_grad()
def attack_rates(clean_logits, adv_logits, labels, footprint,
                 target_class: Optional[int] = None) -> dict:
    """any_flip_rate : % of remote pixels whose argmax changed at all. target_hit_rate : % of NON-target remote pixels flipped TO target_class."""
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
