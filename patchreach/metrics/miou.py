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


CLASS_SETS = ("gt", "union")


class SegMetric:
    r"""
    Confusion-matrix mIoU. iou_c = TP/(TP+FP+FN).

    WHICH CLASSES ENTER THE MEAN — this decides whether a clean/attacked
    comparison is valid at all.

      'gt'    (default) count a class iff it has GROUND TRUTH in the evaluated
              pixels. The set then depends only on the labels, which are
              identical for the clean and the attacked pass, so both means run
              over the SAME denominator.

      'union' count a class iff it has ground truth OR is predicted anywhere.
              The historical behaviour, kept so earlier runs stay reproducible.

    WHY 'union' BREAKS ATTACK EVALUATION. Under 'union' a class with no ground
    truth that the model nevertheless predicts scores IoU = 0 and drags the
    nanmean down. Suppress that spurious prediction and the class becomes NaN,
    leaves the denominator, and the mean JUMPS — by roughly mean/(n-1), about
    3.7 points at 18 classes. Clean and attacked are then averaged over
    different class sets and the difference between them is not a measurement.

    Observed: a 150-epoch run reported drop_remote = -3.63, i.e. the attack
    apparently IMPROVED mIoU by 3.6 points, entirely because one rare class
    (no GT in the 20 val images, a handful of predicted pixels) stopped being
    counted. A toy case makes the sensitivity plain: 5 spurious pixels in
    40,000 move mIoU by 33 points.

    'gt' cannot do this: a class absent from the ground truth is not evaluated
    in either pass, so nothing can move in or out. The cost is that spurious
    predictions of a GT-absent class are no longer penalised — but they were
    only ever penalised by an unstable all-or-nothing term.
    """

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
        gt = cm.sum(1)                       # ground-truth pixels per class
        denom = cm.sum(0) + gt - tp          # GT + predicted - TP
        # gt > 0 implies denom > 0, so the division is safe either way.
        keep = (gt > 0) if classes == "gt" else (denom > 0)
        return torch.where(keep, tp / denom.clamp(min=1.0),
                           torch.full_like(tp, float("nan"))) * 100.0

    @torch.no_grad()
    def compute(self, classes: str = "gt") -> float:
        return self.per_class(classes).nanmean().item()

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
    r"""
    Clean vs attacked under BOTH class sets, plus the class counts that make
    any divergence between them legible.

    Reporting both is deliberate. 'gt' is the number to trust; 'union' is kept
    so runs recorded before this distinction existed remain comparable, and
    because the GAP between them is itself a diagnostic — a large gap means a
    rare class moved in or out of the denominator, which says something about
    the attack that neither number says alone.
    """
    out = {}
    for cs in CLASS_SETS:
        suffix = "" if cs == "gt" else "_union"
        c, a = clean.compute(cs), adv.compute(cs)
        out[f"{prefix}clean{suffix}"] = c
        out[f"{prefix}adv{suffix}"] = a
        out[f"{prefix}drop{suffix}"] = c - a
    out[f"{prefix}n_classes_gt"] = clean.n_counted("gt")
    out[f"{prefix}n_classes_clean_union"] = clean.n_counted("union")
    out[f"{prefix}n_classes_adv_union"] = adv.n_counted("union")
    return out


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
