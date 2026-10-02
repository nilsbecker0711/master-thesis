r"""
The defence interface: a defence LOCALISES and then REMOVES.

WHY TWO OUTPUTS AND NOT ONE. Every input-space patch defence is a detector
followed by an eraser, and those two halves fail for different reasons. A
defence can score well on mIoU recovery while localising nothing (it erased
enough of the image that the patch went with it), or localise perfectly and
recover nothing (the model was never the problem). Returning only the purified
image makes those two indistinguishable, so `Purified` carries the mask too.

THE MASK IS COMPARABLE AGAINST GROUND TRUTH, FOR FREE. `Patch.apply()` already
returns the footprint it composited — the exact pixels the patch occupies. So
unlike the detection/classification settings these defences were published in,
here the predicted mask has a ground truth sitting right beside it and
`patchreach.metrics.detection` scores one against the other. That is the
measurement the defence papers could not make, and it is the one that answers
whether a CSF-bounded patch is invisible to an eraser as well as to an eye.

PIXEL SPACE, NOT NORMALISED SPACE. Every defence here takes and returns
x01 in [0,1] RGB. The published implementations assume that range (SAC's
patch segmenter was trained on it), entropy and contrast statistics are only
meaningful in it, and a defence that silently received ImageNet-normalised
input would still run and still produce a plausible-looking mask. The
denormalise/renormalise round trip lives in DefendedSegModel, once, for the
same reason --inference is decided once in setup_model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Purified:
    r"""
    What a defence returns.

    x01      [B,3,H,W] the defended image, pixel space [0,1]
    mask     [B,1,H,W] float 0/1 — the FINAL mask, after any shape completion.
             1 = "this is patch", which is the region removed from x01.
    raw_mask [B,1,H,W] float 0/1 — the mask BEFORE completion/post-processing.

    prob     [B,1,H,W] float — the detector's PRE-THRESHOLD score, if it has one.

    raw_mask is not diagnostic clutter: SAC's and Jedi's completion steps both
    grow the mask, and growth is what produces false positives on clean images.
    Scoring both against the footprint separates "the detector missed it" from
    "the detector found it and the completion over-grew".

    AND `prob` SEPARATES THE THIRD CASE, WHICH IS THE ONE THAT ACTUALLY HAPPENS.
    An empty raw_mask is consistent with two completely different failures: the
    detector scored the patch near zero (out of domain — no threshold rescues
    it), or it scored it at 0.4 and lost to the 0.5 cut (in domain, wrong
    operating point — a scale change or a threshold sweep fixes it). Those have
    opposite conclusions, and a 0/1 mask cannot tell them apart. Keeping the
    score costs one float map per image.
    """
    x01: torch.Tensor
    mask: torch.Tensor
    raw_mask: torch.Tensor
    prob: Optional[torch.Tensor] = None

    def area_frac(self) -> torch.Tensor:
        """Fraction of the frame the final mask claims, per image. [B]"""
        return self.mask.flatten(1).mean(1)


class PatchDefence(nn.Module):
    r"""
    Base class. Subclasses implement forward(x01) -> Purified.

    A defence is an nn.Module so its weights (SAC has a U-Net) move with
    .to(device) and register in the usual way, NOT because it is ever trained
    here. Everything in this package is evaluation-only; see
    DefendedSegModel's grad guard for what enforces that.
    """

    #: short name used by --defence and in output paths
    name: str = "base"

    def forward(self, x01: torch.Tensor) -> Purified:
        raise NotImplementedError

    def describe(self, log=print) -> None:
        log(f"[def ] {self.name}")
