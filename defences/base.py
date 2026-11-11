"""The defence interface: a defence LOCALISES and then REMOVES."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Purified:
    """What a defence returns."""
    x01: torch.Tensor
    mask: torch.Tensor
    raw_mask: torch.Tensor
    prob: Optional[torch.Tensor] = None

    def area_frac(self) -> torch.Tensor:
        """Fraction of the frame the final mask claims, per image. [B]"""
        return self.mask.flatten(1).mean(1)


class PatchDefence(nn.Module):
    """Base class."""

    name: str = "base"

    def forward(self, x01: torch.Tensor) -> Purified:
        raise NotImplementedError

    def describe(self, log=print) -> None:
        log(f"[def ] {self.name}")
