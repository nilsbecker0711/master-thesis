r"""
DefendedSegModel — the defence as a model wrapper, decided once.

WHY A WRAPPER AND NOT A PREPROCESSING CALL IN EACH SCRIPT. setup_model() already
settles --inference by wrapping, for the reason its comment gives: the ~30
`model(x)` sites across scripts/ and patchreach/ then honour it with no
per-call branching. A defence has exactly the same shape — it is a decision
about what happens between the image and the logits — so it is settled the same
way, and a script that forgets to apply it cannot exist.

WRAPPING ORDER: DEFENSE OUTSIDE SLIDE.

    DefendedSegModel( SlidingWindowSegModel( WrappedSegModel( mmseg ) ) )

Under slide each window is an independent forward. Purifying per window would
make the defence see a 768px crop instead of the frame, and both defences in
scope are GLOBAL operations over the frame: SAC's shape completion searches
squares against the whole raw mask, and an entropy threshold is a statistic of
the whole image. A per-window defence is a different defence with the same
name. So the frame is purified once, then slid over.

GRADIENTS ARE REFUSED, LOUDLY.

Both defences in scope threshold somewhere, which zeroes the gradient. An
attack optimised through that block does not fail — it receives no signal,
stops moving, and reports a patch that does nothing. The run completes, the
numbers look like a successful defence, and nothing in the logs says otherwise.
That is the single most expensive mistake available here, so a grad-requiring
input raises instead. The adaptive arm (SAC ships a straight-through estimator
for it) is a deliberate, separate piece of work, not a default.

The practical consequence: --defence belongs to the EVAL scripts only. The
training entry points never call add_defence_args(), so argparse rejects the
flag before a GPU is touched.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from .base import PatchDefence, Purified


class DefendedSegModel(nn.Module):
    r"""
    Purify in pixel space, then forward. Same [B,3,H,W] -> [B,K,h,w] contract as
    everything else it wraps, so metrics, panels and Patch.apply need no change.

    `last` holds the Purified from the most recent defended forward, which is how
    evaluate.py reaches the predicted mask to score against the footprint. It is
    None after a bypassed forward, so a caller cannot accidentally score the
    previous image's mask against this image's footprint.
    """

    def __init__(self, model: nn.Module, defence: PatchDefence,
                 mean_t: torch.Tensor, std_t: torch.Tensor):
        super().__init__()
        self.model = model
        self.defence = defence
        self.register_buffer("_mean", mean_t.detach().clone())
        self.register_buffer("_std", std_t.detach().clone())
        self.enabled = True
        self.last: Optional[Purified] = None
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    # The inner model keeps whatever mode it was wrapped in, and aggregate.py
    # and the diagnostics read `.mode` to label reach plots.
    @property
    def mode(self) -> str:
        return getattr(self.model, "mode", "whole")

    @contextmanager
    def bypassed(self):
        r"""
        Run the inner model with no defence.

        THIS IS WHAT MAKES THE 2x2 HONEST. Undefended and defended numbers come
        from the same model instance, the same weights and the same images in
        one process, so the only difference between the two columns is the
        defence. Rebuilding the model to get the undefended column reintroduces
        every load-order and seeding difference the comparison is meant to
        exclude — and costs a second checkpoint load per architecture.
        """
        was = self.enabled
        self.enabled = False
        try:
            yield self
        finally:
            self.enabled = was

    def to_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised -> [0,1] RGB. Clamped: a patch may sit outside gamut."""
        return (x * self._std + self._mean).clamp(0.0, 1.0)

    def to_normed(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._mean) / self._std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last = None
            return self.model(x)
        if x.requires_grad or x.grad_fn is not None:
            raise RuntimeError(
                "DefendedSegModel received an input that carries gradient.\n\n"
                "Every defence here thresholds, so the gradient through it is "
                "ZERO. An attack optimised through this block does not crash — "
                "it gets no signal, the patch stops moving, and the run reports "
                "a robust model. Nothing downstream would flag it.\n\n"
                "  * evaluating a patch: wrap the forward in torch.no_grad().\n"
                "  * the per-image diagnostics need gradients, so pass\n"
                "    --diagnostics_on -1 when --defence is set.\n"
                "  * an ADAPTIVE attack through the defence is separate work: "
                "it needs a straight-through estimator (BPDA), and it must be "
                "reported as a different threat model, not as this one.\n")
        self.last = self.defence(self.to_pixels(x))
        return self.model(self.to_normed(self.last.x01))

    def describe(self, log=print) -> None:
        self.defence.describe(log)
