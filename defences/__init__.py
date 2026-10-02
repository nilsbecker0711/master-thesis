r"""
Input-space patch defences. SELF-CONTAINED BY DESIGN.

Nothing outside this folder is modified to make it work. The defence is applied
by wrapping the model that `scripts/_common.py:setup_model` already returns, and
the entry point is `defences/evaluate_defended.py`, so `scripts/`, `patchreach/`
and `tests/` are untouched and every existing run keeps producing exactly the
numbers it produced before.

WHAT THAT COSTS, stated plainly: evaluate_defended.py repeats the per-image
sequence that `scripts/evaluate.py` runs (resolve_placement -> maybe
set_reference_from_image -> apply), because the alternative was a --defence flag
threaded through the shared plumbing. If that sequence ever changes in
evaluate.py it must be changed here too — `test_defences.py` pins the ordering
and the reason, but a test cannot notice a change made only on the other side.
The undefended/defended COMPARISON is not at risk either way: both columns come
from one model instance in one process via DefendedSegModel.bypassed(), so they
share whatever this file does.

    defences/
      base.py         PatchDefence, Purified — the interface
      wrapper.py      DefendedSegModel — denormalise, purify, renormalise
      detection.py    predicted mask vs the TRUE footprint
      sac/            Segment and Complete (Liu et al., CVPR 2022)
      jedi/           Jedi, entropy-based (Tarchoun et al., CVPR 2023)
      evaluate_defended.py
      test_defences.py

    --defence none   no wrapper at all
    --defence sac    Segment and Complete (learned segmenter + square prior)
    --defence jedi   entropy threshold + blob cleanup + inpainting

SAC AND JEDI DETECT ON DIFFERENT PRINCIPLES, which is the only reason having both
is worth anything. SAC is a LEARNED detector: model-independent, but neither
dataset- nor scale-independent, and the first real run found it out of domain on
Cityscapes at 1024x2048. Jedi is arithmetic plus twelve calibration scalars, so
it has no weights to fall out of domain — and it thresholds on local ENTROPY,
which a contrast budget bounds directly, making it the defence whose detection
statistic is the same physical quantity the attack constrains.

ram (Yuan et al., IJCV 2024 — arXiv:2401.01750, the same paper the bracket
taxonomy in patchreach/models/registry.py rests on) is named here with a `None`
builder so `--defence ram` fails with "not implemented yet" rather than "invalid
choice", and so a sweep over the full list fails on the cell instead of silently
skipping it.

RAM WILL NOT LIVE HERE. It refines the attention matrix inside the backbone, so
it is model surgery rather than an input-space wrapper, and it exists only for
segformer_* and setr_pup. Its NaN column for the no-attention bracket is a
finding, not a gap.
"""
from __future__ import annotations

from .base import PatchDefence, Purified
from .wrapper import DefendedSegModel

#: name -> builder key, or None for "named but unimplemented".
DEFENCES = {
    "none": None,
    "sac": "sac",
    "jedi": "jedi",
    "ram": None,
}

_UNIMPLEMENTED = {
    "ram": "RAM (Yuan et al., IJCV 2024, arXiv:2401.01750). Not an input-space "
           "defence — it rewrites the attention matrix inside the backbone, so "
           "it belongs beside the model, applies only to segformer_*/setr_pup, "
           "and its clean-mIoU cost on stock mmseg checkpoints has to be "
           "measured before any robustness number from it means anything.",
}


def build_defence(name: str, device="cpu", log=print, **kw) -> PatchDefence:
    """Construct a defence by --defence name. 'none' is the caller's job."""
    if name not in DEFENCES:
        raise SystemExit(f"\nunknown --defence {name!r}. "
                         f"Known: {sorted(DEFENCES)}\n")
    if name == "none":
        raise ValueError("build_defence('none') — the caller should not wrap.")
    if DEFENCES[name] is None:
        raise SystemExit(f"\n--defence {name} is not implemented yet.\n\n"
                         f"  {_UNIMPLEMENTED[name]}\n")
    if name == "sac":
        from .sac import SACDefence
        return SACDefence(device=device, log=log, **kw)
    if name == "jedi":
        from .jedi import JediDefence
        return JediDefence(device=device, log=log, **kw)
    raise AssertionError(f"builder for {name!r} is registered but unreachable")


__all__ = ["PatchDefence", "Purified", "DefendedSegModel", "DEFENCES",
           "build_defence"]
