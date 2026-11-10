"""Input-space patch defences."""
from __future__ import annotations

from .base import PatchDefence, Purified
from .wrapper import DefendedSegModel

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
    """Construct a defence by --defence name."""
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
