"""
Small utilities, vendored so the package has no project-local dependencies.

Replaces ensemble_tool.utils. Keeping them here means the repo installs and
runs standalone, which matters for the thesis: results should be reproducible
from this repo alone.
"""
from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok: bool = False) -> Path:
    """`runs/exp` -> `runs/exp_1` if it exists. Never overwrites."""
    path = Path(path)
    if not path.exists() or exist_ok:
        return path
    n = 1
    while Path(f"{path}_{n}").exists():
        n += 1
    return Path(f"{path}_{n}")


def channel_probe(model, device, h: int = 512, w: int = 1024):
    """
    (n_channels, n_numerically_active) of a segmentation head.

    Some checkpoints emit an ADE20K-dimensioned 150-channel head even on
    Cityscapes, with channels 19..149 inert. Diagnostics index by channel, so
    probe rather than assume.

    Uses amax, NOT Tensor.any — multi-dim any/all only landed in torch 2.0 and
    this codebase targets 1.11.
    """
    with torch.no_grad():
        out = model(torch.zeros(1, 3, h, w, device=device))
        return out.shape[1], int((out.abs().amax(dim=(0, 2, 3)) > 1e-3).sum())


@contextmanager
def tee_output(path):
    r"""
    Mirror stdout AND stderr into `path` for the lifetime of the block.

    WHY THIS EXISTS. results.json answers "what was the drop". It does not
    answer "did the run complain on the way there", and several things this
    pipeline says only once are said on the console: the CSF budget table, the
    degraded-after-peak warning naming best.pt, the NOT CONVERGED verdict, and
    the classes-present line. Those are the difference between a number and a
    number you can defend, and they were previously kept only in terminal
    scrollback -- or, on the cluster, in a slurm file named after a job id that
    nothing connects back to the run directory.

    Captured at the STREAM rather than through logging, because the scripts
    print, the diagnostics print, and torch prints; a logger would catch only
    the first of the three.

    Line-buffered so a run killed by the scheduler still leaves a readable log
    of everything up to the kill.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, stream):
            self._stream = stream

        def write(self, data):
            self._stream.write(data)
            fh.write(data)
            return len(data)

        def flush(self):
            self._stream.flush()
            fh.flush()

        # Delegated so anything probing the stream -- tqdm asking whether it is
        # a terminal, a subprocess asking for a descriptor -- sees the real
        # answer rather than an AttributeError.
        def isatty(self):
            return self._stream.isatty()

        def fileno(self):
            return self._stream.fileno()

    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(saved_out), _Tee(saved_err)
    try:
        yield path
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        fh.close()
