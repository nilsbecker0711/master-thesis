"""Small utilities, vendored so the package has no project-local dependencies."""
from __future__ import annotations

import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atomic_save(obj, path) -> None:
    """Write to a temporary file, then rename over the target."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok: bool = False) -> Path:
    """`runs/exp` -> `runs/exp_1` if it exists."""
    path = Path(path)
    if not path.exists() or exist_ok:
        return path
    n = 1
    while Path(f"{path}_{n}").exists():
        n += 1
    return Path(f"{path}_{n}")


def channel_probe(model, device, h: int = 512, w: int = 1024):
    """(n_channels, n_numerically_active) of a segmentation head."""
    with torch.no_grad():
        out = model(torch.zeros(1, 3, h, w, device=device))
        return out.shape[1], int((out.abs().amax(dim=(0, 2, 3)) > 1e-3).sum())


@contextmanager
def tee_output(path):
    """Mirror stdout AND stderr into `path` for the lifetime of the block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)

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
