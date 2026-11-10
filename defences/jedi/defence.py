"""Jedi — entropy-based localisation and removal of adversarial patches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ..base import PatchDefence, Purified

PATCH_MEAN = (5.32, 6.06, 6.47)
PATCH_STD = (0.32, 0.34, 0.35)
CLEAN_MEAN = (2.75, 3.19, 3.47)
CLEAN_STD = (1.16, 1.13, 1.07)

MIN_BLOB_FRAC = 0.005


def window_sizes(h: int, w: int) -> tuple:
    """The three (window, stride) pairs, from the frame size. jedi.m lines 56-63."""
    sx = -(-h // 100)
    sy = -(-w // 100)
    sx += sx % 2
    sy += sy % 2
    s2 = max(sx, sy, 8)
    ws = []
    for mult in (1.0, 1.5, 2.0):
        v = int(s2 * mult)
        ws.append(v + v % 2)
    return tuple((v, v // 2) for v in ws)


def _symmetric_pad(x: torch.Tensor, py: int, px: int) -> torch.Tensor:
    """MATLAB padarray(...,'symmetric'): reflect INCLUDING the edge pixel."""
    if py:
        top = x[..., :py, :].flip(-2)
        bot = x[..., -py:, :].flip(-2)
        x = torch.cat([top, x, bot], dim=-2)
    if px:
        left = x[..., :, :px].flip(-1)
        right = x[..., :, -px:].flip(-1)
        x = torch.cat([left, x, right], dim=-1)
    return x


def _window_entropy(gray_u8: torch.Tensor, win: int,
                    stride: int) -> torch.Tensor:
    """256-bin Shannon entropy in bits for every strided window. [ny, nx]."""
    k = win + 1
    patches = F.unfold(gray_u8.view(1, 1, *gray_u8.shape[-2:]).float(),
                       kernel_size=k, stride=stride)
    vals = patches[0].long().T.contiguous()
    L = vals.shape[0]
    hist = torch.zeros(L, 256, device=vals.device, dtype=torch.float64)
    hist.scatter_add_(1, vals, torch.ones_like(vals, dtype=torch.float64))
    p = hist / float(k * k)
    ent = -(p * torch.where(p > 0, p.log2(), torch.zeros_like(p))).sum(1)
    H, W = gray_u8.shape[-2:]
    ny = (H - k) // stride + 1
    nx = (W - k) // stride + 1
    return ent[:ny * nx].view(ny, nx)


def entropy_heatmap(gray_u8: torch.Tensor, win: int, stride: int):
    """(heatmap [H,W], mean window entropy)."""
    H, W = gray_u8.shape[-2:]
    py = px = win // 2
    padded = _symmetric_pad(gray_u8, py, px)
    ent = _window_entropy(padded, win, stride)
    ny, nx = ent.shape
    ph, pw = padded.shape[-2:]
    iy = (torch.arange(ph, device=ent.device) // stride).clamp(max=ny - 1)
    ix = (torch.arange(pw, device=ent.device) // stride).clamp(max=nx - 1)
    full = ent[iy][:, ix]
    return full[py:py + H, px:px + W], float(ent.mean())


def entropy_threshold(img_mean: float, i: int, cal: dict) -> float:
    """The per-image adaptive threshold, jedi.m lines 176-186."""
    pm, ps = cal["patch_mean"][i], cal["patch_std"][i]
    cm, cs = cal["clean_mean"][i], cal["clean_std"][i]
    img_spread = (img_mean - cm) / cs
    dev_mult = ((pm - cm) / pm) * 2.0
    return (pm + 1.5 * ps) - (dev_mult * ps) + (img_spread * ps)


def drop_small_blobs(mask: torch.Tensor,
                     min_frac: float = MIN_BLOB_FRAC) -> torch.Tensor:
    """Remove connected components below `min_frac` of the frame."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    out = torch.zeros_like(mask)
    npx = mask.shape[-1] * mask.shape[-2]
    for b in range(mask.shape[0]):
        lab, n = ndimage.label(mask[b, 0].detach().cpu().numpy() > 0.5)
        if n == 0:
            continue
        keep = torch.zeros_like(mask[b, 0])
        lab_t = torch.as_tensor(lab, device=mask.device)
        for comp in range(1, n + 1):
            sel = lab_t == comp
            if float(sel.sum()) / npx >= min_frac:
                keep[sel] = 1.0
        out[b, 0] = keep
    return out


def load_calibration(path: Optional[str] = None) -> dict:
    """The authors' constants, or a JSON written by measure_entropy_stats()."""
    cal = {"patch_mean": list(PATCH_MEAN), "patch_std": list(PATCH_STD),
           "clean_mean": list(CLEAN_MEAN), "clean_std": list(CLEAN_STD),
           "source": "Tarchoun et al. jedi.m (PASCAL/INRIA scale, YOLO patches)"}
    if not path:
        return cal
    with open(path) as f:
        got = json.load(f)
    for k in ("patch_mean", "patch_std", "clean_mean", "clean_std"):
        if k not in got or len(got[k]) != 3:
            raise SystemExit(
                f"\n--jedi_calibration {path} is missing '{k}' or it is not 3 "
                f"values (one per window size). Write it with "
                f"measure_entropy_stats().\n")
        cal[k] = [float(v) for v in got[k]]
    cal["source"] = got.get("source", str(Path(path).name))
    return cal


def measure_entropy_stats(images, masks=None, device="cpu") -> dict:
    """Re-measure the twelve calibration constants on OUR data."""
    acc = {"clean": [[], [], []], "patch": [[], [], []]}
    for n, img in enumerate(images):
        img = img.to(device)
        gray = _to_gray_u8(img)
        H, W = gray.shape[-2:]
        m = None
        if masks is not None:
            m = masks[n] if not torch.is_tensor(masks) else masks
            m = m.view(1, 1, H, W).to(device) > 0.5
        for i, (win, stride) in enumerate(window_sizes(H, W)):
            padded = _symmetric_pad(gray, win // 2, win // 2)
            ent = _window_entropy(padded, win, stride)
            if m is None:
                acc["clean"][i].append(ent.flatten())
                continue
            ny, nx = ent.shape
            cy = (torch.arange(ny, device=device) * stride + win // 2
                  - win // 2).clamp(0, H - 1)
            cx = (torch.arange(nx, device=device) * stride + win // 2
                  - win // 2).clamp(0, W - 1)
            inside = m[0, 0][cy][:, cx]
            acc["patch"][i].append(ent[inside].flatten())
            acc["clean"][i].append(ent[~inside].flatten())
    out = {"source": "measured by defences.jedi.measure_entropy_stats"}
    for pop, keys in (("clean", ("clean_mean", "clean_std")),
                      ("patch", ("patch_mean", "patch_std"))):
        means, stds = [], []
        for i in range(3):
            vals = [v for v in acc[pop][i] if v.numel()]
            if not vals:
                means.append(float(PATCH_MEAN[i] if pop == "patch"
                                   else CLEAN_MEAN[i]))
                stds.append(float(PATCH_STD[i] if pop == "patch"
                                  else CLEAN_STD[i]))
                continue
            v = torch.cat(vals)
            means.append(float(v.mean()))
            stds.append(float(v.std()))
        out[keys[0]], out[keys[1]] = means, stds
    return out


def _to_gray_u8(x01: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] in [0,1] -> [B,1,H,W] uint8-valued float, MATLAB rgb2gray."""
    w = torch.tensor([0.298936, 0.587043, 0.114021],
                     device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    g = (x01 * w).sum(1, keepdim=True)
    return (g * 255.0).round().clamp(0, 255)


class JediDefence(PatchDefence):
    """Jedi as an input-space defence."""

    name = "jedi"

    def __init__(self, calibration: Optional[str] = None,
                 fill: str = "inpaint", inpaint_radius: int = 3,
                 autoencoder=None, ae_threshold: float = 0.33,
                 blobs: bool = True, min_blob_frac: float = MIN_BLOB_FRAC,
                 device="cpu", log=print):
        super().__init__()
        if fill not in ("inpaint", "grey", "zero"):
            raise SystemExit(f"\n--jedi_fill must be inpaint|grey|zero, "
                             f"got {fill!r}\n")
        self.cal = load_calibration(calibration)
        self.fill = fill
        self.inpaint_radius = int(inpaint_radius)
        self.autoencoder = autoencoder
        self.ae_threshold = float(ae_threshold)
        self.blobs = bool(blobs)
        self.min_blob_frac = float(min_blob_frac)
        self._warned_cv2 = False
        self.log = log
        if autoencoder is not None:
            autoencoder.to(device).eval()
        self.eval()

    def masks(self, x01: torch.Tensor):
        """(final mask, pre-cleanup mask, summed heatmap)."""
        gray = _to_gray_u8(x01)
        H, W = x01.shape[-2:]
        raw = torch.zeros(x01.shape[0], 1, H, W, device=x01.device,
                          dtype=x01.dtype)
        score = torch.zeros_like(raw)
        for b in range(x01.shape[0]):
            for i, (win, stride) in enumerate(window_sizes(H, W)):
                hm, img_mean = entropy_heatmap(gray[b:b + 1], win, stride)
                thr = entropy_threshold(img_mean, i, self.cal)
                over = (hm - thr).clamp(min=0)
                raw[b, 0] += (over > 0).to(raw.dtype)
                score[b, 0] = torch.maximum(score[b, 0], over.to(score.dtype))
        raw = (raw > 0).to(x01.dtype)
        mask = drop_small_blobs(raw, self.min_blob_frac) if self.blobs else raw
        if self.autoencoder is not None:
            with torch.no_grad():
                mask = (self.autoencoder(mask) > self.ae_threshold).to(x01.dtype)
        return mask, raw, score

    def _remove(self, x01: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.fill == "zero":
            return x01 * (1.0 - mask)
        if self.fill == "grey":
            return x01 * (1.0 - mask) + 0.5 * mask
        try:
            import cv2
        except ImportError:
            if not self._warned_cv2:
                self.log("[def ] jedi: cv2 unavailable, falling back to a grey "
                         "fill. This is NOT the published removal — say so.")
                self._warned_cv2 = True
            return x01 * (1.0 - mask) + 0.5 * mask
        out = x01.clone()
        for b in range(x01.shape[0]):
            m = (mask[b, 0].detach().cpu().numpy() > 0.5).astype("uint8")
            if not m.any():
                continue
            img = (x01[b].detach().cpu().permute(1, 2, 0).numpy() * 255.0)
            img = img.round().clip(0, 255).astype("uint8")
            fixed = cv2.inpaint(img, m, self.inpaint_radius, cv2.INPAINT_TELEA)
            f = torch.as_tensor(fixed, device=x01.device,
                                dtype=x01.dtype).permute(2, 0, 1) / 255.0
            mb = mask[b]
            out[b] = x01[b] * (1.0 - mb) + f * mb
        return out

    def forward(self, x01: torch.Tensor) -> Purified:
        mask, raw, score = self.masks(x01)
        return Purified(x01=self._remove(x01, mask), mask=mask, raw_mask=raw,
                        prob=score)

    def describe(self, log=print) -> None:
        log(f"[def ] jedi  fill={self.fill}  blobs={self.blobs}"
            f"  ae={'yes' if self.autoencoder is not None else 'no'}")
        log(f"[def ] calibration: {self.cal['source']}")
        log(f"[def ]   patch mean {self.cal['patch_mean']} "
            f"std {self.cal['patch_std']}")
        log(f"[def ]   clean mean {self.cal['clean_mean']} "
            f"std {self.cal['clean_std']}")
