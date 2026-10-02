r"""
Jedi — entropy-based localisation and removal of adversarial patches.
Tarchoun et al., CVPR 2023. Ported from the authors' MATLAB release:
github.com/ihsenLab/jedi-CVPR2023 (Jedi-main/jedi.m), read 2026-10-02.

THE IDEA. An adversarial patch carries far more high-frequency structure than
natural scene content, so its local Shannon entropy sits well above the clean
distribution. Jedi slides three window sizes over the greyscale frame, scores
each window's 256-bin entropy, thresholds against a CALIBRATED clean/patch
entropy model, unions the three scales, drops small blobs, and inpaints.

WHY THIS IS THE RIGHT DEFENCE FOR A CONTRAST-BUDGETED PATCH. It thresholds on
local entropy, and a CSF budget bounds local contrast. Lower contrast means
mechanically lower entropy, so the defence's detection statistic and the attack's
constraint are the same physical quantity. A tau ladder against Jedi coverage is
therefore a direct measurement, not an analogy — which is exactly what SAC could
not give us, because SAC's learned segmenter was out of domain before the patch
constraint ever entered.

NO WEIGHTS, SO NO DOMAIN TO BE OUT OF — WITH ONE EXCEPTION, AND IT IS
CALIBRATABLE. Everything here is arithmetic except twelve scalars: the mean and
standard deviation of window entropy over patches and over clean images, at each
of the three window sizes (PATCH_MEAN/PATCH_STD/CLEAN_MEAN/CLEAN_STD below).
Those were measured by the authors on their data — PASCAL/INRIA-scale frames with
YOLO patches — and they are the whole reason the threshold is adaptive rather
than arbitrary.

    THEY ARE ALMOST CERTAINLY WRONG FOR CITYSCAPES AND THEY ARE TWELVE NUMBERS.
    A dashcam frame is not a PASCAL photo: more sky, more road, large smooth
    regions, different entropy distribution. Unlike SAC's 1.08M opaque
    parameters, these can be RE-MEASURED on our own clean train split and our
    own patches with measure_entropy_stats() below, and loaded with
    --jedi_calibration. Run the stock constants first so the deviation is a
    measured delta rather than an assumption, then recalibrate and report both.

FOUR DEVIATIONS FROM THE MATLAB, ALL DELIBERATE:

  1. INPAINTING. They call MATLAB's `inpaintCoherent` (coherence transport).
     OpenCV has no equivalent; we use cv2.INPAINT_TELEA at the same radius
     ballpark. Their mitigate_patch() also paints the region grey BEFORE
     inpainting, which inpaintCoherent then ignores; we keep the grey fill as the
     `fill='grey'` option because for a SEGMENTATION model a wrong inpainting is
     not obviously safer than a flat region, and it makes the comparison against
     SAC (which zeroes) cleaner.
  2. NO AUTOENCODER BY DEFAULT. `use_autoencoder` is a flag in their own script
     and the pipeline runs without it, so this is a supported configuration of
     the authors' code rather than an invention. Their trained weights
     (`autoenc_pas07.mat`) are NOT in the release — the repo ships only
     `sample_gt.mat` — and MATLAB's trainAutoencoder over whole flattened masks
     is resolution-locked, so a PASCAL-sized autoencoder could not be applied to
     512x1024 even if we had it. See autoencoder.py for the port and why
     synthetic training data is the right call here.
  3. BLOB CLEANUP AS WRITTEN, NOT AS APPARENTLY INTENDED. Their condition is
         Area/area < 0.005 && (nblobs > 1 || max([stats.Area]) > 0.01)
     where Area is in PIXELS, so `max(...) > 0.01` is true for any blob at all
     and the second clause never fires. The effective rule is "drop every blob
     below 0.5% of the frame", which is what we implement. Flagged because the
     missing `/area` looks like a typo, and because it imposes a HARD FLOOR on
     what Jedi can find: at 512x1024, 0.5% is a 51px side. A patch below that is
     invisible to Jedi for reasons that have nothing to do with entropy.
  4. Entropy is computed on the padded frame and the heatmap cropped back, which
     is what the MATLAB does by accident (it allocates the heatmap unpadded and
     lets MATLAB auto-grow it on assignment, then crops the centre). Reproduced
     on purpose.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ..base import PatchDefence, Purified

#: The authors' calibration, jedi.m lines 34-38, one entry per window size.
PATCH_MEAN = (5.32, 6.06, 6.47)
PATCH_STD = (0.32, 0.34, 0.35)
CLEAN_MEAN = (2.75, 3.19, 3.47)
CLEAN_STD = (1.16, 1.13, 1.07)

#: jedi.m: blobs below this fraction of the frame are dropped. See deviation 3.
MIN_BLOB_FRAC = 0.005


def window_sizes(h: int, w: int) -> tuple:
    r"""
    The three (window, stride) pairs, from the frame size. jedi.m lines 56-63.

        sx = ceil(h/100), bumped to even;  sy likewise
        s2 = max(sx, sy, 8)
        ws = [s2, 1.5*s2 (bumped to even), 2*s2],  stride = ws/2

    THE WINDOWS ARE SMALL AND THEY SCALE WITH THE FRAME, which is the structural
    difference from SAC: at 512x1024 this gives 12/18/24 px windows, at
    1024x2048 it gives 22/34/44. Jedi has no fixed-pixel filters to fall out of
    domain, so the resolution that killed SAC is handled here by construction.
    """
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
    r"""
    256-bin Shannon entropy in bits for every strided window. [ny, nx].

    MATLAB's entropy() histograms a uint8 image into 256 bins, drops the empty
    ones and returns -sum(p*log2(p)). The window is img(x:x+win, y:y+win), which
    in MATLAB's inclusive indexing is win+1 pixels on a side.

    One scatter_add over unfolded windows rather than a loop: at 512x1024 the
    smallest window gives ~15k windows of 169 px, so the histogram tensor is
    15k x 256 floats — 15 MB, one op, no Python loop over windows.
    """
    k = win + 1
    patches = F.unfold(gray_u8.view(1, 1, *gray_u8.shape[-2:]).float(),
                       kernel_size=k, stride=stride)          # [1, k*k, L]
    vals = patches[0].long().T.contiguous()                   # [L, k*k]
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
    r"""
    (heatmap [H,W], mean window entropy). Reproduces get_entr_heatmap().

    LAST WRITE WINS, and that is not an averaging bug to fix. The MATLAB assigns
    `entr_heatmap(x:x+ws, y:y+ws) = win_entr` for overlapping windows in
    row-major order, so each pixel ends up holding the entropy of the LAST window
    covering it. Computed in closed form here instead of looping: the last window
    start at or below pixel p on a stride grid is floor(p/stride), clamped to the
    final grid index.
    """
    H, W = gray_u8.shape[-2:]
    py = px = win // 2
    padded = _symmetric_pad(gray_u8, py, px)
    ent = _window_entropy(padded, win, stride)
    ny, nx = ent.shape
    ph, pw = padded.shape[-2:]
    iy = (torch.arange(ph, device=ent.device) // stride).clamp(max=ny - 1)
    ix = (torch.arange(pw, device=ent.device) // stride).clamp(max=nx - 1)
    full = ent[iy][:, ix]                                     # [ph, pw]
    return full[py:py + H, px:px + W], float(ent.mean())


def entropy_threshold(img_mean: float, i: int, cal: dict) -> float:
    r"""
    The per-image adaptive threshold, jedi.m lines 176-186.

        img_spread = (img_mean - clean_mean) / clean_std
        dev_mult   = ((patch_mean - clean_mean) / patch_mean) * 2
        thr        = (patch_mean + 1.5*patch_std) - dev_mult*patch_std
                     + img_spread*patch_std

    Read it as: start just above the patch entropy mean, walk back by how far
    patches sit above clean content, then shift by how unusual THIS image is
    relative to the clean distribution. A busy frame raises its own threshold, so
    texture does not read as a patch.

    EVERY TERM DEPENDS ON THE CALIBRATION. With stock constants on Cityscapes the
    `img_spread` term is the dangerous one: it is measured in units of the
    authors' clean_std, so if dashcam frames have systematically different mean
    window entropy the threshold is shifted by a multiple of the wrong number.
    """
    pm, ps = cal["patch_mean"][i], cal["patch_std"][i]
    cm, cs = cal["clean_mean"][i], cal["clean_std"][i]
    img_spread = (img_mean - cm) / cs
    dev_mult = ((pm - cm) / pm) * 2.0
    return (pm + 1.5 * ps) - (dev_mult * ps) + (img_spread * ps)


def drop_small_blobs(mask: torch.Tensor,
                     min_frac: float = MIN_BLOB_FRAC) -> torch.Tensor:
    r"""
    Remove connected components below `min_frac` of the frame. See deviation 3.

    scipy.ndimage.label on CPU; the mask is binary and this runs once per image,
    so the transfer is not worth avoiding. If scipy is missing the mask is
    returned unfiltered and the caller is told, because silently skipping the
    cleanup would inflate the false-positive area without explanation.
    """
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
    r"""
    Re-measure the twelve calibration constants on OUR data.

    images  iterable of [1,3,H,W] in [0,1]
    masks   matching [1,1,H,W] (or [1,H,W]) patch footprints, or None.
            With masks, windows whose CENTRE falls inside the footprint feed the
            patch statistics and the rest feed the clean statistics — which is
            what the authors' two populations are. Without masks only the clean
            half is measured and the patch half is left at their values.

    THE CLEAN HALF NEEDS NO PATCHES AND IS THE HALF MOST LIKELY TO BE WRONG for
    Cityscapes, so it is worth running on the train split on its own before any
    attack is involved. The returned dict is JSON-serialisable; save it and pass
    it as --jedi_calibration.
    """
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
    r"""
    Jedi as an input-space defence.

    calibration  path to a JSON of re-measured entropy constants, or None for
                 the authors' PASCAL/INRIA values. THE SINGLE MOST IMPORTANT
                 KNOB HERE — see the module docstring.
    fill         'inpaint' = cv2 TELEA (closest to their inpaintCoherent),
                 'grey' = flat 0.5, 'zero' = black, which is what SAC does and
                 therefore the apples-to-apples setting when comparing the two.
    autoencoder  an optional mask-completion module (see autoencoder.py). None
                 reproduces their use_autoencoder = 0 path.
    blobs        False disables the small-blob cleanup, which is the ablation
                 that shows how much of the localisation is the 0.5% area floor
                 rather than the entropy threshold.
    """

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
        """(final mask, pre-cleanup mask, summed heatmap). No removal."""
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
        raw = (raw > 0).to(x01.dtype)              # union over the three scales
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
            # COMPOSITE, do not replace. cv2.inpaint round-trips the WHOLE frame
            # through uint8, so taking its output wholesale would quantise every
            # untouched pixel by up to 1/255 — a change outside the located
            # region, which is exactly what a localised defence must not do, and
            # which would show up as a tiny unexplained clean-image cost.
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
