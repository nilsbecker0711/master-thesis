"""Patch shape masking — the Bg() term of Tan et al. (ACM MM 2021) Eq 5."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_reference_rgba(path: str, size: int, device, fit: str = "auto"
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load a reference preserving alpha."""
    from PIL import Image
    img = Image.open(path)
    has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
    img = img.convert("RGBA" if has_alpha else "RGB")

    if fit == "auto":
        fit = "bbox" if has_alpha else "crop"

    if fit == "bbox":
        a = np.asarray(img)[..., 3]
        ys, xs = np.nonzero(a > 127)
        if len(ys):
            img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        fit = "pad"

    w, h = img.size
    if fit == "crop" and w != h:
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2,
                        (w - m) // 2 + m, (h - m) // 2 + m))
    elif fit == "pad" and w != h:
        m = max(w, h)
        bg = (0, 0, 0, 0) if has_alpha else (255, 255, 255)
        canvas = Image.new(img.mode, (m, m), bg)
        canvas.paste(img, ((m - w) // 2, (m - h) // 2))
        img = canvas

    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    if has_alpha:
        rgb = torch.from_numpy(arr[..., :3]).permute(2, 0, 1).to(device)
        return rgb, torch.from_numpy(arr[..., 3]).to(device)
    return torch.from_numpy(arr).permute(2, 0, 1).to(device), None


def derive_shape_mask(rgb: torch.Tensor,
                      alpha: Optional[torch.Tensor],
                      method: str = "square",
                      bg_color: str = "white",
                      thresh: float = 0.15,
                      min_frac: float = 0.05) -> torch.Tensor:
    """[S,S] bool silhouette that will actually be pasted."""
    S = rgb.shape[-1]
    if method == "square":
        return torch.ones(S, S, dtype=torch.bool, device=rgb.device)

    if method == "alpha":
        if alpha is None:
            raise ValueError(
                "shape='alpha' requires an RGBA reference (this one has no "
                "alpha channel). Use 'chroma' or 'auto'.")
        mask = alpha > 0.5
    else:
        if method == "auto":
            k = max(2, S // 16)
            corners = torch.stack([rgb[:, :k, :k].reshape(3, -1),
                                   rgb[:, :k, -k:].reshape(3, -1),
                                   rgb[:, -k:, :k].reshape(3, -1),
                                   rgb[:, -k:, -k:].reshape(3, -1)],
                                  dim=-1).reshape(3, -1)
            key = corners.median(dim=1).values
        else:
            key = (torch.ones(3, device=rgb.device) if bg_color == "white"
                   else torch.zeros(3, device=rgb.device))
        mask = (rgb - key.view(3, 1, 1)).pow(2).sum(0).sqrt() > thresh

    mask = largest_component(mask)

    frac = mask.float().mean().item()
    if frac < min_frac:
        raise ValueError(
            f"Silhouette covers only {100*frac:.1f}% of the patch (< "
            f"{100*min_frac:.0f}%). The background key is probably wrong — try "
            f"bg_color='black', method='auto', or a larger thresh.")
    return mask


def largest_component(mask: torch.Tensor) -> torch.Tensor:
    """Keep only the largest 4-connected component of a [S,S] bool mask."""
    m = mask.clone()
    if m.sum() == 0:
        return m
    idx = m.nonzero()
    c = idx.float().mean(0)
    seed = idx[(idx.float() - c).pow(2).sum(1).argmin()]

    comp = torch.zeros_like(m)
    comp[seed[0], seed[1]] = True
    k = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],
                     device=m.device).view(1, 1, 3, 3)
    for _ in range(2 * m.shape[-1]):
        grown = (F.conv2d(comp.float().view(1, 1, *comp.shape), k, padding=1)
                 .view(*comp.shape) > 0) & m
        if grown.sum() == comp.sum():
            break
        comp = grown
    return comp
