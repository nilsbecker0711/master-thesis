r"""
Patch shape masking — the Bg() term of Tan et al. (ACM MM 2021) Eq 5.

    M(p) = p - Grad(p, theta) - Bg(p)
               \___________/   \____/
               edge freezing    background removal   <- this file

lap.py implements the Grad term. Here we paste only the OBJECT SILHOUETTE
rather than its bounding box, so the patch reads as an object rather than as a
rectangle of noise.

COST: silhouettes cover 50-65% of their bounding box, so the adversarial
surface shrinks. Yuan et al. Table 6 shows attack strength scaling with patch
area (100px -> 175px moves Permute mIoU 31.69 -> 55.55), so expect a
proportionate drop. Measure it; do not hide it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_reference_rgba(path: str, size: int, device, fit: str = "auto"
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load a reference preserving alpha. Returns (rgb [3,S,S] in [0,1],
    alpha [S,S] in [0,1] or None).

    ASPECT RATIO — the patch is rendered SQUARE (p x p, p = int(H*scale)), so a
    non-square source has to be reconciled somehow. `fit` decides how:

      auto    (default) With alpha: crop to the alpha BOUNDING BOX, then pad to
              square. The object ends up filling the patch instead of sitting in
              transparent margins — a cutout with wide empty borders otherwise
              wastes most of the adversarial surface (measured: silhouette 0.12
              of the box, i.e. only 5% of parameters optimisable after the edge
              freeze). Without alpha: centre-crop to square.
      crop    Centre-crop to square, then resize. Preserves aspect, uses the
              whole patch, may cut content at the edges.
      pad     Pad to square (transparent if RGBA, white otherwise). Preserves
              aspect AND all content, at the cost of unused patch area.
      stretch Resize straight to SxS. Preserves all content and uses the whole
              patch, but DISTORTS geometry — a round manhole becomes an ellipse.

    Cutout PNGs usually carry usable alpha; JPEGs never do, so those need
    derive_shape_mask(method="chroma"|"auto").
    """
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
        fit = "pad"                      # then square it without distortion

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
    # fit == "stretch" falls through to the resize below

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
    """
    [S,S] bool silhouette that will actually be pasted.

    square : all True — filled bounding box.
    alpha  : alpha > 0.5. Requires an RGBA reference.
    chroma : drop pixels within `thresh` L2 of the named background colour.
    auto   : same, but the key colour is the median of the four corner regions,
             which handles references on arbitrary flat backgrounds.

    After keying, keep only the LARGEST CONNECTED COMPONENT so stray specks
    that happen to differ from the background do not survive as isolated dots.
    """
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
    """
    Keep only the largest 4-connected component of a [S,S] bool mask.

    Iterative dilation-under-constraint rather than scipy.label, to avoid the
    dependency. Converges in O(diameter) steps on patch-sized masks. Seeded at
    the pixel nearest the mask centroid, which for a cutout is reliably inside
    the object rather than in a speck.
    """
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