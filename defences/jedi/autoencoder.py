"""Jedi's optional mask-completion autoencoder."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAutoencoder(nn.Module):
    """One hidden layer, logistic in and out — MATLAB's trainAutoencoder topology."""

    def __init__(self, grid: Tuple[int, int] = (64, 128), hidden: int = 100):
        super().__init__()
        self.grid = tuple(grid)
        n = self.grid[0] * self.grid[1]
        self.enc = nn.Linear(n, hidden)
        self.dec = nn.Linear(hidden, n)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        H, W = mask.shape[-2:]
        x = F.interpolate(mask.float(), size=self.grid, mode="area")
        h = torch.sigmoid(self.enc(x.flatten(1)))
        y = torch.sigmoid(self.dec(h)).view(-1, 1, *self.grid)
        return F.interpolate(y, size=(H, W), mode="bilinear",
                             align_corners=False)

    def encode(self, mask: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(mask.float(), size=self.grid, mode="area")
        return torch.sigmoid(self.enc(x.flatten(1)))


def synthetic_mask_batch(n: int, hw: Tuple[int, int], sides: Sequence[int],
                         erode: float = 0.3, device="cpu", generator=None):
    """(corrupted, clean) pairs of square masks."""
    H, W = hw
    clean = torch.zeros(n, 1, H, W, device=device)
    g = generator
    for i in range(n):
        s = int(sides[torch.randint(len(sides), (1,), generator=g).item()])
        s = min(s, H, W)
        top = int(torch.randint(0, max(1, H - s + 1), (1,), generator=g).item())
        left = int(torch.randint(0, max(1, W - s + 1), (1,), generator=g).item())
        clean[i, 0, top:top + s, left:left + s] = 1.0
    keep = (torch.rand(clean.shape, device=device, generator=g) > erode).float()
    return clean * keep, clean


def train_on_synthetic_masks(hw, sides, grid=(64, 128), hidden=100,
                             steps: int = 1500, batch: int = 16,
                             lr: float = 3e-3, erode: float = 0.3,
                             sparsity: float = 0.05, target: float = 0.15,
                             seed: int = 42, device="cpu", log=print):
    """Train a MaskAutoencoder on synthetic square masks."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    net = MaskAutoencoder(grid, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for step in range(steps):
        x, y = synthetic_mask_batch(batch, hw, sides, erode, device, g)
        pred = net(x)
        h = net.encode(x).mean(0).clamp(1e-6, 1 - 1e-6)
        rho = torch.full_like(h, target)
        kl = (rho * (rho / h).log() + (1 - rho) * ((1 - rho) / (1 - h)).log())
        loss = F.mse_loss(pred, y) + sparsity * kl.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and (step + 1) % max(1, steps // 4) == 0:
            log(f"[ae  ] step {step+1}/{steps}  loss {float(loss):.5f}")
    return net.eval()
