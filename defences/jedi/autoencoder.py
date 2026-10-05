r"""
Jedi's optional mask-completion autoencoder. OPTIONAL, AND OFF BY DEFAULT.

THE AUTHORS' WEIGHTS ARE NOT RELEASED. `jedi.m` loads `./autoenc_pas07.mat`;
the repository ships `sample_gt.mat` (3.5 KB of ground-truth masks) and that is
all. `autoencoder_train.m` is the recipe, not a checkpoint:

    autoenc1 = trainAutoencoder(patch_gt, 100, 'MaxEpochs', 100,
                                'L2WeightRegularization', 0.004,
                                'SparsityRegularization', 4,
                                'SparsityProportion', 0.15,
                                'ScaleData', false)

so one hidden layer of 100 units, logistic activations, sparse, trained over a
cell array of BINARY PATCH MASKS. At inference `jedi.m` does
`mask = predict(autoenc, mask) > 0.33`.

THREE CONSEQUENCES, IN ORDER OF HOW MUCH THEY MATTER:

  1. You do not need it. `use_autoencoder` is a flag in their own script and the
     pipeline runs without it, so "Jedi without the autoencoder" is a supported
     configuration of the authors' code rather than something we invented. Run
     that first.
  2. Even if the weights existed they would not transfer. MATLAB's
     trainAutoencoder over whole flattened images is RESOLUTION-LOCKED: the
     input layer has one unit per pixel, so a model trained on PASCAL-sized
     masks cannot be applied to 512x1024. Any usable autoencoder here has to be
     trained at our resolution regardless of what they published.
  3. Which makes training it cheap rather than expensive. The training set is
     binary masks of OUR OWN patch geometry, and we know that exactly — squares
     of a known side at known placements. It is synthetic, free, and needs no
     labelling. train_on_synthetic_masks() below does it in seconds.

     But note what that means for the claim: an autoencoder trained on squares
     completes TO squares. It is a shape prior, the same kind SAC's L1 square
     search is, and it should be reported as one. If it lifts coverage, the
     honest statement is "entropy localised part of the patch and a square prior
     completed it", not "Jedi found the patch".

RESOLUTION. Rather than one input unit per pixel at 512x1024 (52M parameters in
the first layer alone), the mask is resized to `grid` (default 64x128), passed
through, and resized back. That is a deviation from the MATLAB and it is the only
way the architecture is usable at our frame size.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAutoencoder(nn.Module):
    r"""
    One hidden layer, logistic in and out — MATLAB's trainAutoencoder topology.

    Operates on a fixed `grid`; forward() resizes in and out, so callers hand it
    a full-resolution [B,1,H,W] mask and get one back.
    """

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
    r"""
    (corrupted, clean) pairs of square masks. The autoencoder's training data.

    `erode` is the fraction of patch pixels randomly dropped from the INPUT, so
    the model learns to complete a ragged entropy mask back to a solid square —
    which is the job jedi.m gives it. The target is the intact square.
    """
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
    r"""
    Train a MaskAutoencoder on synthetic square masks. Seconds, not hours.

    `sides` should BRACKET the patch side you evaluate, not equal it — a model
    trained only on 128px squares has memorised one size and its completion is
    then indistinguishable from pasting the answer in. Reporting the bracket is
    part of reporting the result.

    MATLAB's SparsityRegularization = 4 DOES NOT TRANSFER as a number. It weighs
    a KL penalty against MATLAB's own loss scaling; dropped naively beside an MSE
    of order 1e-2 it dominates completely and the model collapses to a constant
    (measured: coverage fell from 0.58 to 0.04). `sparsity` is therefore ours,
    defaulting to a value that trains, and the KL form below is the standard one
    rather than the absolute deviation first tried here.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    net = MaskAutoencoder(grid, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for step in range(steps):
        x, y = synthetic_mask_batch(batch, hw, sides, erode, device, g)
        # L2 plus the sparsity pull trainAutoencoder applies to the hidden code
        # (SparsityProportion 0.15, SparsityRegularization 4).
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
