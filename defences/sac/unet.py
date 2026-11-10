"""The SAC patch segmenter's U-Net topology."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv 3x3 => BN => ReLU) x 2"""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """maxpool 2x then DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """bilinear x2 then DoubleConv on the concatenated skip."""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                  align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Emits a ONE-channel LOGIT map."""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True,
                 base_filter=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        f = base_filter
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16 // factor)
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)
        self.outc = OutConv(f, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def load_sac_unet(ckpt: str, base_filter: int = 16, device="cpu",
                  log=print) -> UNet:
    """Build the segmenter and load a released SAC checkpoint, strictly."""
    p = Path(ckpt)
    if not p.is_file():
        raise SystemExit(
            f"\n--sac_ckpt not found: {ckpt}\n\n"
            f"The SAC patch segmenter weights are not part of patch-reach. Fetch\n"
            f"the self-adversarially-trained COCO segmenter (4.4 MB) onto the\n"
            f"cluster and point --sac_ckpt at it:\n\n"
            f"  curl -L -o $MA/checkpoints/sac_coco_at.pth \\n"
            f"    https://raw.githubusercontent.com/joellliu/SegmentAndComplete"
            f"/main/ckpts/coco_at.pth\n")
    net = UNet(3, 1, bilinear=True, base_filter=base_filter)
    sd = torch.load(str(p), map_location="cpu")
    if not any(k.startswith("inc.") for k in sd):
        for pre in ("module.", "unet.", "module.unet."):
            if any(k.startswith(pre) for k in sd):
                sd = {k[len(pre):]: v for k, v in sd.items()
                      if k.startswith(pre)}
                break
    net.load_state_dict(sd, strict=True)
    n = sum(v.numel() for v in net.parameters())
    log(f"[def ] SAC U-Net base_filter={base_filter}  {n/1e6:.2f}M params  "
        f"<- {p.name}")
    return net.eval().to(device)
