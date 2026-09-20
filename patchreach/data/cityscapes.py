"""Cityscapes loading, label remapping, palette, normalisation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF

# ImageNet normalisation on the [0,1] scale. Arithmetically identical to
# mmseg's img_norm_cfg mean=[123.675,116.28,103.53] std=[58.395,57.12,57.375]
# on the 0-255 scale (0.485*255=123.675, 0.229*255=58.395), so a model trained
# under an mmseg Cityscapes config sees exactly what it expects.
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "light",
         "sign", "veg", "terrain", "sky", "person", "rider", "car", "truck",
         "bus", "train", "moto", "bike"]

PALETTE = torch.tensor([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32],
], dtype=torch.float32) / 255.0

_ID_TO_TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
                22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
                32: 17, 33: 18}


def class_name(c: int) -> str:
    return NAMES[c] if 0 <= c < len(NAMES) else f"cls{c}"


def remap_labels(lbl: torch.Tensor) -> torch.Tensor:
    """Raw Cityscapes ids -> trainIds 0..18, everything else -> 255 (void)."""
    out = torch.full_like(lbl, 255)
    for raw, train in _ID_TO_TRAIN.items():
        out[lbl == raw] = train
    return out


def label_to_colour(label: torch.Tensor) -> torch.Tensor:
    """[H,W] trainIds -> [3,H,W] RGB in [0,1]. Void (255) -> black."""
    label = label.detach().cpu()
    out = torch.zeros(3, *label.shape)
    valid = label < 19
    out[:, valid] = PALETTE[label[valid]].T
    return out


def blend(img: torch.Tensor, seg: torch.Tensor, alpha: float = 0.5):
    return (1.0 - alpha) * img + alpha * seg


def denormalise(x: torch.Tensor, mean_t, std_t) -> torch.Tensor:
    """[1,3,H,W] normalised -> [3,H,W] in [0,1] on CPU, for saving/plotting."""
    return (x[0] * std_t[0] + mean_t[0]).clamp(0, 1).cpu()


def norm_tensors(device):
    m = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1)
    s = torch.tensor(IMG_STD, device=device).view(1, 3, 1, 1)
    return m, s


class CityscapesSeg(torch.utils.data.Dataset):
    r"""
    scale='resize'  DEFAULT, and what every run before 2026-09 used. Squeeze the
                    whole 2048x1024 frame into (img_h, img_w). At 512x1024 the
                    tensor SHAPE matches the mmseg training crop but the pixel
                    DENSITY does not: every object is 2x smaller than anything
                    the model saw, which costs deeplab_18 ~5 mIoU and hits thin
                    classes (pole, fence, rider) hardest.

    scale='crop'    Cut an (img_h, img_w) window out of the NATIVE frame. Pixel
                    density is native — what ratio_range=(0.5,2.0) is centred on
                    and what every published number is measured at. The scene is
                    partial; that is the trade. At 512x1024 this is exactly the
                    three DeepLabs' training condition, at 1024x1024 SegFormer's,
                    at 768x768 SETR's.

    random_crop     Only meaningful with scale='crop'. Use it for the TRAIN split
                    (a centre crop gives the identical window every epoch, which
                    kills augmentation diversity for universal patches) and leave
                    it off for val so runs stay comparable.
    """

    def __init__(self, root, split="train", img_h=512, img_w=1024,
                 scale="resize", random_crop=False):
        if scale not in ("resize", "crop"):
            raise ValueError(f"scale must be 'resize' or 'crop', got {scale!r}")
        self.ds = Cityscapes(root, split=split, mode="fine",
                             target_type="semantic")
        self.hw = (int(img_h), int(img_w))
        self.scale = scale
        self.random_crop = bool(random_crop) and scale == "crop"
        self.post_img = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, lbl = self.ds[idx]
        if self.scale == "resize":
            img = TF.resize(img, list(self.hw))
            lbl = TF.resize(lbl, list(self.hw),
                            interpolation=transforms.InterpolationMode.NEAREST)
        elif self.random_crop:
            # Params drawn ONCE and applied to both. Two independent RandomCrop
            # transforms would draw different offsets for image and label and
            # misalign them silently — invisible in the loss, visible only as a
            # mIoU that will not rise.
            i, j, h, w = transforms.RandomCrop.get_params(img, self.hw)
            img, lbl = TF.crop(img, i, j, h, w), TF.crop(lbl, i, j, h, w)
        else:
            img = TF.center_crop(img, list(self.hw))
            lbl = TF.center_crop(lbl, list(self.hw))
        return self.post_img(img), remap_labels(
            TF.pil_to_tensor(lbl).squeeze(0).long())


def upsample_to(logits: torch.Tensor, hw) -> torch.Tensor:
    if logits.shape[-2:] != tuple(hw):
        logits = F.interpolate(logits, size=tuple(hw), mode="bilinear",
                               align_corners=False)
    return logits
