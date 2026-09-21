"""
scripts/lpips_eval.py without the lpips package.

The metric is swapped for a mean-absolute-difference stand-in: what is under
test is the reconstruction (right crop, right base, residual actually applied)
and the run discovery, not LPIPS itself.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lpips_eval as le
from patchreach.data.cityscapes import norm_tensors
from patchreach.patch.spec import Patch, PatchConfig

DEV = torch.device("cpu")
MEAN, STD = norm_tensors(DEV)
H, W = 64, 128


class MAD(torch.nn.Module):
    """Stand-in metric on the same [-1,1] convention lpips uses."""
    def forward(self, a, b):
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


METRICS = {"alex": MAD()}


def _img(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, H, W, generator=g) * 0.6 + 0.2
    return (le.quantize8(x) - MEAN) / STD


def _csf_patch(size, tau=4.0, seed=0):
    torch.manual_seed(seed)
    cfg = PatchConfig(mode="csf", size=size, scale=0.5, csf_threshold=tau)
    return Patch(cfg, DEV, MEAN, STD)


def test_box_matches_apply_footprint():
    """The crop the metric sees must be the pixels apply() replaced."""
    patch = _csf_patch(32)
    patch.placement = (5, 90)                   # right edge: gets clamped
    _, fp = patch.apply(_img())
    top, left, p = le.footprint_box(patch.placement, H, W, 0.5)
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[top:top + p, left:left + p] = True
    assert torch.equal(fp[0], mask)


def test_residual_is_scored_and_outside_is_untouched():
    patch = _csf_patch(32)
    img = _img()
    row, (c, p) = le.evaluate_image(patch, img, MEAN, STD, METRICS,
                                    {"from_image": True}, 0.25, True)
    assert row["lpips_crop_alex"] > 0
    # size == p, so the zero-residual composite is the clean crop exactly
    assert row["lpips_floor_alex"] == pytest.approx(0.0, abs=1e-6)
    # dilution: the full frame averages the same change over 4x the area
    assert row["lpips_full_alex"] < row["lpips_crop_alex"]
    assert "visibility" in row


def test_floor_is_nonzero_when_the_base_is_resampled():
    """patch_size < p: part of lpips_crop is blur, and floor must say so."""
    patch = _csf_patch(16)
    row, _ = le.evaluate_image(patch, _img(), MEAN, STD, METRICS,
                               {"from_image": True}, 0.25, True)
    assert row["lpips_floor_alex"] > 0


def test_base_only_restores_render():
    patch = _csf_patch(32)
    patch.set_reference_from_image(_img(), MEAN, STD)
    le.composite(patch, _img(), MEAN, STD, True, base_only=True)
    assert "render" not in vars(patch)          # shadow removed again


def test_discover_layouts(tmp_path):
    ck = _csf_patch(32)

    # overfit.py, --seeds 2
    ov = tmp_path / "ov"
    (ov / "seed42").mkdir(parents=True)
    (ov / "seed43").mkdir()
    ck.save(ov / "seed42" / "final.pt")
    ck.save(ov / "seed43" / "best.pt")
    (ov / "config.json").write_text(json.dumps({"image": 420}))
    jobs = le.discover(ov, "auto", [1, 2])
    assert [(j.image, j.ckpt.name) for j in jobs] == [(420, "final.pt"),
                                                     (420, "best.pt")]
    assert len(le.discover(ov, "final", [])) == 1

    # overfit_population.py
    pop = tmp_path / "pop"
    for i in (7, 12):
        ck.save(pop / "patches" / f"img{i:04d}" / "best.pt")
    (pop / "config.json").write_text(json.dumps({"images": [7, 12]}))
    assert [j.image for j in le.discover(pop, "auto", [])] == [7, 12]

    # train.py — shared patch over the requested images
    tr = tmp_path / "tr"
    ck.save(tr / "final.pt")
    (tr / "config.json").write_text(json.dumps({"epochs": 20}))
    assert [j.image for j in le.discover(tr, "auto", [3, 4, 5])] == [3, 4, 5]


def test_saved_checkpoint_round_trips_through_the_pipeline(tmp_path):
    """Load exactly as main() does and get the same number as in memory."""
    patch = _csf_patch(32)
    img = _img(1)
    live, _ = le.evaluate_image(patch, img, MEAN, STD, METRICS,
                                {"from_image": True}, 0.25, True)
    patch.save(tmp_path / "final.pt")
    loaded = Patch.load(tmp_path / "final.pt", DEV, MEAN, STD)
    again, _ = le.evaluate_image(loaded, img, MEAN, STD, METRICS,
                                 {"from_image": True}, 0.25, True)
    assert again["lpips_crop_alex"] == pytest.approx(live["lpips_crop_alex"])


def test_parse_images():
    assert le.parse_images("fixed10", 3) == le.FIXED10[:3]
    assert le.parse_images("first5", 0) == [0, 1, 2, 3, 4]
    assert le.parse_images("4, 9 11", 0) == [4, 9, 11]
