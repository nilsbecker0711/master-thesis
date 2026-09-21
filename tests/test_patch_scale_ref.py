"""
--patch_scale_ref: what --patch_scale is a fraction of.

The guarantees that matter are the ones a silent regression would break:
'height' must stay BIT-IDENTICAL to the historical int(H*scale) that every
existing checkpoint was trained at, 'area' must hold frame coverage constant
across input shapes, and the pasted footprint must actually be the size the
rule reports — a rule that only fed the log line would pass every other check.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from patchreach.data.cityscapes import norm_tensors
from patchreach.patch import conditional_generator as cg
from patchreach.patch.placement import SCALE_REFS, footprint_side
from patchreach.patch.spec import Patch, PatchConfig

DEV = torch.device("cpu")
MEAN, STD = norm_tensors(DEV)

SHAPES = [(512, 1024), (768, 768), (1024, 1024), (1024, 2048), (128, 256),
          (97, 131)]
SCALES = [0.0625, 0.1, 0.125, 0.1768, 0.25, 0.3333, 0.5]


def test_height_is_bit_identical_to_the_historical_rule():
    for H, W in SHAPES:
        for s in SCALES:
            assert footprint_side(H, W, s, "height") == int(H * s)


def test_height_ignores_width():
    for s in SCALES:
        assert footprint_side(1024, 1024, s) == footprint_side(1024, 2048, s)


def test_area_holds_frame_coverage_across_shapes():
    s = 0.25
    for H, W in SHAPES:
        p = footprint_side(H, W, s, "area")
        # int() floors, so coverage sits at or just under s**2. One pixel of
        # side is the whole tolerance — anything looser would hide a bug.
        assert p == int(s * math.sqrt(H * W))
        assert (p / math.sqrt(H * W)) ** 2 <= s ** 2 + 1e-12
        assert ((p + 1) / math.sqrt(H * W)) ** 2 > s ** 2


def test_area_equals_height_on_square_inputs():
    for n in (64, 512, 768, 1024):
        for s in SCALES:
            assert footprint_side(n, n, s, "area") == footprint_side(n, n, s,
                                                                     "height")


def test_the_worked_values_in_the_help_text():
    assert footprint_side(512, 1024, 0.25, "height") == 128
    assert footprint_side(512, 1024, 0.25, "area") == 181
    assert footprint_side(1024, 2048, 0.25, "height") == 256
    assert footprint_side(1024, 2048, 0.25, "area") == 362


def test_unknown_ref_is_rejected():
    with pytest.raises(ValueError):
        footprint_side(512, 1024, 0.25, "width")
    with pytest.raises(ValueError):
        PatchConfig(scale_ref="width").validate()
    for ref in SCALE_REFS:
        PatchConfig(scale_ref=ref).validate()


@pytest.mark.parametrize("ref", SCALE_REFS)
@pytest.mark.parametrize("H,W", [(128, 256), (96, 96), (64, 256)])
def test_pasted_footprint_is_the_size_the_rule_reports(ref, H, W):
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25, scale_ref=ref),
                  DEV, MEAN, STD)
    p = patch.side(H, W)
    assert p == footprint_side(H, W, 0.25, ref)
    patch.resolve_placement(H, W)
    _, fp = patch.apply(torch.zeros(1, 3, H, W))
    assert int(fp[0].sum()) == p * p


def test_area_changes_the_pasted_size_on_a_wide_input():
    H, W = 128, 256
    sizes = {}
    for ref in SCALE_REFS:
        patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                                  scale_ref=ref), DEV, MEAN, STD)
        patch.resolve_placement(H, W)
        _, fp = patch.apply(torch.zeros(1, 3, H, W))
        sizes[ref] = int(fp[0].sum())
    assert sizes["height"] == 32 * 32
    assert sizes["area"] == 45 * 45          # int(0.25 * sqrt(128*256)) = 45


def test_checkpoints_without_the_field_load_as_height(tmp_path):
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25), DEV, MEAN, STD)
    patch.save(tmp_path / "old.pt")
    ck = torch.load(tmp_path / "old.pt", map_location="cpu")
    ck["config"].pop("scale_ref")                     # a pre-field checkpoint
    torch.save(ck, tmp_path / "old.pt")
    loaded = Patch.load(tmp_path / "old.pt", DEV, MEAN, STD)
    assert loaded.cfg.scale_ref == "height"


def test_area_round_trips_through_a_checkpoint(tmp_path):
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                              scale_ref="area"), DEV, MEAN, STD)
    patch.save(tmp_path / "a.pt")
    loaded = Patch.load(tmp_path / "a.pt", DEV, MEAN, STD)
    assert loaded.cfg.scale_ref == "area"
    assert loaded.side(128, 256) == patch.side(128, 256) == 45


def test_universal_csf_remedy_is_correct_under_area():
    # The refusal suggests a --patch_scale. Under 'area' it must invert the
    # AREA rule — a height-rule suggestion would name a scale that fails again.
    H, W, S = 128, 256, 32
    cfg = PatchConfig(mode="universal_csf", size=S, scale=0.25,
                      scale_ref="area")
    patch = Patch(cfg, DEV, MEAN, STD)
    patch.resolve_placement(H, W)
    with pytest.raises(ValueError) as e:
        patch.apply(torch.zeros(1, 3, H, W))
    msg = str(e.value)
    assert "area rule gives p=45" in msg
    suggested = float(msg.rsplit("--patch_scale ", 1)[1].rstrip(").").strip())
    assert footprint_side(H, W, suggested, "area") in (S, S - 1)


def test_generator_rule_is_unchanged_and_shared():
    for H, W in SHAPES:
        for s in SCALES:
            assert cg.patch_side(H, s) == int(H * s)
            assert cg.patch_side(H, s, W, "area") == footprint_side(H, W, s,
                                                                    "area")
    with pytest.raises(ValueError):
        cg.patch_side(128, 0.25, None, "area")


def test_cli_threads_the_flag_into_the_patch():
    from _common import add_patch_args, build_patch
    p = add_patch_args(argparse.ArgumentParser())
    assert p.parse_args([]).patch_scale_ref == "height"
    a = p.parse_args(["--patch_scale_ref", "area", "--patch_size", "16"])
    patch = build_patch(a, DEV, MEAN, STD)
    assert patch.cfg.scale_ref == "area"
    with pytest.raises(SystemExit):
        p.parse_args(["--patch_scale_ref", "width"])
