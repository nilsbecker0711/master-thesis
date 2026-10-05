"""
The Jedi port, against the MATLAB it came from.

github.com/ihsenLab/jedi-CVPR2023, Jedi-main/jedi.m, read 2026-10-02. Each test
names the lines it pins. What a silent regression breaks here is a COMPARISON:
Jedi exists in this repo to be run against a tau ladder, so a threshold or a
window schedule that quietly stopped matching the paper would turn that curve
into a statement about our own arithmetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defences import DEFENCES, build_defence
from defences.detection import mask_detection
from defences.jedi import (CLEAN_MEAN, CLEAN_STD, MIN_BLOB_FRAC, PATCH_MEAN,
                           PATCH_STD, JediDefence, MaskAutoencoder,
                           drop_small_blobs, entropy_heatmap,
                           entropy_threshold, load_calibration,
                           measure_entropy_stats, train_on_synthetic_masks,
                           window_sizes)
from defences.jedi.defence import _symmetric_pad, _window_entropy


# -- the window schedule, jedi.m 56-63 ---------------------------------------

def test_window_schedule_matches_the_matlab_formula():
    """
    sx = ceil(h/100) bumped to even, sy likewise, s2 = max(sx, sy, 8),
    ws = [s2, 1.5*s2 bumped to even, 2*s2], stride = ws/2.

    PINNED AT OUR TWO OPERATING POINTS. These windows SCALE WITH THE FRAME, which
    is the structural reason Jedi is not vulnerable to the resolution that put
    SAC's fixed-pixel segmenter out of domain.
    """
    assert window_sizes(512, 1024) == ((12, 6), (18, 9), (24, 12))
    assert window_sizes(1024, 2048) == ((22, 11), (34, 17), (44, 22))


def test_window_schedule_has_a_floor_of_eight():
    """s2 = max(s1, 8): tiny frames do not get degenerate windows."""
    assert window_sizes(64, 64)[0][0] == 8


def test_every_window_is_even_so_the_stride_and_pad_are_integral():
    for hw in ((512, 1024), (1024, 2048), (256, 512), (100, 137)):
        for win, stride in window_sizes(*hw):
            assert win % 2 == 0, (hw, win)
            assert stride * 2 == win


# -- entropy -----------------------------------------------------------------

def test_constant_image_has_zero_entropy():
    hm, _ = entropy_heatmap(torch.full((1, 1, 64, 64), 100.0), 12, 6)
    assert float(hm.abs().max()) < 1e-9


def test_entropy_is_bounded_by_the_window_sample_count():
    """
    256 bins but only (win+1)^2 samples, so entropy cannot exceed log2(n_px).
    MATLAB's entropy() has the same ceiling; a port that normalised by the bin
    count instead would silently read high everywhere.
    """
    g = torch.Generator().manual_seed(0)
    noise = torch.randint(0, 256, (1, 1, 64, 64), generator=g).float()
    hm, _ = entropy_heatmap(noise, 12, 6)
    assert float(hm.max()) <= float(torch.log2(torch.tensor(169.0))) + 1e-6


def test_heatmap_assignment_is_last_write_wins():
    """
    THE ONE OPTIMISATION THAT COULD BE SUBTLY WRONG. jedi.m assigns
    entr_heatmap(x:x+ws, y:y+ws) = win_entr for OVERLAPPING windows in row-major
    order, so a pixel keeps the entropy of the LAST window covering it -- not an
    average. We compute that in closed form via floor(p/stride) instead of
    looping. This checks the closed form against the literal loop.
    """
    g = torch.Generator().manual_seed(1)
    img = torch.randint(0, 256, (1, 1, 48, 56), generator=g).float()
    win, stride = 8, 4
    H, W = img.shape[-2:]
    got, _ = entropy_heatmap(img, win, stride)

    padded = _symmetric_pad(img, win // 2, win // 2)
    ent = _window_entropy(padded, win, stride)
    ref = torch.zeros(padded.shape[-2], padded.shape[-1], dtype=ent.dtype)
    ny, nx = ent.shape
    for iy in range(ny):                       # row-major, exactly as the .m
        for ix in range(nx):
            y0, x0 = iy * stride, ix * stride
            ref[y0:y0 + win + 1, x0:x0 + win + 1] = ent[iy, ix]
    ref = ref[win // 2:win // 2 + H, win // 2:win // 2 + W]
    # The loop leaves the trailing margin past the last window at 0; the closed
    # form clamps to the last window instead. Compare where the loop wrote.
    wrote = ref != 0
    assert torch.allclose(got[wrote], ref[wrote], atol=1e-9)


def test_symmetric_padding_reflects_including_the_edge():
    """MATLAB padarray 'symmetric', not torch 'reflect' which drops the edge."""
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    assert _symmetric_pad(x, 0, 2)[0, 0, 0].tolist() == [2, 1, 1, 2, 3, 4, 4, 3]


# -- the adaptive threshold, jedi.m 176-186 ----------------------------------

def test_threshold_matches_the_formula_by_hand():
    cal = load_calibration()
    i, img_mean = 0, 3.0
    pm, ps = PATCH_MEAN[i], PATCH_STD[i]
    cm, cs = CLEAN_MEAN[i], CLEAN_STD[i]
    want = ((pm + 1.5 * ps) - ((pm - cm) / pm * 2.0) * ps
            + ((img_mean - cm) / cs) * ps)
    assert entropy_threshold(img_mean, i, cal) == pytest.approx(want)


def test_a_busier_image_raises_its_own_threshold():
    """
    The img_spread term. A high-entropy scene must not read as a patch simply
    for being textured; the threshold tracks the frame.
    """
    cal = load_calibration()
    assert entropy_threshold(5.0, 0, cal) > entropy_threshold(2.0, 0, cal)


def test_calibration_defaults_to_the_authors_constants_and_says_so():
    cal = load_calibration()
    assert tuple(cal["patch_mean"]) == PATCH_MEAN
    assert tuple(cal["clean_std"]) == CLEAN_STD
    assert "Tarchoun" in cal["source"]


def test_calibration_json_round_trips(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"patch_mean": [7, 8, 9], "patch_std": [1, 1, 1],
                             "clean_mean": [2, 2, 2], "clean_std": [1, 1, 1],
                             "source": "cityscapes_train"}))
    cal = load_calibration(str(p))
    assert cal["patch_mean"] == [7.0, 8.0, 9.0]
    assert cal["source"] == "cityscapes_train"


def test_a_malformed_calibration_is_refused_not_half_applied(tmp_path):
    """
    Twelve numbers decide every threshold. A file missing one of them must stop
    the run, not leave a mix of ours and theirs in place.
    """
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"patch_mean": [7, 8], "patch_std": [1, 1, 1],
                             "clean_mean": [2, 2, 2], "clean_std": [1, 1, 1]}))
    with pytest.raises(SystemExit, match="patch_mean"):
        load_calibration(str(p))


# -- blob cleanup, jedi.m 72-80 ---------------------------------------------

def test_blobs_below_half_a_percent_are_dropped():
    m = torch.zeros(1, 1, 200, 200)
    m[..., :4, :4] = 1.0                                   # 16/40000 = 0.04%
    assert float(drop_small_blobs(m).sum()) == 0.0


def test_a_blob_above_the_floor_survives_intact():
    m = torch.zeros(1, 1, 200, 200)
    m[..., :40, :40] = 1.0                                 # 1600/40000 = 4%
    assert torch.equal(drop_small_blobs(m), m)


def test_the_area_floor_is_a_hard_detection_limit():
    """
    DEVIATION 3, AND IT IS A RESULT NOT A BUG. Their condition is
    Area/area < 0.005 && (nblobs > 1 || max([stats.Area]) > 0.01) where Area is
    in PIXELS, so the second clause is true for any blob and never protects
    anything. The effective rule is "drop everything below 0.5% of the frame",
    which at 512x1024 is a 51px side. A patch smaller than that is invisible to
    Jedi for reasons that have nothing to do with entropy, so a patch-scale
    ablation (T19) crossed with Jedi has a floor built into it.
    """
    side_floor = (MIN_BLOB_FRAC * 512 * 1024) ** 0.5
    assert 50 < side_floor < 52
    m = torch.zeros(1, 1, 512, 1024)
    m[..., :40, :40] = 1.0                                 # 40px, under the floor
    assert float(drop_small_blobs(m).sum()) == 0.0


# -- end to end -------------------------------------------------------------

def _scene(h=256, w=512, side=96, noisy=True, seed=0):
    """A smooth gradient, optionally with a high-entropy square pasted in."""
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
    xx = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
    img = (0.3 * yy + 0.4 * xx).clamp(0, 1).expand(1, 3, h, w).contiguous()
    true = torch.zeros(1, 1, h, w)
    if noisy:
        top, left = h // 4, w // 3
        img[:, :, top:top + side, left:left + side] = torch.rand(
            1, 3, side, side, generator=g)
        true[..., top:top + side, left:left + side] = 1.0
    return img, true


def test_jedi_localises_a_high_entropy_patch():
    img, true = _scene()
    out = JediDefence(fill="zero")(img)
    m = mask_detection(out.mask, true)
    assert m["coverage"] > 0.8
    assert m["precision"] > 0.8


def test_jedi_fires_nowhere_on_a_patch_free_scene():
    img, _ = _scene(noisy=False)
    assert float(JediDefence(fill="zero")(img).mask.sum()) == 0.0


def test_removal_touches_only_the_located_region_and_stays_in_gamut():
    img, _ = _scene()
    for fill in ("zero", "grey", "inpaint"):
        out = JediDefence(fill=fill)(img)
        keep = out.mask < 0.5
        assert torch.allclose(out.x01 * keep, img * keep, atol=1e-5), fill
        assert float(out.x01.min()) >= 0.0
        assert float(out.x01.max()) <= 1.0


def test_an_unknown_fill_is_refused():
    with pytest.raises(SystemExit, match="inpaint"):
        JediDefence(fill="diffusion")


def test_the_score_map_is_carried_for_the_empty_mask_diagnostic():
    """Purified.prob must survive, so detection.score_stats() can read it."""
    img, _ = _scene()
    assert JediDefence(fill="zero")(img).prob is not None


def test_jedi_is_registered_and_buildable():
    assert DEFENCES["jedi"] == "jedi"
    assert isinstance(build_defence("jedi", log=lambda *a: None), JediDefence)


# -- calibration measurement ------------------------------------------------

def test_measure_entropy_stats_separates_patch_from_clean_windows():
    """
    The tool that makes Jedi usable here at all. The authors' twelve constants
    come from PASCAL-scale photos with YOLO patches; these are re-measurable on
    our own data. A high-entropy square must land in the patch population and
    read ABOVE the clean one, or the measurement is wired backwards.
    """
    img, true = _scene()
    got = measure_entropy_stats([img], [true])
    assert len(got["patch_mean"]) == 3 and len(got["clean_mean"]) == 3
    assert got["patch_mean"][0] > got["clean_mean"][0]


def test_measuring_without_masks_leaves_the_patch_half_alone():
    """Clean-split calibration needs no patches and must not invent them."""
    img, _ = _scene(noisy=False)
    assert tuple(measure_entropy_stats([img])["patch_mean"]) == PATCH_MEAN


# -- the optional autoencoder ----------------------------------------------

def test_jedi_runs_without_an_autoencoder_by_default():
    """
    The authors' weights are NOT in the release: jedi.m loads autoenc_pas07.mat
    and the repo ships only sample_gt.mat. use_autoencoder is a flag in their own
    script, so running without it is their configuration, not our shortcut.
    """
    assert JediDefence().autoencoder is None


def test_the_autoencoder_is_resolution_locked_but_wraps_any_frame():
    """
    MATLAB trainAutoencoder has one input unit per pixel, so it cannot be applied
    at a resolution it was not trained at. We resize into a fixed grid and back,
    which is a deviation and the only way the topology is usable at 512x1024 --
    one unit per pixel there would be 52M parameters in the first layer.
    """
    ae = MaskAutoencoder(grid=(16, 32), hidden=8)
    for hw in ((64, 128), (512, 1024)):
        assert ae(torch.zeros(1, 1, *hw)).shape == (1, 1, *hw)


def test_a_synthetic_autoencoder_completes_an_eroded_square():
    """
    Trained on squares it completes TO squares -- a SHAPE PRIOR, the same kind
    SAC's L1 square search is. If it lifts coverage, the honest reading is
    "entropy found part of the patch and a square prior finished it", not "Jedi
    found the patch".
    """
    ae = train_on_synthetic_masks((64, 128), sides=(24, 32, 40), grid=(32, 64),
                                  hidden=64, steps=250, batch=24, log=None)
    true = torch.zeros(1, 1, 64, 128)
    true[..., 20:52, 40:72] = 1.0
    g = torch.Generator().manual_seed(3)
    eroded = true * (torch.rand(true.shape, generator=g) > 0.4).float()
    with torch.no_grad():
        done = (ae(eroded) > 0.33).float()
    before = mask_detection(eroded, true)["coverage"]
    after = mask_detection(done, true)["coverage"]
    assert after > before, (before, after)
