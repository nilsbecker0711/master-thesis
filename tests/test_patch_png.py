"""
--patch_png: evaluating the PIXELS OF A PNG instead of a checkpoint parameter.

The claim the flag rests on is narrow. For mode='raw' the PNG written beside
best.pt IS render()'s output, so re-parameterising those pixels as a direct
parameter gives back the same patch to within 8-bit quantisation. These tests
pin the ways that could stop being true without anyone noticing: the sigmoid
applied a second time (every pixel would land in [0.5, 0.73] and the patch
would still look plausible in a panel), a PNG from another run silently
resized to fit, and a csf checkpoint waved through — there the PNG is
base+residual FUSED for one image, so the pasted square is not the patch the
drop would be attributed to.

What apply() produces matters more than what render() produces, because the
drop is computed from the paste, so the footprint is checked too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patchreach.data.cityscapes import norm_tensors
from patchreach.patch.spec import Patch, PatchConfig

DEV = torch.device("cpu")
MEAN, STD = norm_tensors(DEV)
QUANT = 1.0 / 255.0


def _raw_patch(size=32, pixel_param="sigmoid", seed=0):
    """A raw patch with REAL STRUCTURE. A grey patch would round-trip through
    8 bits exactly and every tolerance below would pass vacuously."""
    torch.manual_seed(seed)
    p = Patch(PatchConfig(mode="raw", size=size, scale=0.25,
                          pixel_param=pixel_param), DEV, MEAN, STD)
    p.param.data.normal_(0.0, 2.0)
    if pixel_param == "direct":
        p.project()
    return p


def _write(patch, path):
    save_image(patch.render().detach().cpu(), path)
    return path


def test_round_trip_is_within_one_8bit_step(tmp_path):
    p = _raw_patch()
    before = p.render().detach().clone()
    delta = p.param_from_png(_write(p, tmp_path / "best_patch.png"))

    assert 0.0 < delta <= QUANT
    after = p.render().detach()
    assert after.shape == before.shape
    # the RETURNED number is the deviation, not an estimate of it: the eval
    # log prints it, so a wrong one would be read as evidence.
    assert float((after - before).abs().max()) == pytest.approx(delta)


def test_the_sigmoid_is_not_applied_twice(tmp_path):
    """Under 'sigmoid' the pixels would go through the sigmoid again and the
    patch would lose all of its contrast into [0.5, 0.73]."""
    p = _raw_patch(pixel_param="sigmoid")
    before = p.render().detach().clone()
    assert float(before.min()) < 0.1          # the dark end exists to lose

    p.param_from_png(_write(p, tmp_path / "p.png"))

    assert p.cfg.pixel_param == "direct"
    after = p.render().detach()
    assert float(after.min()) < 0.1
    assert float((after - before).abs().max()) <= QUANT


def test_direct_parameterisation_round_trips_too(tmp_path):
    p = _raw_patch(pixel_param="direct")
    before = p.render().detach().clone()
    delta = p.param_from_png(_write(p, tmp_path / "p.png"))
    assert delta <= QUANT
    assert float((p.render().detach() - before).abs().max()) <= QUANT


def test_the_pasted_footprint_is_unchanged(tmp_path):
    """The drop is computed from apply(), so this is the guarantee that
    carries the numbers — render() agreeing is only how it gets there."""
    p = _raw_patch()
    img = torch.rand(1, 3, 256, 512)
    p.resolve_placement(256, 512)
    patched_ck, fp_ck = p.apply(img)

    p.param_from_png(_write(p, tmp_path / "p.png"))
    patched_png, fp_png = p.apply(img)

    assert torch.equal(fp_ck, fp_png)
    # apply() pastes in NORMALISED space, so one 8-bit step of pixel error
    # arrives scaled by 1/std. Bilinear resizing is linear and cannot grow it.
    assert float((patched_png - patched_ck).abs().max()) <= (
        QUANT / float(STD.min()) + 1e-6)


def test_geometry_and_resolved_placement_survive(tmp_path):
    """Only the parameter is replaced. The placement in particular is NOT
    recoverable from a PNG, which is why the checkpoint is still required."""
    p = _raw_patch()
    png = _write(p, tmp_path / "p.png")
    p.placement = (7, 9)
    geom = (p.cfg.size, p.cfg.scale, p.cfg.scale_ref, p.cfg.shape, p.cfg.mode)

    p.param_from_png(png)

    assert p.placement == (7, 9)
    assert (p.cfg.size, p.cfg.scale, p.cfg.scale_ref, p.cfg.shape,
            p.cfg.mode) == geom


def test_param_stays_a_leaf_that_requires_grad(tmp_path):
    """report.run's ERF probe and the CAM differentiate through the patch."""
    p = _raw_patch()
    p.param_from_png(_write(p, tmp_path / "p.png"))
    assert p.param.is_leaf and p.param.requires_grad
    p.render().sum().backward()
    assert p.param.grad is not None


def test_a_png_from_another_run_is_refused(tmp_path):
    p = _raw_patch(size=32)
    other = tmp_path / "other.png"
    save_image(torch.rand(3, 64, 64), other)
    with pytest.raises(ValueError, match="different run"):
        p.param_from_png(other)


def test_a_real_csf_patch_is_refused(tmp_path):
    """The dangerous one: its PNG is base+residual fused for one image, and
    the base is not in the checkpoint to subtract back out."""
    p = Patch(PatchConfig(mode="csf", size=32, scale=0.25), DEV, MEAN, STD)
    with pytest.raises(ValueError, match="mode='raw'"):
        p.param_from_png(_write(p, tmp_path / "p.png"))


@pytest.mark.parametrize("mode", ["csf", "universal_csf", "gan",
                                  "raw_ganinit", "lap"])
def test_every_other_mode_is_refused(mode, tmp_path):
    """The gate reads cfg.mode, so a raw patch with its mode relabelled tests
    exactly the gate. Building real gan/lap patches needs BigGAN weights and
    a reference file, which this does not depend on."""
    p = _raw_patch()
    png = _write(p, tmp_path / "p.png")
    p.cfg.mode = mode
    with pytest.raises(ValueError, match="mode='raw'"):
        p.param_from_png(png)


# ── Patch.from_png: the PNG with no checkpoint at all ────────────────────────
#
# The claim here is narrower than it looks. A square raw patch at the centre
# needs three facts to be pasted and the PNG supplies one of them; the other
# two have answers that are not information at all — its own pixel size, and
# the middle. Everything these tests pin is one of those two, plus the refusal
# for the case where the PNG is genuinely missing something.


def test_from_png_takes_its_size_and_pastes_1to1(tmp_path):
    px = torch.rand(3, 256, 256)
    save_image(px, tmp_path / "patch.png")
    p = Patch.from_png(tmp_path / "patch.png", DEV, MEAN, STD, img_h=1024)

    assert p.cfg.size == 256
    assert p.cfg.scale == pytest.approx(0.25)
    # the whole point of the default scale: int(H*scale) == the PNG's side, so
    # the paste is 1:1 and no bilinear resampling happens at this img_h.
    assert p.side(1024, 2048) == 256


def test_from_png_is_raw_direct_and_centred(tmp_path):
    save_image(torch.rand(3, 64, 64), tmp_path / "patch.png")
    p = Patch.from_png(tmp_path / "patch.png", DEV, MEAN, STD, img_h=256)

    assert (p.cfg.mode, p.cfg.pixel_param) == ("raw", "direct")
    assert p.cfg.placement == "center"
    assert p.cfg.scale_ref == "height"
    p.resolve_placement(256, 512)
    assert p.placement == ((256 - 64) // 2, (512 - 64) // 2)


def test_from_png_renders_the_png_exactly(tmp_path):
    """No checkpoint to compare against, so the parameter IS the file: the
    only error in this path is the 8-bit grid the PNG was written on."""
    px = torch.rand(3, 32, 32)
    save_image(px, tmp_path / "patch.png")
    p = Patch.from_png(tmp_path / "patch.png", DEV, MEAN, STD, img_h=128)

    assert float((p.render().detach() - px).abs().max()) <= QUANT
    # and the parameter is the file read back, not a re-render of it
    assert torch.equal(p.param.detach(), p.render().detach())


def test_from_png_explicit_scale_is_honoured(tmp_path):
    save_image(torch.rand(3, 256, 256), tmp_path / "patch.png")
    p = Patch.from_png(tmp_path / "patch.png", DEV, MEAN, STD, img_h=1024,
                       scale=0.5)
    assert p.cfg.scale == pytest.approx(0.5)
    # 256px of parameter pasted over 512px: resampled, and evaluate.py says so
    assert p.side(1024, 2048) == 512


def test_from_png_refuses_a_non_square_png(tmp_path):
    """A two-panel figure and a cutout both land here, and the cutout is the
    dangerous one: its silhouette is not in an RGB render."""
    save_image(torch.rand(3, 540, 1050), tmp_path / "figure.png")
    with pytest.raises(ValueError, match="not square"):
        Patch.from_png(tmp_path / "figure.png", DEV, MEAN, STD, img_h=1024)


def test_from_png_and_a_checkpoint_paste_the_same_patch(tmp_path):
    """The equivalence the no-checkpoint path rests on: for a square raw patch
    at the centre, the PNG alone reproduces what the checkpoint would have
    pasted, to within the 8 bits it was written on."""
    p = _raw_patch(size=64, pixel_param="direct")
    p.cfg.scale = 64 / 256
    png = _write(p, tmp_path / "patch.png")
    img = torch.rand(1, 3, 256, 512)
    p.resolve_placement(256, 512)
    from_ck, fp_ck = p.apply(img)

    q = Patch.from_png(png, DEV, MEAN, STD, img_h=256)
    q.resolve_placement(256, 512)
    from_png, fp_png = q.apply(img)

    assert torch.equal(fp_ck, fp_png)
    assert float((from_png - from_ck).abs().max()) <= (
        QUANT / float(STD.min()) + 1e-6)
