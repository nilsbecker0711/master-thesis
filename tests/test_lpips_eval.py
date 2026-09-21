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


@pytest.mark.parametrize("ref", ["height", "area"])
def test_box_matches_apply_footprint_under_both_scale_rules(ref):
    """
    Under --patch_scale_ref area, apply() pastes at int(scale*sqrt(H*W)). A box
    cut by the height rule would score the wrong region with no error — on this
    2:1 input that is 32px against a 45px patch.
    """
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.5, scale_ref=ref),
                  DEV, MEAN, STD)
    patch.resolve_placement(H, W)
    _, fp = patch.apply(_img())
    top, left, p = le.footprint_box(patch.placement, H, W, 0.5,
                                    patch.cfg.scale_ref)
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[top:top + p, left:left + p] = True
    assert torch.equal(fp[0], mask)
    assert p == {"height": 32, "area": 45}[ref]
    if ref == "area":                       # the bug this guards against
        _, _, p_old = le.footprint_box(patch.placement, H, W, 0.5)
        assert p_old != p


def test_residual_is_scored_and_outside_is_untouched():
    patch = _csf_patch(32)
    img = _img()
    row, views = le.evaluate_image(patch, img, MEAN, STD, METRICS,
                                    {"from_image": True}, 0.25, True)
    assert row["lpips_crop_alex"] > 0
    # size == p, so the zero-residual composite is the clean crop exactly
    assert row["lpips_floor_crop_alex"] == pytest.approx(0.0, abs=1e-6)
    assert row["lpips_floor_full_alex"] == pytest.approx(0.0, abs=1e-6)
    # dilution: the full frame averages the same change over 4x the area
    assert row["lpips_full_alex"] < row["lpips_crop_alex"]
    # no blur, so the residual-only composite IS the patched frame
    assert row["lpips_resid_full_alex"] == pytest.approx(
        row["lpips_full_alex"], abs=1e-6)
    assert "visibility" in row


def test_anchors_grow_with_epsilon_and_dilute_when_local():
    row, _ = le.evaluate_image(_csf_patch(32), _img(), MEAN, STD, METRICS,
                               {"from_image": True, "image": 420}, 0.25,
                               True, anchors=(4.0, 8.0))
    g4 = row["anchor_linf4_global_full_alex"]
    g8 = row["anchor_linf8_global_full_alex"]
    assert 0 < g4 < g8
    assert row["anchor_linf8_patch_full_alex"] < g8


def test_floor_is_nonzero_when_the_base_is_resampled():
    """patch_size < p: part of lpips_crop is blur, and floor must say so."""
    patch = _csf_patch(16)
    row, _ = le.evaluate_image(patch, _img(), MEAN, STD, METRICS,
                               {"from_image": True}, 0.25, True)
    assert row["lpips_floor_crop_alex"] > 0
    assert row["lpips_floor_full_alex"] > 0


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


# 16: resampled base, floor > 0. area: on this 2:1 frame the patch is pasted
# at 45 px, not 32, and the box must follow -- with and without a checkpoint.
@pytest.mark.parametrize("size,ref,with_ckpt", [
    (32, "height", True), (16, "height", True),
    (16, "area", True), (16, "area", False)])
def test_png_source_matches_checkpoint_source(tmp_path, size, ref, with_ckpt):
    """
    The PNGs report.py writes and the frames rebuilt from the checkpoint are
    the same image, so every shared column must agree -- including the blur
    control, which the PNG path rebuilds from geometry alone.
    """
    from torchvision.utils import save_image
    from patchreach.data.cityscapes import denormalise

    torch.manual_seed(0)
    patch = Patch(PatchConfig(mode="csf", size=size, scale=0.5,
                              csf_threshold=4.0, scale_ref=ref),
                  DEV, MEAN, STD)
    img = _img(3)
    cfg = {"image": 420, "patch_mode": "csf", "from_image": True,
           "patch_size": size, "patch_scale": 0.5, "patch_scale_ref": ref,
           "placement": "center"}
    ck_row, _ = le.evaluate_image(patch, img, MEAN, STD, METRICS, cfg,
                                  0.25, True)

    run = tmp_path / "run"
    panels = run / "diagnostics" / "panels"
    panels.mkdir(parents=True)
    with torch.no_grad():
        patched, _ = patch.apply(img)
    save_image(denormalise(img, MEAN, STD), panels / "a_clean.png")
    save_image(denormalise(patched, MEAN, STD), panels / "c_patched.png")
    if with_ckpt:
        patch.save(run / "final.pt")
    (run / "config.json").write_text(json.dumps(cfg))

    jobs = le.discover_pngs(run, cfg)
    assert [(j.image, j.label) for j in jobs] == [(420, "run")]
    png_row, views = le.evaluate_png(jobs[0], cfg, METRICS, 0.25, True, (),
                                     DEV)
    assert png_row["box_from"] == ("checkpoint" if with_ckpt else "center")
    assert views["box"][2] == {"height": 32, "area": 45}[ref]

    shared = [k for k in ck_row if k.startswith("lpips_")]
    assert any(k.startswith("lpips_floor_") for k in shared)
    for k in shared:
        assert png_row[k] == pytest.approx(ck_row[k], abs=2e-4), k


class _SpatialMAD(torch.nn.Module):
    """Stand-in for lpips.LPIPS(spatial=True): a [B,1,H,W] map."""
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))    # for .device

    def forward(self, a, b):
        return (a - b).abs().mean(dim=1, keepdim=True)


def _fake_run(tmp_path):
    """A run as overfit.py leaves it: checkpoint, json, diagnostic PNGs."""
    from torchvision.utils import save_image
    from patchreach.data.cityscapes import denormalise

    patch = _csf_patch(16)
    img = _img(5)
    patch.set_reference_from_image(img, MEAN, STD)
    with torch.no_grad():
        patched, _ = patch.apply(img)
    run = tmp_path / "run"
    panels = run / "diagnostics" / "panels"
    panels.mkdir(parents=True)
    save_image(denormalise(img, MEAN, STD), panels / "a_clean.png")
    save_image(denormalise(patched, MEAN, STD), panels / "c_patched.png")
    (panels / "panel.png").write_bytes(b"attack panel")
    patch.save(run / "final.pt")
    (run / "results.json").write_text('{"drop_remote": 12.3}')
    (run / "summary.json").write_text('{"attack": "summary"}')
    (run / "config.json").write_text(json.dumps(
        {"image": 420, "patch_mode": "csf", "from_image": True,
         "patch_size": 16, "patch_scale": 0.5, "placement": "center"}))
    return run


def _snapshot(root: Path):
    return {p.relative_to(root): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def test_evaluation_never_touches_existing_run_files(tmp_path, monkeypatch):
    """
    The whole point: scoring a run must leave every file the attack wrote
    byte-identical, and write nothing outside <run>/lpips_<tag>/.
    """
    monkeypatch.setattr(le, "build_metrics", lambda nets, device,
                        spatial=False: {n: (_SpatialMAD() if spatial
                                            else MAD()) for n in nets})
    run = _fake_run(tmp_path)
    before = _snapshot(tmp_path)

    le.main([str(run), "--tag", "t", "--panels", "1", "--anchors", "8",
             "--device", "cpu"])

    after = _snapshot(tmp_path)
    for path, data in before.items():
        assert after[path] == data, f"{path} was modified"
    new = set(after) - set(before)
    assert new and all(p.parts[:2] == ("run", "lpips_t") for p in new), new
    assert (run / "lpips_t" / "per_image.csv").exists()


@pytest.mark.parametrize("args", [
    ["--out_name", "."], ["--out_name", "diagnostics"],
    ["--out_name", "panels"], ["--out_name", "a/b"],
    ["--out_name", "", "--tag", "x"]])
def test_output_locations_that_could_hit_run_files_are_refused(tmp_path,
                                                               args):
    run = _fake_run(tmp_path)
    with pytest.raises(SystemExit):
        le.check_outputs(le.parse_args([str(run), *args]))


def test_out_csv_never_overwrites_a_foreign_file(tmp_path):
    foreign = tmp_path / "results.csv"
    foreign.write_text("image,miou\n1,50\n")
    with pytest.raises(SystemExit):
        le.check_outputs(le.parse_args(["x", "--out_csv", str(foreign)]))
    with pytest.raises(SystemExit):
        le.check_outputs(le.parse_args(["x", "--out_csv",
                                        str(tmp_path / "r.json")]))
    ours = tmp_path / "lpips.csv"
    ours.write_text("run,lpips_full_alex\n")
    assert le.check_outputs(le.parse_args(["x", "--out_csv", str(ours)]))
