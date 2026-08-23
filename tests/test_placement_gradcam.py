r"""
Invariants for --placement gradcam on the per-image path, and for the paired
comparison that makes it a result rather than two numbers.

The failure this file is mostly written against is SILENT CENTRING: a gradcam
run that quietly falls back to the image centre still produces a plausible
drop_remote, still writes a config.json saying placement=gradcam, and is
indistinguishable from a real one in every artefact the run emits.

Run with: pytest -q tests/test_placement_gradcam.py
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from patchreach.metrics import population as P
from patchreach.patch import optimise, placement as pl
from patchreach.patch.conditional_generator import resolve_batch_placement
from patchreach.patch.spec import Patch, PatchConfig

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _hotspot_map(H, W, top, left, p, peak=1.0):
    """A score map with a single bright p x p block at (top, left)."""
    m = torch.zeros(H, W)
    m[top:top + p, left:left + p] = peak
    return m


# ═════════════════════════════════════════════════════════════════════════════
#  The policy
# ═════════════════════════════════════════════════════════════════════════════

def test_gradcam_finds_the_hotspot():
    H, W, p = 64, 128, 16
    m = _hotspot_map(H, W, 8, 40, p)
    assert pl.resolve("gradcam", H, W, p, score_map=m) == (8, 40)


def test_gradcam_without_a_map_raises_instead_of_centring():
    """
    THE LOAD-BEARING TEST. Falling back to centre would report a gradcam run
    that never ran — a plausible number for a policy that was not applied, and
    invisible in every artefact.
    """
    with pytest.raises(ValueError, match="needs the sensitivity map"):
        pl.resolve("gradcam", 64, 128, 16)


def test_gradcam_on_a_constant_map_centres_explicitly():
    """A fully-suppressed CAM carries no localisation information. argmax over
    a constant map returns index 0 — the top-left corner — which reads as a
    deliberate placement and is not."""
    H, W, p = 64, 128, 16
    flat = torch.full((H, W), 0.5)
    assert pl.resolve("gradcam", H, W, p, score_map=flat) == \
        ((H - p) // 2, (W - p) // 2)


def test_margin_keeps_the_window_off_the_border():
    """
    placement.py records the measured failure: the hottest ridge in a dashcam
    frame is the near-field road boundary along the BOTTOM, so the argmax pins
    the patch flush against the edge, where ~half its receptive field lies
    outside the image and can never be influenced.
    """
    H, W, p = 64, 128, 16
    m = _hotspot_map(H, W, H - p, 0, p)            # hottest at the very corner
    assert pl.resolve("gradcam", H, W, p, score_map=m) == (H - p, 0)

    top, left = pl.resolve("gradcam", H, W, p, score_map=m, margin=8)
    assert top <= H - p - 8 and left >= 8


def test_margin_degrades_rather_than_raising_when_it_cannot_fit():
    H, W, p = 40, 40, 32                            # only 8px of slack
    m = _hotspot_map(H, W, 0, 0, p)
    top, left = pl.resolve("gradcam", H, W, p, score_map=m, margin=999)
    assert 0 <= top <= H - p and 0 <= left <= W - p


def test_gradcam_matches_the_generators_batch_policy_exactly():
    r"""
    ANTI-DRIFT. The per-image ablation and the generator's ablation A-vs-B are
    only comparable if both mean the same thing by 'gradcam'. Two
    implementations of the same argmax would drift, and the drift would read
    as a result: the generator's localisation gain would stop matching
    baseline B's for no stated reason.
    """
    H, W, p = 64, 128, 16
    g = torch.Generator().manual_seed(0)
    for margin in (0, 8):
        cam = torch.rand(1, 1, H, W, generator=g)
        mine = pl.resolve("gradcam", H, W, p, score_map=cam[0, 0],
                          margin=margin)
        theirs = resolve_batch_placement("gradcam", H, W, p, cam=cam,
                                         margin=margin)[0]
        assert mine == theirs


@pytest.mark.parametrize("policy", ["center", "fixed", "semantic"])
def test_existing_policies_are_byte_identical_with_the_new_arguments(policy):
    """gradcam must be an ADDITION. Every pre-existing run has to resolve to
    exactly the same corner it did before."""
    H, W, p = 64, 128, 16
    pred = torch.zeros(H, W, dtype=torch.long)
    pred[30:50, 20:60] = 3
    a = pl.resolve(policy, H, W, p, pred, 3, (0.75, 0.5))
    b = pl.resolve(policy, H, W, p, pred, 3, (0.75, 0.5),
                   score_map=torch.rand(H, W), margin=16)
    assert a == b


# ═════════════════════════════════════════════════════════════════════════════
#  Wiring through Patch and prepare()
# ═════════════════════════════════════════════════════════════════════════════

class _TinySeg(nn.Module):
    """Frozen stand-in exposing `backbone` as SegmentationCAM expects."""

    class _BB(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, stride=4, padding=1)

        def forward(self, x):
            return (torch.relu(self.c1(x)),)

    def __init__(self, K=4):
        super().__init__()
        self.backbone = self._BB()
        self.head = nn.Conv2d(8, K, 1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.head(self.backbone(x)[-1])


def _cam_model():
    torch.manual_seed(0)
    m = _TinySeg()
    w = nn.Module()
    w.model = m
    w.forward = m.forward
    return w


def test_patch_resolve_placement_threads_the_score_map():
    H, W = 64, 128
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                              placement="gradcam"),
                  torch.device("cpu"), MEAN, STD)
    p = int(H * 0.25)
    m = _hotspot_map(H, W, 4, 30, p)
    assert patch.resolve_placement(H, W, None, score_map=m) == (4, 30)


def test_patch_config_carries_the_margin():
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                              placement="gradcam", placement_margin=8),
                  torch.device("cpu"), MEAN, STD)
    H, W = 64, 128
    p = int(H * 0.25)
    m = _hotspot_map(H, W, H - p, 0, p)
    top, left = patch.resolve_placement(H, W, None, score_map=m)
    assert top <= H - p - 8 and left >= 8


def test_prepare_without_a_cam_raises_for_gradcam():
    """The same silent-centring guard, one level up."""
    model, img = _cam_model(), torch.rand(1, 3, 64, 128)
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                              placement="gradcam"),
                  torch.device("cpu"), MEAN, STD)
    with pytest.raises(ValueError, match="needs a SegmentationCAM"):
        optimise.prepare(model, img, patch, 64, 128)


def test_prepare_with_a_cam_places_off_centre_and_returns_clean_logits():
    from patchreach.patch import segmentation_cam
    model = _cam_model()
    img = torch.rand(1, 3, 64, 128)
    label = torch.randint(0, 4, (1, 64, 128))
    cam = segmentation_cam.build(model, "ce", 0, -1, "backbone", "pred")
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                              placement="gradcam"),
                  torch.device("cpu"), MEAN, STD)
    clean = optimise.prepare(model, img, patch, 64, 128, cam=cam, label=label)
    cam.close()
    assert clean.shape == (1, 4, 64, 128)
    assert patch.placement is not None
    H, W, p = 64, 128, 16
    assert 0 <= patch.placement[0] <= H - p
    assert 0 <= patch.placement[1] <= W - p


def test_record_carries_where_the_patch_actually_went():
    """
    Under gradcam the placement varies per image, so a population run cannot be
    interpreted without it: a drop measured at the centre and one measured at a
    near-field hotspot are not the same measurement.
    """
    torch.manual_seed(0)
    model = _TinySeg()
    img = (torch.rand(1, 3, 64, 64) - MEAN) / STD
    label = torch.randint(0, 4, (1, 64, 64))
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25),
                  torch.device("cpu"), MEAN, STD)
    clean = optimise.prepare(model, img, patch, 64, 64)
    rec = optimise.attack_image(model, img, label, patch, steps=4, log_every=2,
                                num_classes=4, clean_logits=clean,
                                verbose=False)
    assert rec["placement"] == list(patch.placement)
    assert rec["placement_policy"] == "center"
    assert rec["placement_dist_from_centre"] == pytest.approx(0.0)
    assert rec["placement_on_border"] is False


# ═════════════════════════════════════════════════════════════════════════════
#  Paired comparison
# ═════════════════════════════════════════════════════════════════════════════

def _recs(vals, start=0):
    return [{"image": i + start, "drop_remote": v} for i, v in enumerate(vals)]


def test_paired_detects_a_consistent_shift_that_unpaired_would_miss():
    r"""
    THE REASON PAIRED EXISTS. Scene variance is huge (0..40) and the effect is
    small (+2 everywhere). The paired difference cancels the scene and resolves
    it; comparing the two means against their own spreads would not.
    """
    scenes = [0.0, 5.0, 12.0, 21.0, 33.0, 40.0, 8.0, 17.0, 26.0, 3.0]
    a = _recs(scenes)
    b = _recs([s + 2.0 for s in scenes])
    out = P.paired_compare(a, b, label_a="center", label_b="gradcam",
                           log=lambda *x, **k: None)
    assert out["difference"]["mean"] == pytest.approx(2.0)
    assert out["conclusive"] is True
    assert out["n_better"] == len(scenes)
    # the unpaired spread is an order of magnitude larger than the effect
    assert P.describe(scenes)["std"] > 5 * out["difference"]["std"] + 1


def test_paired_refuses_a_verdict_when_the_ci_spans_zero():
    a = _recs([10.0, 12.0, 8.0, 11.0, 9.0])
    b = _recs([11.0, 11.0, 9.0, 10.0, 9.5])
    out = P.paired_compare(a, b, log=lambda *x, **k: None)
    assert out["conclusive"] is False


def test_paired_intersects_on_image_index():
    """Both arms must have run the SAME images; anything else is not paired."""
    a = _recs([1.0, 2.0, 3.0], start=0)          # images 0,1,2
    b = _recs([2.0, 3.0, 4.0], start=1)          # images 1,2,3
    out = P.paired_compare(a, b, log=lambda *x, **k: None)
    assert out["n"] == 2
    assert out["images"] == [1, 2]
    assert out["dropped"] == [1, 1]


def test_paired_handles_no_overlap():
    out = P.paired_compare(_recs([1.0], 0), _recs([1.0], 99),
                           log=lambda *x, **k: None)
    assert out["n"] == 0


def test_paired_difference_direction_is_b_minus_a():
    """A sign error here would invert every conclusion in the chapter."""
    out = P.paired_compare(_recs([1.0, 1.0, 1.0]), _recs([5.0, 5.0, 5.0]),
                           log=lambda *x, **k: None)
    assert out["difference"]["mean"] == pytest.approx(4.0)
    assert out["mean_a"] == pytest.approx(1.0)
    assert out["mean_b"] == pytest.approx(5.0)
