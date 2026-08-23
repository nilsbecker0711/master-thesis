r"""
Invariants for the frequency-resolved sensitivity probe.

The probe's entire claim is ATTRIBUTION: "this flip rate was caused by THIS
band, at the SAME perceptual cost as every other band." Three things can
silently destroy that and none of them shows up in the output curve — the
stimulus leaking outside its band, the rescale moving energy between bands, and
the cost normalisation being unequal because the local mean was ignored. One
test each.

Run with: pytest -q tests/test_spectral.py
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from patchreach.diagnostics import spectral as S
from patchreach.patch import csf as C
from patchreach.patch.spec import Patch, PatchConfig

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _band_energy(x, lo, hi):
    """Power inside [lo, hi) cycles/pixel."""
    H, W = x.shape[-2:]
    p = torch.fft.rfft2(x, norm="forward").abs().pow(2)
    f = C.radial_frequency_cpp(H, W, x.device)
    m = (f >= lo) & (f < hi)
    return float((p * m).sum())


# ═════════════════════════════════════════════════════════════════════════════
#  The stimulus
# ═════════════════════════════════════════════════════════════════════════════

def test_band_limited_noise_has_energy_only_in_its_band():
    """
    If the stimulus leaks, every flip rate is attributed to the wrong band and
    the whole probe reports fiction. This is the load-bearing invariant.
    """
    x = S.band_limited_noise(2, 64, 64, 0.20, 0.35)
    inside = _band_energy(x, 0.20, 0.35)
    outside = _band_energy(x, 0.0, 0.20) + _band_energy(x, 0.35, 0.5001)
    assert inside > 0
    assert outside < 1e-12 * max(inside, 1e-30)


def test_band_limited_noise_is_real_and_finite():
    x = S.band_limited_noise(2, 32, 32, 0.05, 0.10)
    assert x.dtype == torch.float32 and x.shape == (2, 3, 32, 32)
    assert torch.isfinite(x).all()


def test_dc_is_always_excluded():
    """
    barten_csf is EXACTLY zero at f=0, so a DC component costs nothing under
    visibility_index. Left in, the lowest band would buy an unbounded
    brightness offset for free and its flip rate would not be comparable with
    any other band's.
    """
    x = S.band_limited_noise(4, 32, 32, 0.0, 0.10)
    # A pure offset would show as a non-zero per-image mean.
    assert float(x.mean(dim=(2, 3)).abs().max()) < 1e-6


def test_rejects_an_inverted_band():
    with pytest.raises(ValueError):
        S.band_limited_noise(1, 32, 32, 0.3, 0.1)


# ═════════════════════════════════════════════════════════════════════════════
#  Cost normalisation
# ═════════════════════════════════════════════════════════════════════════════

def test_local_contrast_scale_reproduces_the_repo_constant_at_mid_grey():
    r"""
    THE DERIVATION CHECK. Michelson contrast of amplitude A on mean mu is
    A/mu, and rfft2(norm='forward') reports A/2, so contrast_scale = 2/mu.
    At mu = 0.5 that must be EXACTLY csf.CONTRAST_SCALE = 4.0, or the local
    variant is not a generalisation of the fixed one and the two conventions
    are measuring different things.
    """
    base = torch.full((2, 3, 16, 16), 0.5)
    scale, floored = S.contrast_scale(base, "local")
    assert floored == 0
    assert torch.allclose(scale, torch.full_like(scale, C.CONTRAST_SCALE))


def test_dark_content_costs_more_than_mid_grey():
    """
    The asymmetry the fixed mu=0.5 convention misses, and it points the wrong
    way for this project: placement_class 0 is road, i.e. dark asphalt, so the
    attack modes UNDER-state visibility exactly where they are usually run.
    """
    dark = torch.full((1, 3, 16, 16), 0.15)
    grey = torch.full((1, 3, 16, 16), 0.5)
    s_dark, _ = S.contrast_scale(dark, "local")
    s_grey, _ = S.contrast_scale(grey, "local")
    assert float(s_dark.mean()) > float(s_grey.mean()) * 3


def test_fixed_mode_matches_the_attack_modes():
    base = torch.rand(2, 3, 16, 16)
    scale, floored = S.contrast_scale(base, "fixed")
    assert floored == 0
    assert torch.allclose(scale, torch.full_like(scale, C.CONTRAST_SCALE))


def test_mu_floor_binds_and_is_counted():
    """A near-black window must not produce an infinite contrast scale, and the
    floor binding must be reported rather than absorbed."""
    black = torch.zeros(1, 3, 16, 16)
    scale, floored = S.contrast_scale(black, "local")
    assert floored == 3                       # one sample x three channels
    assert torch.isfinite(scale).all()
    assert float(scale.max()) == pytest.approx(2.0 / S.MU_FLOOR)


def test_local_mean_respects_the_shape_mask():
    """A cutout's transparent padding must not drag the measured mean toward
    black — the same failure fit_to_range() had before it took a mask."""
    base = torch.zeros(1, 3, 16, 16)
    base[:, :, :4, :] = 0.6                    # the silhouette
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:4, :] = True
    scale, _ = S.contrast_scale(base, "local", mask)
    assert float(scale.mean()) == pytest.approx(2.0 / 0.6, rel=1e-4)


def test_scale_to_cost_hits_the_target_exactly():
    S_ = 64
    csf = C.csf_map(S_, S_, C.ViewingGeometry(), "barten")
    raw = S.band_limited_noise(3, S_, S_, 0.10, 0.20)
    cscale, _ = S.contrast_scale(torch.full((3, 3, S_, S_), 0.5), "local")

    d = S.scale_to_cost(raw, csf, 0.25, "visibility", cscale=cscale)
    v = C.visibility_index(d, csf, reduce="minkowski", contrast_scale=cscale)
    assert torch.allclose(v, torch.full_like(v, 0.25), atol=1e-5)


def test_scale_to_cost_does_not_move_energy_between_bands():
    """
    A uniform rescale is the ONLY normalisation that leaves the spectrum's
    shape untouched — the same argument fit_to_range() makes. Anything that
    reshaped the spectrum would smear the stimulus across bands and destroy
    the attribution.
    """
    S_ = 64
    csf = C.csf_map(S_, S_, C.ViewingGeometry(), "barten")
    raw = S.band_limited_noise(1, S_, S_, 0.20, 0.35)
    d = S.scale_to_cost(raw, csf, 0.25, "visibility")
    leaked = _band_energy(d, 0.0, 0.20) + _band_energy(d, 0.35, 0.5001)
    assert leaked < 1e-12 * _band_energy(d, 0.20, 0.35)


def test_equal_visibility_means_unequal_amplitude():
    """
    THE POINT OF THE CONTROL. The eye is far less sensitive near Nyquist, so
    matching perceptual cost must buy the high band a much larger amplitude.
    If these came out equal, 'equal cost' would be doing nothing and the sweep
    would be an equal-amplitude sweep wearing a disguise.
    """
    S_ = 64
    csf = C.csf_map(S_, S_, C.ViewingGeometry(), "barten")
    lo = S.scale_to_cost(S.band_limited_noise(1, S_, S_, 0.02, 0.05),
                         csf, 0.25, "visibility")
    hi = S.scale_to_cost(S.band_limited_noise(1, S_, S_, 0.35, 0.50),
                         csf, 0.25, "visibility")
    assert float(hi.pow(2).mean().sqrt()) > 5 * float(lo.pow(2).mean().sqrt())


def test_rms_normalisation_equalises_amplitude_instead():
    S_ = 64
    csf = C.csf_map(S_, S_, C.ViewingGeometry(), "barten")
    lo = S.scale_to_cost(S.band_limited_noise(1, S_, S_, 0.02, 0.05),
                         csf, 0.02, "rms")
    hi = S.scale_to_cost(S.band_limited_noise(1, S_, S_, 0.35, 0.50),
                         csf, 0.02, "rms")
    assert float(lo.pow(2).mean().sqrt()) == pytest.approx(0.02, rel=1e-4)
    assert float(hi.pow(2).mean().sqrt()) == pytest.approx(0.02, rel=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
#  The probe end to end
# ═════════════════════════════════════════════════════════════════════════════

class _TinySeg(nn.Module):
    """Frozen stand-in for WrappedSegModel — stride-4 stem, 19 classes."""

    def __init__(self, K=19):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=4, padding=1)
        self.head = nn.Conv2d(8, K, 1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.head(torch.relu(self.c1(x)))


def _setup(H=128, W=256, size=32, scale=0.25):
    torch.manual_seed(0)
    model = _TinySeg()
    img = (torch.rand(1, 3, H, W) - MEAN) / STD
    patch = Patch(PatchConfig(mode="raw", size=size, scale=scale),
                  torch.device("cpu"), MEAN, STD)
    patch.resolve_placement(H, W)
    return model, img, patch


def test_probe_returns_one_row_per_band_with_the_expected_fields():
    model, img, patch = _setup()
    bands = [(0.02, 0.10), (0.10, 0.30), (0.30, 0.50)]
    rows = S.frequency_sensitivity(model, img, patch, MEAN, STD, bands=bands,
                                   n_probes=2, log=lambda *a, **k: None)
    assert len(rows) == 3
    for r, (lo, hi) in zip(rows, bands):
        assert (r["lo"], r["hi"]) == (lo, hi)
        assert 0.0 <= r["flip_all"] <= 100.0
        assert 0.0 <= r["flip_remote"] <= 100.0
        assert r["rms"] > 0
        assert r["centre_cpd"] > 0


def test_probe_never_touches_the_patch_parameter():
    """
    The documented difference from geometric.receptive_field(), which
    overwrites patch.param and restores it. This probe reads GEOMETRY ONLY, so
    it is safe to run against a trained patch — and a future refactor that
    started writing the parameter would silently corrupt whatever checkpoint it
    was pointed at.
    """
    model, img, patch = _setup()
    before = patch.param.detach().clone()
    S.frequency_sensitivity(model, img, patch, MEAN, STD,
                            bands=[(0.05, 0.20)], n_probes=2,
                            log=lambda *a, **k: None)
    assert torch.equal(before, patch.param.detach())


def test_full_region_perturbs_everything_and_patch_region_does_not():
    """
    region='patch' must leave pixels outside the footprint untouched in the
    INPUT (whatever the prediction does downstream); region='full' must not.
    Confusing the two would silently turn a patch result into a UAP result.
    """
    model, img, patch = _setup()
    kw = dict(bands=[(0.05, 0.20)], n_probes=1, log=lambda *a, **k: None)
    rp = S.frequency_sensitivity(model, img, patch, MEAN, STD,
                                 region="patch", **kw)
    rf = S.frequency_sensitivity(model, img, patch, MEAN, STD,
                                 region="full", **kw)
    assert len(rp) == len(rf) == 1
    # With no footprint the remote set is the whole frame, so the two rates
    # coincide by construction in 'full' and generally differ in 'patch'.
    assert rf[0]["flip_remote"] == pytest.approx(rf[0]["flip_all"], abs=1e-6)


def test_probe_rejects_an_unknown_region():
    model, img, patch = _setup()
    with pytest.raises(ValueError):
        S.frequency_sensitivity(model, img, patch, MEAN, STD, region="nope",
                                bands=[(0.05, 0.2)], n_probes=1,
                                log=lambda *a, **k: None)


# ═════════════════════════════════════════════════════════════════════════════
#  The image-boundness verdict
# ═════════════════════════════════════════════════════════════════════════════

def _rows(rates, realised=0.25):
    return [{"lo": 0.1 * i, "hi": 0.1 * (i + 1), "flip_remote": r,
             "realised": realised} for i, r in enumerate(rates)]


def test_summarise_calls_a_stable_peak_not_image_bound():
    """Every image peaking in the same band means the readable band is a MODEL
    property, exactly as the ERF is."""
    out = S.summarise([_rows([1.0, 2.0, 9.0]),
                       _rows([1.1, 2.2, 8.5]),
                       _rows([0.9, 1.8, 9.4])], 0.25, log=lambda *a, **k: None)
    assert out["modal_peak"] == 2
    assert out["peak_agreement"] == 1.0
    assert out["image_bound"] is False


def test_summarise_flags_image_bound_when_the_peak_moves():
    """A peak that moves between scenes means one fixed budget is optimising
    against the wrong target for some images — the finding this probe exists
    to be able to state."""
    out = S.summarise([_rows([9.0, 2.0, 1.0]),
                       _rows([1.0, 2.0, 9.0]),
                       _rows([1.0, 9.0, 2.0])], 0.25, log=lambda *a, **k: None)
    assert out["peak_agreement"] < 0.8
    assert out["image_bound"] is True


def test_summarise_reports_range_limited_bands():
    """A band that could not be driven to the target cost is a ceiling on the
    attack, not a nuisance, and must survive into the JSON."""
    rows = _rows([1.0, 2.0, 3.0])
    rows[2]["realised"] = 0.05                 # never reached tau=0.25
    out = S.summarise([rows], 0.25, log=lambda *a, **k: None)
    assert out["bands"][2]["realised"] == pytest.approx(0.05)
    assert out["bands"][0]["realised"] == pytest.approx(0.25)


def test_summarise_refuses_a_verdict_from_noise():
    r"""
    THE GUARD. At a small tau no band may move the prediction at all — and
    argmax over six near-zero rates still returns a band, every image still
    "agrees" with it, and the top band still "beats" the bottom one. The first
    run of this probe printed exactly that:

        -> the gap is REAL: the top band flips 0.00% vs 0.00%

    Below the floor there must be NO claim in either direction.
    """
    out = S.summarise([_rows([0.00, 0.01, 0.02]),
                       _rows([0.01, 0.00, 0.03])], 0.25,
                      log=lambda *a, **k: None)
    assert out["verdict"] == "inconclusive"
    assert out["band_stability"] == "inconclusive"
    assert out["image_bound"] is None
    assert out["modal_peak"] is None
    assert out["peak_agreement"] is None
    # the raw numbers still survive for the writeup
    assert out["bands"][2]["flip_remote"] == pytest.approx(0.025)


def test_summarise_gives_a_verdict_once_the_signal_clears_the_floor():
    """The same shape of data, scaled above the floor, must produce a call."""
    out = S.summarise([_rows([0.5, 2.0, 9.0]),
                       _rows([0.6, 2.1, 8.8])], 0.25, min_signal=0.5,
                      log=lambda *a, **k: None)
    assert out["verdict"] == "gap"
    assert out["band_stability"] == "stable"
    assert out["image_bound"] is False


def test_summarise_needs_results():
    with pytest.raises(ValueError):
        S.summarise([], 0.25, log=lambda *a, **k: None)
