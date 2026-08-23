r"""
Invariants for the CSF-constrained patch family.

Each test pins a property that was WRONG in a first implementation and whose
failure would not have shown up in any loss curve — the budget silently
inverting, the visibility criterion flattering the result, or a clamp
destroying the guarantee it was supposed to enforce.

Run with: pytest -q tests/test_csf.py
"""
import pytest

torch = pytest.importorskip("torch")

from patchreach.patch import csf as C
from patchreach.patch.spec import Patch, PatchConfig

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  The perceptual model itself
# ═════════════════════════════════════════════════════════════════════════════

def test_barten_reproduces_the_paper():
    """
    CSFlow Fig 9 puts peak sensitivity near 700. If this drifts, every
    visibility number in the thesis moves with it.
    """
    assert float(C.barten_csf(torch.tensor([3.0]))) == pytest.approx(700, rel=0.02)


def test_barten_is_zero_at_dc_not_nan():
    """
    The lateral-inhibition term vanishes at f=0, so the bracket diverges. A NaN
    here would propagate into the budget and poison every patch silently.
    """
    v = C.barten_csf(torch.tensor([0.0]))
    assert torch.isfinite(v).all() and float(v) == 0.0


def test_barten_is_bandpass():
    """Sensitivity must rise then fall — a monotone CSF would mean there is no
    high-frequency hiding place at all."""
    f = torch.linspace(0.1, 40, 400)
    v = C.barten_csf(f)
    peak = int(v.argmax())
    assert 0 < peak < len(f) - 1
    assert float(v[peak]) > float(v[-1]) * 10


def test_viewing_geometry_matches_appendix_d():
    g = C.ViewingGeometry(0.0114, 50.0)
    assert g.degrees_per_pixel == pytest.approx(0.013063, rel=1e-3)
    # Nyquist lands BELOW the human cutoff, so no frequency is truly invisible
    assert 30 < g.nyquist_cpd < 45


def test_distance_changes_the_budget_in_the_right_direction():
    """
    Further away, one pixel subtends a SMALLER visual angle, so a given image
    frequency lands at a HIGHER cycles/degree — further down the CSF tail,
    less visible, larger budget. Move closer and the same perturbation becomes
    easier to see. This is why the viewing geometry must be swept rather than
    assumed: it sets how much the attack is allowed.
    """
    near, far = C.ViewingGeometry(0.0114, 25.0), C.ViewingGeometry(0.0114, 100.0)
    assert far.degrees_per_pixel < near.degrees_per_pixel
    assert far.nyquist_cpd > near.nyquist_cpd

    _, b_near = C.patch_budget(64, near, threshold=1.0)
    _, b_far = C.patch_budget(64, far, threshold=1.0)
    assert float(b_far.max()) > float(b_near.max())


def test_sso_oblique_effect_favours_diagonals():
    """The oblique effect is the free asymmetry this family can exploit:
    a diagonal carrier is less visible than an axis-aligned one."""
    f = 8.0
    axis = C.sso_csf(torch.tensor([0.0]), torch.tensor([f]))
    diag = C.sso_csf(torch.tensor([f / 2 ** 0.5]), torch.tensor([f / 2 ** 0.5]))
    assert float(diag) < float(axis)


# ═════════════════════════════════════════════════════════════════════════════
#  The budget
# ═════════════════════════════════════════════════════════════════════════════

def test_budget_grows_toward_nyquist():
    """
    THE PREMISE OF THE WHOLE ATTACK. If high frequency does not buy a larger
    amplitude allowance there is nowhere to hide and the method is pointless.
    """
    S = 128
    csf, budget = C.patch_budget(S, threshold=1.0)
    f = C.radial_frequency_cpp(S, S)
    low = budget[(f >= 0.05) & (f < 0.10)].mean()
    high = budget[f >= 0.35].mean()
    assert float(high) > 10 * float(low)


def test_low_band_is_removed():
    """
    Without this the budget INVERTS: CSF -> 0 at DC, so 1/CSF hands the
    near-DC bins the largest allowance of any frequency and the residual
    becomes a brightness offset — a visible rectangle, whatever the CSF says.
    """
    S = 128
    _, budget = C.patch_budget(S, threshold=1.0, min_cycles=2.0)
    f = C.radial_frequency_cpp(S, S)
    assert float(budget[f < 2.0 / S].max()) == 0.0
    assert float(budget[f > 4.0 / S].max()) > 0.0

    _, naive = C.patch_budget(S, threshold=1.0, min_cycles=0.0)
    assert float(naive[f < 2.0 / S].max()) > 0.0     # the failure, reproduced


def test_residual_energy_concentrates_at_high_frequency():
    """The RAPSD is the direct evidence that camouflage happened."""
    torch.manual_seed(0)
    S = 128
    csf, budget = C.patch_budget(S, threshold=0.25)
    d = C.csf_residual(torch.randn(1, 3, S, S), budget, csf, threshold=0.25)
    f, power = C.rapsd(d, n_bins=6)
    assert float(power[0, -1]) > 100 * float(power[0, 0])


# ═════════════════════════════════════════════════════════════════════════════
#  The reparameterisation
# ═════════════════════════════════════════════════════════════════════════════

def test_bounded_residual_never_exceeds_the_budget():
    torch.manual_seed(0)
    S = 64
    csf, budget = C.patch_budget(S, threshold=1.0)
    for scale in (0.01, 1.0, 1000.0):
        d = C.bounded_residual(torch.randn(2, 3, S, S) * scale, budget)
        spec = torch.fft.rfft2(d, norm="forward").abs()
        # Absolute floor as well. The low band's budget is exactly 0, and the
        # float32 irfft2 -> rfft2 round trip leaves residue there that scales
        # with the input magnitude: ~1e-12 at scale 0.01, ~7e-9 at scale 1000.
        # 1e-7 is still five orders below the smallest meaningful budget.
        assert bool((spec <= budget * (1 + 1e-4) + 1e-7).all()), scale


def test_visibility_is_normalised_to_tau_and_scale_invariant():
    """
    tau must mean something. The reparameterisation is scale-invariant, so the
    raw magnitude must not leak into the result.
    """
    torch.manual_seed(0)
    S = 128
    csf, budget = C.patch_budget(S, threshold=0.25)
    for scale in (0.01, 1.0, 100.0):
        d = C.csf_residual(torch.randn(2, 3, S, S) * scale, budget, csf, 0.25)
        v = C.visibility_index(d, csf)
        assert float(v.mean()) == pytest.approx(0.25, rel=1e-3), scale


def test_gradients_reach_the_raw_parameter():
    S = 64
    csf, budget = C.patch_budget(S, threshold=0.25)
    raw = torch.randn(1, 3, S, S, requires_grad=True)
    C.csf_residual(raw, budget, csf, 0.25).pow(2).sum().backward()
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0


def test_minkowski_is_stricter_than_max():
    """
    max-over-frequency ignores that the eye sums across channels. A residual
    measured 0.235 by max while reaching +/-7.65 in pixel space — nominally
    invisible, actually catastrophic. Defaulting to max would flatter every
    result, so the default must be the stricter reduction.
    """
    torch.manual_seed(0)
    S = 128
    csf, budget = C.patch_budget(S, threshold=1.0)
    d = C.csf_residual(torch.randn(1, 3, S, S), budget, csf, 1.0)
    assert float(C.visibility_index(d, csf, "minkowski")) > \
           float(C.visibility_index(d, csf, "max"))


def test_fit_to_range_preserves_the_spectrum_shape():
    """
    A uniform rescale is the ONLY way to fit the residual into [0,1] without
    destroying the guarantee: it scales every Fourier coefficient equally, so a
    CSF-shaped residual stays CSF-shaped. A clamp would not — it generates
    broadband harmonics right where the CSF peaks.
    """
    torch.manual_seed(0)
    S = 64
    csf, budget = C.patch_budget(S, threshold=0.5)
    d = C.csf_residual(torch.randn(1, 3, S, S), budget, csf, 0.5)
    ref = torch.rand(1, 3, S, S) * 0.6 + 0.2
    fitted = C.fit_to_range(d, ref)

    sd = torch.fft.rfft2(d, norm="forward").abs().reshape(-1)
    sf = torch.fft.rfft2(fitted, norm="forward").abs().reshape(-1)
    ratio = sf[sd > 1e-9] / sd[sd > 1e-9]
    assert float(ratio.std()) < 1e-4          # one global factor, not per-bin
    assert float(C.visibility_index(fitted, csf)) <= 0.5 + 1e-4


def test_clamping_would_have_broken_the_guarantee():
    """Documents WHY fit_to_range exists, by reproducing the failure."""
    torch.manual_seed(0)
    S = 128
    csf, budget = C.patch_budget(S, threshold=1.0)
    d = C.csf_residual(torch.randn(2, 3, S, S), budget, csf, 1.0)
    ref = torch.rand(2, 3, S, S)
    clamped = (ref + d).clamp(0, 1) - ref
    assert float(C.visibility_index(clamped, csf).mean()) > 2.0   # ~5x in practice


# ═════════════════════════════════════════════════════════════════════════════
#  Integration
# ═════════════════════════════════════════════════════════════════════════════

def test_patch_csf_mode_renders_in_range_and_trains():
    img = (torch.rand(1, 3, 256, 512) - MEAN) / STD
    p = Patch(PatchConfig(mode="csf", size=64, scale=0.25),
              torch.device("cpu"), MEAN, STD)
    p.resolve_placement(256, 512)
    p.set_reference_from_image(img, MEAN, STD)

    r = p.render()
    assert r.shape == (3, 64, 64)
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0
    assert set(("visibility", "resid_rms", "resid_absmax")) <= set(p.stats())

    p.apply(img)[0].pow(2).sum().backward()
    assert float(p.param.grad.abs().sum()) > 0


def test_patch_csf_base_is_the_covered_region():
    """set_reference_from_image is what makes this an invisible-residual attack
    rather than a textured square."""
    img01 = torch.zeros(1, 3, 256, 512)
    img01[:, :, 96:160, 224:288] = 1.0            # mark the centre window
    img = (img01 - MEAN) / STD

    p = Patch(PatchConfig(mode="csf", size=64, scale=0.25),
              torch.device("cpu"), MEAN, STD)
    p.resolve_placement(256, 512)                  # centre -> (96, 224)
    base = p.set_reference_from_image(img, MEAN, STD)
    assert float(base.mean()) > 0.99


def test_generator_csf_residual_respects_tau():
    from patchreach.patch.conditional_generator import (
        ConditionalPatchGenerator, GeneratorConfig)
    torch.manual_seed(0)
    g = ConditionalPatchGenerator(GeneratorConfig(
        size=64, base_ch=8, depth=2, residual="csf", csf_threshold=0.25))
    ref = torch.rand(2, 3, 64, 64) * 0.6 + 0.2
    out = g(image=torch.rand(2, 3, 64, 64), reference=ref,
            cam_global=torch.rand(2, 1, 64, 64),
            cam_local=torch.rand(2, 1, 64, 64))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    v = C.realised_visibility(out, ref, g.csf_values)
    assert float(v.max()) <= 0.25 + 1e-3      # a ceiling, never exceeded


def test_generator_csf_head_is_not_zero_initialised():
    """
    The zero-init/baseline-A identity does NOT hold for csf: csf_residual
    normalises to exactly tau, so a zero raw tensor is a 0/0 with an undefined
    gradient. The natural init is random noise at the full budget.
    """
    from patchreach.patch.conditional_generator import (
        ConditionalPatchGenerator, GeneratorConfig)
    csf_g = ConditionalPatchGenerator(GeneratorConfig(size=64, base_ch=8, depth=2,
                                                      residual="csf"))
    logit_g = ConditionalPatchGenerator(GeneratorConfig(size=64, base_ch=8, depth=2,
                                                        residual="logit"))
    assert float(csf_g.head.weight.abs().sum()) > 0
    assert float(logit_g.head.weight.abs().sum()) == 0


def test_existing_patch_modes_are_untouched():
    """csf must be an addition, not a change."""
    for mode in ("raw", "lap"):
        cfg = PatchConfig(mode=mode, size=16, scale=0.25,
                          reference="refs/pothole.png" if mode == "lap" else None)
        if mode == "lap":
            continue                       # needs the file; covered in test_core
        p = Patch(cfg, torch.device("cpu"), MEAN, STD)
        assert torch.allclose(p.render(), torch.full((3, 16, 16), 0.5), atol=1e-6)
        assert p._csf_values is None


def test_shape_mask_excludes_padding_from_the_range_fit():
    """
    THE BUG THIS PINS. load_reference_rgba pads a cutout with TRANSPARENT
    BLACK, so 45% of refs/cover_cut.png is RGB 0 and 95% of the region outside
    the silhouette is. Those pixels have zero headroom for a negative residual
    and are never pasted — but unmasked they drove fit_to_range's quantile to
    0.0000, scaled the residual to EXACTLY ZERO, and turned a five-epoch run
    into a patch that was the unmodified reference image.
    """
    import os
    if not os.path.exists("refs/cover_cut.png"):
        pytest.skip("reference image not present")

    shaped = Patch(PatchConfig(mode="csf", size=128, scale=0.25,
                               reference="refs/cover_cut.png", shape="alpha",
                               csf_threshold=0.25),
                   torch.device("cpu"), MEAN, STD)
    st = shaped.stats()
    assert st["visibility"] == pytest.approx(0.25, rel=0.1)
    assert st["resid_rms"] > 1e-3

    # The mask still matters, but no longer HERE: the transparent padding is
    # itself saturated (exactly 0.0), so the min_headroom exclusion added for
    # the full-region case now catches it too and the unmasked fit survives.
    # The two fixes overlap on this reference; neither is redundant, because
    # padding need not be saturated in general.
    d = torch.randn(1, 3, 128, 128) * 0.01
    ref = shaped.reference.unsqueeze(0)
    assert float(C.fit_to_range(d, ref).abs().max()) > 1e-4
    assert float(C.fit_to_range(d, ref, mask=shaped.shape_mask
                                ).abs().max()) > 1e-4

    # Where the mask is still load-bearing is the VISIBILITY normalisation:
    # tau describes the perturbation that is composited, so spending budget on
    # padding that is discarded would under-shoot the real cost.
    csf, budget = C.patch_budget(128, threshold=0.25)
    raw = torch.randn(1, 3, 128, 128)
    m = C.csf_residual(raw, budget, csf, 0.25, mask=shaped.shape_mask)
    u = C.csf_residual(raw, budget, csf, 0.25)
    assert float(C.visibility_index(C._masked(m, shaped.shape_mask), csf))         == pytest.approx(0.25, rel=1e-3)
    assert not torch.allclose(m, u)


def test_saturated_pixels_do_not_veto_the_whole_residual():
    """
    A pixel already at 0 or 1 has zero headroom in one direction, so its
    headroom ratio is 0 and it drags any quantile down with it. Over a full
    512x1024 frame a 0.1% quantile tolerates 524 such pixels and a Cityscapes
    sky blow-out alone exceeds that: a full-image spectral probe scaled EVERY
    band to exactly zero and reported "range-limited" for all six.

    Excluding them is also physically right — clamping a pixel that is already
    at 1.0 returns 1.0, so it was never going to move and must not veto the
    scale of every pixel that could.
    """
    torch.manual_seed(0)
    img = torch.rand(1, 3, 256, 512) * 0.5 + 0.2
    img[:, :, :40, :] = 1.0                      # blown sky
    img[:, :, -30:, :] = 0.0                     # black hood
    d = torch.randn(1, 3, 256, 512) * 0.01

    fitted = C.fit_to_range(d, img)
    assert float(fitted.pow(2).mean().sqrt()) > 1e-3
    assert float(fitted.abs().max() / d.abs().max()) > 0.5


def test_tau_ladder_is_monotone_where_it_controls():
    """tau must move the needle in its usable range, or it is not a knob."""
    import os
    if not os.path.exists("refs/cover_cut.png"):
        pytest.skip("reference image not present")
    seen = []
    for tau in (0.05, 0.1, 0.25):
        p = Patch(PatchConfig(mode="csf", size=128, scale=0.25,
                              reference="refs/cover_cut.png", shape="alpha",
                              csf_threshold=tau), torch.device("cpu"), MEAN, STD)
        seen.append(p.stats()["visibility"])
        assert seen[-1] == pytest.approx(tau, rel=0.1)
    assert seen == sorted(seen)
