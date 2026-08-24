"""
universal_csf — the shared-residual control for mode='csf'.

The tests that matter here are the ones that would let a broken guarantee pass
silently: the per-bin bound, the pooled bound, and the fact that ONE residual
lands on DIFFERENT content. Everything else is plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.data.cityscapes import norm_tensors
from patchreach.diagnostics import universal as univ
from patchreach.patch import csf as C
from patchreach.patch.spec import Patch, PatchConfig

DEV = torch.device("cpu")
MEAN, STD = norm_tensors(DEV)
H, W, S = 128, 256, 32


def make(**kw):
    cfg = PatchConfig(mode="universal_csf", size=S, scale=S / H, **kw)
    p = Patch(cfg, DEV, MEAN, STD)
    p.resolve_placement(H, W)
    return p


def batch(n=4, lo=0.0, hi=0.5):
    """n images of DIFFERENT brightness, normalised."""
    img01 = (torch.rand(n, 3, H, W) * 0.4
             + torch.linspace(lo, hi, n).view(n, 1, 1, 1)).clamp(0, 1)
    return (img01 - MEAN) / STD, img01


# ═════════════════════════════════════════════════════════════════════════════
#  The guarantee
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tau", [0.05, 0.25, 1.0])
def test_no_bin_exceeds_its_budget(tau):
    """
    The projection is the whole point: the bound holds on the PARAMETER, not
    asymptotically and not only on the render.
    """
    p = make(csf_threshold=tau)
    mag = torch.fft.rfft2(p.residual(), norm="forward").abs()
    live = p._csf_budget > 0
    assert float((mag[:, :, live] / p._csf_budget[live]).max()) <= 1.0 + 1e-4
    # bins below min_cycles carry no energy beyond FFT round-trip noise
    assert float(mag[:, :, ~live].max()) < 1e-6


@pytest.mark.parametrize("tau", [0.05, 0.25, 1.0])
def test_pooled_visibility_never_exceeds_tau(tau):
    """
    THE BUG THIS PINS, and it was live for one commit. visibility_index pools
    over all THREE colour channels; normalise_budget_to_tau summed one. The
    envelope came out 3**(1/beta) = 1.44x too generous and a nominal tau of
    0.05 realised 0.0541 -- an 8% overshoot that every assertion passed,
    because the assertion and the normaliser shared the mistake.
    """
    p = make(csf_threshold=tau)
    for scale in (1e-3, 1.0, 1e3):
        # THE CONTRACT: the constraint is maintained by project(), which the
        # training loop calls after every optimiser step. Writing param.data
        # and reading residual() without projecting is the one ordering under
        # which the bound legitimately does not hold -- so the test does what
        # the loop does rather than asserting an invariant nothing maintains.
        p.param.data = torch.randn(3, S, S) * scale
        p.project()
        v = float(C.visibility_index(p.residual(), p._csf_values,
                                     beta=p.cfg.csf_beta,
                                     contrast_scale=p._contrast_scale))
        assert v <= tau + 1e-6, (v, tau)


def test_large_init_saturates_and_pools_to_exactly_tau():
    """
    Comparability with mode='csf' depends on this. csf enforces tau as an
    EQUALITY; here the bound is an inequality, so a universal run would be
    quietly weaker at the same nominal tau unless the init saturates the
    envelope. The default init is large for exactly this reason.
    """
    p = make(csf_threshold=0.25)
    v = float(C.visibility_index(p.residual(), p._csf_values,
                                 beta=p.cfg.csf_beta,
                                 contrast_scale=p._contrast_scale))
    assert v == pytest.approx(0.25, rel=1e-3)
    assert p.stats()["frac_at_bound"] > 0.95


def test_tau_is_monotone():
    p_lo, p_hi = make(csf_threshold=0.1), make(csf_threshold=0.4)
    assert float(p_hi._csf_budget.max()) > float(p_lo._csf_budget.max())
    assert p_hi.stats()["resid_rms"] > p_lo.stats()["resid_rms"]


# ═════════════════════════════════════════════════════════════════════════════
#  Universality
# ═════════════════════════════════════════════════════════════════════════════

def test_one_residual_lands_on_different_content():
    """
    Not a tautology. apply() for every other mode renders one patch and expands
    it, so each image receives identical PIXELS. Here each image must receive
    identical RESIDUAL and therefore different pixels.
    """
    p = make(csf_threshold=0.25)
    imgs, img01 = batch()
    patched, fp = p.apply(imgs)
    back = patched * STD + MEAN
    top, left = p.placement
    win_new = back[:, :, top:top + S, left:left + S]
    win_old = img01[:, :, top:top + S, left:left + S]

    # the pasted content differs across images ...
    assert not torch.allclose(win_new[0], win_new[1], atol=1e-3)
    # ... but the residual applied is the same one
    d = win_new - win_old
    assert torch.allclose(d[1], d[2], atol=1e-5)


def test_nothing_outside_the_footprint_moves():
    p = make(csf_threshold=1.0)
    imgs, img01 = batch()
    patched, fp = p.apply(imgs)
    back = patched * STD + MEAN
    outside = ~fp[0]
    assert float((back - img01)[:, :, outside].abs().max()) < 1e-5


def test_gradient_reaches_the_parameter():
    """F.pad rather than slice assignment, same reason apply() documents."""
    p = make(csf_threshold=0.25)
    imgs, _ = batch()
    patched, _ = p.apply(imgs)
    patched.pow(2).mean().backward()
    assert p.param.grad is not None
    assert float(p.param.grad.abs().sum()) > 0


def test_clipping_is_tracked_not_hidden():
    """
    frac_clipped must rise when the content has no headroom, and stay near zero
    when it does.

    NOT "exactly zero on mid grey": at a large tau the residual's peak exceeds
    0.5 and a handful of pixels clip against 0/1 even from the middle of the
    range. That is the constraint behaving correctly, not a leak — which is
    why the assertion is on the RATIO between the two cases rather than on an
    absolute floor that a big tau would trip.
    """
    p = make(csf_threshold=2.0)

    p.apply((torch.zeros(2, 3, H, W) - MEAN) / STD)
    dark_frac = p._frac_clipped
    assert dark_frac > 0.1, dark_frac

    p.apply((torch.full((2, 3, H, W), 0.5) - MEAN) / STD)
    mid_frac = p._frac_clipped
    assert mid_frac < 0.01, mid_frac
    assert dark_frac > 100 * mid_frac

    # and at a tau the residual comfortably fits, mid grey clips not at all
    q = make(csf_threshold=0.25)
    q.apply((torch.full((2, 3, H, W), 0.5) - MEAN) / STD)
    assert q._frac_clipped == 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  Calibration and reporting
# ═════════════════════════════════════════════════════════════════════════════

def test_calibrated_tau_exceeds_nominal_on_dark_content():
    """
    The measured direction: Cityscapes footprints sit near Y ~ 0.1, far below
    the mu = 0.5 the legacy budget assumes, so the same residual costs MORE
    than its nominal tau claims. If this ever inverts, the sign of the
    calibration has been flipped somewhere.
    """
    p = make(csf_threshold=0.25)
    d = p.residual().detach()
    dark = torch.full((3, 3, S, S), 0.30)          # ~ the measured median code
    t_cal, Y = C.calibrated_visibility(d, dark, p._geometry, "barten",
                                       p.cfg.csf_beta)
    assert float(Y[0]) == pytest.approx(0.0732, abs=5e-3)
    assert float(t_cal[0]) > 0.25


def test_calibrated_tau_varies_with_content_luminance():
    p = make(csf_threshold=0.25)
    d = p.residual().detach()
    base = torch.stack([torch.full((3, S, S), v) for v in (0.1, 0.3, 0.7)])
    t_cal, Y = C.calibrated_visibility(d, base, p._geometry, "barten",
                                       p.cfg.csf_beta)
    assert Y[0] < Y[1] < Y[2]
    assert t_cal[0] > t_cal[2]        # darker content -> more visible


def test_mean_ci_reports_null_not_zero_for_n_equals_one():
    """A zero-width interval reads as 'measured exactly'. One sample is not."""
    assert univ.mean_ci([1.0])["sd"] is None
    assert univ.mean_ci([1.0])["ci_lo"] is None
    r = univ.mean_ci([1.0, 2.0, 3.0, 4.0])
    assert r["n"] == 4 and r["ci_lo"] < r["mean"] < r["ci_hi"]


def test_spectral_allocation_never_exceeds_the_budget():
    p = make(csf_threshold=0.25)
    a = univ.spectral_allocation(p, n_bins=16)
    assert max(a["spend_ratio"]) <= 1.0 + 1e-3
    assert a["live_bins"] > 0


# ═════════════════════════════════════════════════════════════════════════════
#  Guards
# ═════════════════════════════════════════════════════════════════════════════

def test_shaped_patch_is_refused():
    """
    The spectral bound is defined on the full square. A silhouette would make
    the pasted signal different from the one tau was measured on.
    """
    with pytest.raises(ValueError, match="square"):
        PatchConfig(mode="universal_csf", size=S, scale=0.25,
                    shape="alpha", reference="refs/pothole.png").validate()


def test_resampling_is_refused():
    """
    Resampling a residual resamples its SPECTRUM, so the budget its bins were
    projected onto stops describing the pasted signal. Refuse rather than
    report a tau about a different tensor.
    """
    cfg = PatchConfig(mode="universal_csf", size=S, scale=0.5)   # p = 64 != 32
    p = Patch(cfg, DEV, MEAN, STD)
    p.resolve_placement(H, W)
    with pytest.raises(ValueError, match="footprint"):
        p.apply(batch()[0])


def test_existing_modes_are_untouched():
    """universal_csf is an ADDITION. raw must still be flat 0.5 grey."""
    p = Patch(PatchConfig(mode="raw", size=16, scale=0.25), DEV, MEAN, STD)
    assert torch.allclose(p.render(), torch.full((3, 16, 16), 0.5), atol=1e-6)
    assert p._csf_budget is None

    c = Patch(PatchConfig(mode="csf", size=16, scale=0.25), DEV, MEAN, STD)
    assert c._csf_budget is not None
    # csf keeps the LEGACY contrast scale, so its tau means what it always did
    assert not hasattr(c, "_contrast_scale") or \
        c._contrast_scale == C.CONTRAST_SCALE


def test_lref_tightens_the_budget():
    """
    The calibrated budget is measurably tighter than the legacy one on this
    data — ~1.9x at Nyquist. Opt-in, so mode='csf' and every recorded tau are
    unaffected.
    """
    legacy = make(csf_threshold=0.25)
    cal = make(csf_threshold=0.25, csf_lref=0.0971)   # train median at centre
    assert float(cal._csf_budget.max()) < float(legacy._csf_budget.max())


def test_the_optimiser_can_actually_reallocate_the_spectrum():
    """
    THE BUG THIS PINS, and it shipped in the first version of this mode.

    The projection used to live inside residual(), which makes the map radially
    flat above the bound: dL/d|Z| is EXACTLY zero at a saturated bin, so its
    magnitude can never come back down and saturation is an absorbing state.
    Measured from the default init: frac_at_bound stuck at 0.989 across 200
    Adam steps and the per-bin spend ratio moved by 2e-05. The optimiser could
    only rotate phase, and spectral_allocation() -- the metric this mode exists
    to produce -- was pinned at 1.000 by construction rather than by choice.

    Projecting the PARAMETER after the step instead leaves the loss an
    unflattened gradient, so bins can move down as well as up.
    """
    p = make(csf_threshold=0.25)
    opt = torch.optim.Adam([p.param], lr=0.05, betas=(0.5, 0.999))
    B = p._csf_budget
    live = B > 0

    def spend():
        mag = torch.fft.rfft2(p.residual().detach(), norm="forward").abs()
        return (mag / B.clamp(min=1e-12))[:, :, live]

    before = spend().clone()
    target = torch.randn(1, 3, S, S) * 0.01
    for _ in range(50):
        opt.zero_grad()
        ((p.residual() - target) ** 2).mean().backward()
        opt.step()
        p.project()
    after = spend()

    # the allocation MOVED, and moved downward somewhere -- the direction the
    # frozen version could never go
    assert float((after - before).abs().max()) > 0.05
    assert float(after.min()) < 0.95
    assert p.stats()["frac_at_bound"] < 0.99
    # and the bound still holds
    assert float(after.max()) <= 1.0 + 1e-4
