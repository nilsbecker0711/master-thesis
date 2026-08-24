"""
The tau guarantee on the residual that SURVIVES compositing.

Every other csf test bounds the residual we intend to add. These bound what an
observer actually sees, which is clamp(base + delta) - base, and which was
measured at 2.77x tau (worst seed 4.52x) on a real run before csf_enforce
existed. The clamp is the leak: it bends the residual rather than scaling it,
and a bent peak is broadband.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.data.cityscapes import norm_tensors
from patchreach.patch import csf as csf_mod
from patchreach.patch.spec import Patch, PatchConfig

DEV = torch.device("cpu")
MEAN, STD = norm_tensors(DEV)


TAU = 0.25
S = 64


def _csf(size=S):
    return csf_mod.patch_budget(size, csf_mod.ViewingGeometry(), "barten",
                                TAU, 2.0)


def _hostile_base(size=S, seed=0):
    """
    Content that FORCES clipping: half blown to white, half crushed to black.

    A mid-grey base has headroom everywhere and would let any residual through
    untouched, so it cannot exercise the path under test. Cityscapes frames
    contain genuinely saturated regions -- blown sky, black shadow -- which is
    why this matters outside the test suite.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=g) * 0.06
    base[:, :, : size // 2] = 1.0 - base[:, :, : size // 2]
    return base


def test_clipping_really_does_inflate_visibility():
    """The premise. If this fails, csf_enforce is solving a non-problem."""
    csf, budget = _csf()
    g = torch.Generator().manual_seed(1)
    delta = csf_mod.csf_residual(torch.randn(1, 3, S, S, generator=g),
                                 budget, csf, TAU)
    base = _hostile_base()

    intended = float(csf_mod.visibility_index(delta, csf))
    realised = float(csf_mod.visibility_index(
        (base + delta).clamp(0.0, 1.0) - base, csf))

    assert intended == pytest.approx(TAU, rel=1e-3), \
        "csf_residual should normalise to exactly tau before compositing"
    assert realised > intended * 1.5, (
        f"expected the clamp to inflate visibility, got {realised:.4f} "
        f"from {intended:.4f} -- the test base may not be forcing clipping")


def test_fit_to_visibility_restores_the_bound():
    csf, budget = _csf()
    g = torch.Generator().manual_seed(1)
    delta = csf_mod.csf_residual(torch.randn(1, 3, S, S, generator=g),
                                 budget, csf, TAU)
    base = _hostile_base()

    fitted = csf_mod.fit_to_visibility(delta, base, csf, TAU)
    realised = float(csf_mod.visibility_index(
        (base + fitted).clamp(0.0, 1.0) - base, csf))

    assert realised <= TAU + 1e-6, f"bound violated: {realised:.6f} > {TAU}"
    assert realised > 0.0, "scaled to nothing -- the attack would be dead"


def test_a_compliant_residual_is_returned_untouched():
    """No cost when there is nothing to fix: the scale must be exactly 1."""
    csf, budget = _csf()
    g = torch.Generator().manual_seed(2)
    delta = csf_mod.csf_residual(torch.randn(1, 3, S, S, generator=g),
                                 budget, csf, TAU)
    base = torch.full((1, 3, S, S), 0.5)          # headroom everywhere

    fitted = csf_mod.fit_to_visibility(delta, base, csf, TAU)
    assert torch.equal(fitted, delta)


def test_the_scale_is_detached_so_gradients_still_flow():
    """Same contract as gradient clipping: compliant forward, live backward."""
    csf, budget = _csf()
    raw = torch.randn(1, 3, S, S, requires_grad=True)
    base = _hostile_base()

    delta = csf_mod.csf_residual(raw, budget, csf, TAU)
    fitted = csf_mod.fit_to_visibility(delta, base, csf, TAU)
    (base + fitted).clamp(0.0, 1.0).sum().backward()

    assert raw.grad is not None, "graph broken"
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0, "gradient is identically zero"


def test_shared_scale_keeps_one_residual_for_the_batch():
    """
    universal_csf must not acquire a per-image scale by the back door -- that
    is the adaptation the whole mode exists to do without.
    """
    csf, budget = _csf()
    g = torch.Generator().manual_seed(3)
    delta = csf_mod.csf_residual(torch.randn(1, 3, S, S, generator=g),
                                 budget, csf, TAU)
    base = torch.cat([_hostile_base(seed=i) for i in range(4)], dim=0)

    fitted = csf_mod.fit_to_visibility(delta, base, csf, TAU,
                                       shared_scale=True)
    assert fitted.shape[0] == 1, "shared scale must return ONE residual"

    realised = csf_mod.visibility_index(
        (base + fitted).clamp(0.0, 1.0) - base, csf)
    assert realised.shape[0] == 4
    assert float(realised.max()) <= TAU + 1e-6, (
        f"worst image in the batch violates tau: {float(realised.max()):.6f}")


@pytest.mark.parametrize("enforce,must_hold", [("nominal", False),
                                               ("realised", True)])
def test_patch_csf_mode_end_to_end(enforce, must_hold):
    """Through Patch.render(), which is what the attack loop actually calls."""
    cfg = PatchConfig(mode="csf", size=S, scale=0.25, csf_threshold=TAU,
                      csf_enforce=enforce)
    patch = Patch(cfg, DEV, MEAN, STD)
    patch.reference = _hostile_base()[0]
    with torch.no_grad():
        patch.param.normal_(0.0, 3.0)

    rendered = patch.render()
    realised = float(csf_mod.realised_visibility(
        rendered.unsqueeze(0), patch.reference.unsqueeze(0),
        patch._csf_values, cfg.csf_beta))

    if must_hold:
        assert realised <= TAU + 1e-6, f"realised {realised:.6f} > tau {TAU}"
    # The nominal arm is not asserted to violate -- whether it does depends on
    # the content. It is parametrised so a regression that silently enforced
    # everywhere would still show up as this case changing behaviour.
    assert torch.isfinite(rendered).all()
    assert float(rendered.min()) >= 0.0 and float(rendered.max()) <= 1.0


def test_default_is_nominal_so_earlier_taus_keep_their_meaning():
    assert PatchConfig(mode="csf").csf_enforce == "nominal"


def test_an_unknown_enforce_mode_is_refused():
    with pytest.raises(ValueError, match="csf_enforce"):
        PatchConfig(mode="csf", csf_enforce="realized").validate()
