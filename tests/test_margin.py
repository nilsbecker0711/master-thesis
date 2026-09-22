"""
margin loss: the properties the objective exists for.

  1. a pixel flipped by >= kappa contributes zero loss AND zero gradient
  2. an unflipped pixel's gradient is unit size on exactly two logits
  3. void (ref == 255) and the support mask are excluded
  4. build() without the clean prediction fails loudly on use
"""
import pytest
import torch

from patchreach.losses.adversarial import (build, cos_margin_loss,
                                          margin_loss)


def _logits(*cols):
    # cols: one logit vector per pixel -> (1, C, 1, N)
    return torch.tensor(cols, dtype=torch.float32).T[None, :, None, :]


def test_flipped_pixel_has_zero_loss_and_gradient():
    lg = _logits([1.0, 9.0, 0.0]).requires_grad_()   # ref 0, challenger 1 by 8
    ref = torch.tensor([[[0]]])
    loss = margin_loss(lg, ref, kappa=5.0)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.all(lg.grad == 0)


def test_unflipped_pixel_unit_gradient_on_two_logits():
    lg = _logits([9.0, 2.0, 1.0]).requires_grad_()   # ref 0 leads by 7
    ref = torch.tensor([[[0]]])
    loss = margin_loss(lg, ref, kappa=5.0)
    loss.backward()
    assert loss.item() == pytest.approx(12.0)
    g = lg.grad[0, :, 0, 0]
    assert g.tolist() == [1.0, -1.0, 0.0]


def test_worked_example_mean_over_pixels():
    # A road 8 vs 3, B road 4 vs 6, C sky 1 vs 9, D building 9 vs 2
    lg = _logits([8, 3, 0], [4, 6, 0], [9, 1, 0], [0, 9, 2])
    ref = torch.tensor([[[0, 0, 1, 1]]])
    assert margin_loss(lg, ref, kappa=5.0).item() == pytest.approx(
        (10 + 3 + 0 + 12) / 4)


def test_void_and_support_excluded():
    lg = _logits([9.0, 0.0], [9.0, 0.0], [0.0, 9.0])
    ref = torch.tensor([[[0, 255, 1]]])
    support = torch.tensor([[[True, True, False]]])
    # pixel 1 is void, pixel 2 is outside support -> only pixel 0 scored
    assert margin_loss(lg, ref, support, kappa=5.0).item() == pytest.approx(14)


def test_build_binds_clean_prediction_and_ignores_labels():
    lg = _logits([9.0, 0.0])
    ref = torch.tensor([[[0]]])
    gt = torch.tensor([[[1]]])                        # deliberately different
    f = build("margin", margin_ref=ref, margin_kappa=5.0)
    assert f(lg, gt, None, None).item() == pytest.approx(14)


def test_build_without_ref_raises_on_use():
    f = build("margin")
    with pytest.raises(ValueError, match="clean prediction"):
        f(_logits([1.0, 0.0]), torch.tensor([[[0]]]), None, None)


@pytest.mark.parametrize("loss_fn", ["margin", "cos_margin"])
def test_attack_image_runs_margin_end_to_end(loss_fn):
    """The kwarg is wired through attack_image and the patch moves."""
    from torch import nn
    from patchreach.patch import optimise
    from patchreach.patch.spec import Patch, PatchConfig

    class _TinySeg(nn.Module):
        def __init__(self, K=19):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, stride=4, padding=1)
            self.head = nn.Conv2d(8, K, 1)
            for p in self.parameters():
                p.requires_grad_(False)

        def forward(self, x):
            return self.head(torch.relu(self.c1(x)))

    torch.manual_seed(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    H, W = 64, 128
    img = (torch.rand(1, 3, H, W) - mean) / std
    label = torch.randint(0, 19, (1, H, W))
    label[:, :4] = 255
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25),
                  torch.device("cpu"), mean, std)
    patch.resolve_placement(H, W)
    start = patch.param.detach().clone()

    res = optimise.attack_image(
        _TinySeg(), img, label, patch, loss_fn=loss_fn, margin_kappa=2.0,
        steps=5, lr=0.05, log_every=5, out_dir=None, save_best=False,
        verbose=False, log=lambda *a, **k: None)

    assert torch.isfinite(torch.tensor(res["last_loss"]))
    assert not torch.allclose(patch.param.detach(), start)


# ═════════════════════════════════════════════════════════════════════════════
#  cos_margin — the cosine weight on the same hinge
# ═════════════════════════════════════════════════════════════════════════════

def test_cos_margin_leaves_confident_holdouts_at_full_weight():
    """
    The property plain CosPGD lacks. p_ref ~ 1 -> w ~ 1, so the hinge is
    essentially unweighted where the collapse leaves its holdouts.
    """
    lg = _logits([12.0, 0.0, 0.0])
    ref = torch.tensor([[[0]]])
    plain = margin_loss(lg, ref, kappa=5.0).item()
    assert cos_margin_loss(lg, ref, kappa=5.0).item() == pytest.approx(
        plain, rel=1e-3)


def test_cos_margin_releases_a_just_flipped_pixel_early():
    """Between the boundary and kappa the cosine is the whole difference."""
    lg = _logits([0.0, 1.0, 0.0])                  # flipped by 1, kappa 5
    ref = torch.tensor([[[0]]])
    plain = margin_loss(lg, ref, kappa=5.0).item()
    cos = cos_margin_loss(lg, ref, kappa=5.0).item()
    assert 0.0 < cos < plain


def test_cos_margin_is_still_exactly_zero_past_kappa():
    lg = _logits([0.0, 9.0, 0.0]).requires_grad_()
    ref = torch.tensor([[[0]]])
    loss = cos_margin_loss(lg, ref, kappa=5.0)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.all(lg.grad == 0)


def test_cos_margin_weight_is_detached():
    """Gradient reaches z_ref and the challenger ONLY, as for plain margin."""
    lg = _logits([9.0, 2.0, 1.0]).requires_grad_()
    ref = torch.tensor([[[0]]])
    cos_margin_loss(lg, ref, kappa=5.0).backward()
    g = lg.grad[0, :, 0, 0]
    assert g[2].item() == 0.0, "a live cosine would leak gradient to class 2"
    assert g[0] > 0 and g[1] < 0


def test_cos_margin_excludes_void_and_support():
    lg = _logits([9.0, 0.0], [9.0, 0.0], [0.0, 9.0])
    ref = torch.tensor([[[0, 255, 1]]])
    support = torch.tensor([[[True, True, False]]])
    one = cos_margin_loss(lg[:, :, :, :1], ref[:, :, :1], kappa=5.0).item()
    assert cos_margin_loss(lg, ref, support, kappa=5.0).item() == pytest.approx(
        one)


def test_build_dispatches_cos_margin():
    lg = _logits([9.0, 0.0])
    ref = torch.tensor([[[0]]])
    f = build("cos_margin", margin_ref=ref, margin_kappa=5.0)
    assert f(lg, torch.tensor([[[1]]]), None, None).item() == pytest.approx(
        cos_margin_loss(lg, ref, kappa=5.0).item())
