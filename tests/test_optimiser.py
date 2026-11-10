"""The optimiser ablation: step rule (adam | sign) x parameterisation (sigmoid | direct)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from patchreach import optim as optim_mod
from patchreach.patch import optimise
from patchreach.patch.spec import Patch, PatchConfig

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def test_sign_step_displaces_every_coordinate_by_exactly_lr():
    """THE MATCHING CLAIM. optim.py asserts adam and sign are comparable at equal lr because both move a coordinate by ~lr."""
    p = torch.zeros(4, requires_grad=True)
    p.grad = torch.tensor([1e-9, -1e-9, 1e9, -1e9])
    optim_mod.SignSGD([p], lr=0.1).step()
    assert torch.allclose(p.detach(), torch.tensor([-0.1, 0.1, -0.1, 0.1]))


def test_sign_step_leaves_a_dead_coordinate_where_it_is():
    """sign(0) = 0."""
    p = torch.zeros(2, requires_grad=True)
    p.grad = torch.tensor([0.0, 2.0])
    optim_mod.SignSGD([p], lr=0.5).step()
    assert p.detach()[0].item() == 0.0
    assert p.detach()[1].item() == pytest.approx(-0.5)


def test_sign_carries_no_state_between_steps():
    """No momentum, deliberately — see SignSGD's docstring."""
    p = torch.zeros(1, requires_grad=True)
    opt = optim_mod.SignSGD([p], lr=0.25)
    p.grad = torch.tensor([1.0]); opt.step()
    p.grad = torch.tensor([-1.0]); opt.step()
    assert p.detach().item() == pytest.approx(0.0, abs=1e-12)


def test_lr_schedulers_drive_signsgd():
    """THE REASON SignSGD SUBCLASSES Optimizer."""
    p = torch.zeros(1, requires_grad=True)
    opt = optim_mod.SignSGD([p], lr=1.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] < 1.0


def test_build_rejects_an_unknown_rule_rather_than_defaulting():
    with pytest.raises(ValueError, match="optimiser must be"):
        optim_mod.build("pgd", [torch.zeros(1, requires_grad=True)], 0.1)


def test_build_adam_honours_the_betas_it_is_given():
    """Both loops route through build() but disagree on betas; the builder must not impose one of them."""
    p = [torch.zeros(1, requires_grad=True)]
    assert optim_mod.build("adam", p, 0.1,
                           betas=(0.5, 0.999)).param_groups[0]["betas"] == (0.5, 0.999)


def _patch(pixel_param="sigmoid", size=16):
    return Patch(PatchConfig(mode="raw", size=size, scale=0.25,
                             pixel_param=pixel_param),
                 torch.device("cpu"), MEAN, STD)


def test_both_parameterisations_start_from_the_same_grey_image():
    """Or the ablation compares two runs that began at different patches."""
    assert torch.allclose(_patch("sigmoid").render(),
                          _patch("direct").render(), atol=1e-6)


def test_direct_projects_back_into_the_pixel_box_after_a_step():
    patch = _patch("direct")
    patch.param.data.fill_(2.5)
    patch.project()
    assert patch.param.data.max().item() == pytest.approx(1.0)
    patch.param.data.fill_(-2.5)
    patch.project()
    assert patch.param.data.min().item() == pytest.approx(0.0)


def test_direct_does_not_clamp_inside_the_forward_pass():
    """THE WHOLE CONTENT OF 'direct'."""
    patch = _patch("direct")
    patch.param.data.fill_(3.0)
    out = patch.render()
    assert out.max().item() == pytest.approx(3.0)
    out.sum().backward()
    assert patch.param.grad.abs().min().item() > 0


def test_sigmoid_render_is_untouched_by_the_new_branch():
    """The default path must be what it was."""
    patch = _patch("sigmoid")
    patch.param.data.normal_()
    assert torch.allclose(patch.render(), torch.sigmoid(patch.param))


def test_frac_at_clip_counts_pixels_on_the_bound_under_direct():
    """The measurement the ablation exists to produce."""
    patch = _patch("direct")
    patch.param.data.fill_(0.5)
    patch.param.data[0, 0, :] = 1.0
    st = patch.stats()
    n = patch.param.numel()
    assert st["frac_at_clip"] == pytest.approx(patch.cfg.size / n)
    assert "param_absmax" in st and "logit_absmax" not in st


def test_direct_is_refused_on_every_mode_that_has_no_pixel_parameter():
    for mode in ("gan", "csf", "universal_csf", "lap"):
        with pytest.raises(ValueError, match="pixel_param='direct'"):
            PatchConfig(mode=mode, pixel_param="direct",
                        reference="x.png").validate()


def test_checkpoints_record_which_parameterisation_wrote_them(tmp_path):
    _patch("direct").save(tmp_path / "d.pt")
    _patch("sigmoid").save(tmp_path / "s.pt")
    assert torch.load(tmp_path / "d.pt")["parameterisation"] == "pixel"
    assert torch.load(tmp_path / "s.pt")["parameterisation"] == "sigmoid"


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


@pytest.mark.parametrize("optimiser", ["adam", "sign"])
@pytest.mark.parametrize("pixel_param", ["sigmoid", "direct"])
def test_every_arm_of_the_ablation_runs_and_moves_the_patch(optimiser,
                                                            pixel_param):
    """The cheap insurance."""
    torch.manual_seed(0)
    H, W = 64, 128
    model = _TinySeg()
    img = (torch.rand(1, 3, H, W) - MEAN) / STD
    label = torch.randint(0, 19, (1, H, W))
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                              pixel_param=pixel_param),
                  torch.device("cpu"), MEAN, STD)
    patch.resolve_placement(H, W)
    start = patch.param.detach().clone()

    res = optimise.attack_image(
        model, img, label, patch, loss_fn="ce", steps=5, lr=0.05,
        optimiser=optimiser, log_every=5, out_dir=None, save_best=False,
        verbose=False, log=lambda *a, **k: None)

    assert res["optimiser"] == optimiser
    assert res["pixel_param"] == pixel_param
    assert torch.isfinite(torch.tensor(res["last_loss"]))
    assert not torch.allclose(patch.param.detach(), start), \
        "the patch never moved — the graph is broken on this arm"


def test_the_sign_arm_takes_lr_sized_steps_through_attack_image():
    """End-to-end confirmation of the matching claim: after ONE step with no schedule, every coordinate that had a gradient has moved by exactly lr."""
    torch.manual_seed(0)
    H, W = 64, 128
    model = _TinySeg()
    img = (torch.rand(1, 3, H, W) - MEAN) / STD
    label = torch.randint(0, 19, (1, H, W))
    patch = Patch(PatchConfig(mode="raw", size=16, scale=0.25,
                              pixel_param="direct"),
                  torch.device("cpu"), MEAN, STD)
    patch.resolve_placement(H, W)
    start = patch.param.detach().clone()

    optimise.attack_image(model, img, label, patch, loss_fn="ce", steps=1,
                          lr=0.02, optimiser="sign", log_every=100,
                          out_dir=None, save_best=False, verbose=False,
                          log=lambda *a, **k: None)

    moved = (patch.param.detach() - start).abs()
    moved = moved[moved > 0]
    assert moved.numel() > 0
    assert torch.allclose(moved, torch.full_like(moved, 0.02), atol=1e-7)
