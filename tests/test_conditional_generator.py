r"""
Invariants for the image-conditioned generator.

Same spirit as test_core.py: each of these encodes a way the implementation
could be silently wrong in a way no loss curve would reveal. They run without
mmseg — a tiny stand-in segmentor covers the Grad-CAM path — so they are
cheap enough to run on every edit.

Run with: pytest -q tests/test_conditional_generator.py
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from patchreach.data.cityscapes import upsample_to
from patchreach.losses import adversarial
from patchreach.patch import conditional_generator as cg
from patchreach.patch import segmentation_cam
from patchreach.patch.placement import find_max_response_placement
from patchreach.patch.spec import Patch, PatchConfig


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _cfg(**kw):
    base = dict(size=32, base_ch=8, depth=2)
    base.update(kw)
    return cg.GeneratorConfig(**base)


# ═════════════════════════════════════════════════════════════════════════════
#  Compositing — must not drift from Patch.apply()
# ═════════════════════════════════════════════════════════════════════════════

def test_composite_batch_matches_patch_apply():
    """
    THE LOAD-BEARING TEST. composite_batch() mirrors Patch.apply() because
    apply() cannot express per-image patches. If the two ever disagree, every
    conditional-generator number becomes incomparable with every existing
    patch-mode number, and nothing else in the suite would notice.

    Driven in the degenerate case apply() DOES cover: one shared patch, one
    shared placement, broadcast over the batch.
    """
    torch.manual_seed(0)
    H, W, B = 64, 128, 3
    imgs = torch.randn(B, 3, H, W)

    cfg = PatchConfig(mode="raw", size=16, scale=0.25)
    patch = Patch(cfg, torch.device("cpu"), MEAN, STD)
    with torch.no_grad():
        patch.param.normal_(0, 1.0)
    patch.placement = (7, 19)

    ref_patched, ref_fp = patch.apply(imgs)

    p = int(H * cfg.scale)
    rendered = patch.render().unsqueeze(0).expand(B, -1, -1, -1)
    got_patched, got_fp = cg.composite_batch(
        imgs, rendered, [(7, 19)] * B, p, MEAN, STD)

    assert torch.allclose(ref_patched, got_patched, atol=1e-6)
    assert torch.equal(ref_fp, got_fp)


def test_composite_batch_uses_per_image_placement():
    """A per-image attack that quietly shares one placement is a universal
    patch wearing a costume."""
    H, W, B = 64, 128, 2
    imgs = torch.zeros(B, 3, H, W)
    p = 16
    patches = torch.ones(B, 3, p, p)
    _, fp = cg.composite_batch(imgs, patches, [(0, 0), (40, 90)], p, MEAN, STD)

    assert fp[0, 0, 0] and not fp[0, 40, 90]
    assert fp[1, 40, 90] and not fp[1, 0, 0]
    assert int(fp[0].sum()) == int(fp[1].sum()) == p * p


def test_composite_batch_gradient_reaches_the_patch():
    """
    The bug Patch.apply()'s docstring warns about: an in-place slice into a
    no-grad tensor creates no backward node, so the gradient silently vanishes
    and the patch never learns.
    """
    H, W = 64, 128
    imgs = torch.randn(1, 3, H, W)
    patches = torch.rand(1, 3, 16, 16, requires_grad=True)
    patched, _ = cg.composite_batch(imgs, patches, [(10, 20)], 16, MEAN, STD)
    patched.sum().backward()
    assert patches.grad is not None
    assert float(patches.grad.abs().sum()) > 0


def test_composite_batch_rejects_batch_mismatch():
    with pytest.raises(ValueError, match="batch mismatch"):
        cg.composite_batch(torch.zeros(3, 3, 64, 128), torch.zeros(2, 3, 16, 16),
                           [(0, 0)] * 2, 16, MEAN, STD)


# ═════════════════════════════════════════════════════════════════════════════
#  Reference and residual
# ═════════════════════════════════════════════════════════════════════════════

def test_center_crop_reference_is_the_centre():
    """r_i = Resize(CenterCrop(x_i)) — not a corner, not a random crop."""
    H, W, p = 64, 128, 16
    img = torch.zeros(1, 3, H, W)
    top, left = (H - p) // 2, (W - p) // 2
    img[:, :, top:top + p, left:left + p] = 1.0

    r = cg.center_crop_reference(img, p, 32)
    assert r.shape == (1, 3, 32, 32)
    assert torch.allclose(r, torch.ones_like(r), atol=1e-5)
    assert not r.requires_grad          # r_i is never optimised


@pytest.mark.parametrize("residual", ["logit", "clip"])
def test_zero_init_generator_reproduces_the_reference(residual):
    """
    BASELINE A IS THE GENERATOR'S OWN STARTING POINT.

    The head is zero-initialised, so Delta == 0 and p_i == r_i exactly. Any
    later drift from the reference is attributable to training rather than to
    initialisation, which is what makes the A-vs-C comparison exact.
    """
    gen = cg.ConditionalPatchGenerator(_cfg(residual=residual))
    ref = torch.rand(2, 3, 32, 32)
    out = gen(image=torch.rand(2, 3, 32, 32), reference=ref,
              cam_global=torch.rand(2, 1, 32, 32),
              cam_local=torch.rand(2, 1, 32, 32))
    assert torch.allclose(out, ref, atol=1e-4)


def test_residual_none_starts_at_grey_like_raw_mode():
    """residual='none' ignores r as a base; sigmoid(0) = 0.5, matching how
    `raw` mode initialises in spec.py."""
    gen = cg.ConditionalPatchGenerator(_cfg(residual="none"))
    out = gen(image=torch.rand(1, 3, 32, 32), reference=torch.rand(1, 3, 32, 32),
              cam_global=torch.rand(1, 1, 32, 32),
              cam_local=torch.rand(1, 1, 32, 32))
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5)


def test_generated_patch_is_in_unit_range():
    """Patch.render() guarantees [0,1]; so must this, for every residual mode."""
    for residual in ("logit", "clip", "none"):
        gen = cg.ConditionalPatchGenerator(_cfg(residual=residual))
        with torch.no_grad():
            gen.head.weight.normal_(0, 5.0)
            gen.head.bias.normal_(0, 5.0)
        out = gen(image=torch.rand(2, 3, 32, 32),
                  reference=torch.rand(2, 3, 32, 32),
                  cam_global=torch.rand(2, 1, 32, 32),
                  cam_local=torch.rand(2, 1, 32, 32))
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, residual


def test_conditioning_modes_change_input_width_only():
    """Ablation E must vary the conditioning, not the output parameterisation."""
    widths = {c: cg.GeneratorConfig(size=32, cond=c).in_channels
              for c in cg.COND_MODES}
    assert widths == {"image": 3, "image+ref": 6, "image+ref+cam": 8}

    # cond='image' still uses r_i as the RESIDUAL BASE, so the ablation
    # isolates information rather than confounding it with a reparameterisation.
    gen = cg.ConditionalPatchGenerator(_cfg(cond="image"))
    ref = torch.rand(1, 3, 32, 32)
    assert torch.allclose(gen(image=torch.rand(1, 3, 32, 32), reference=ref),
                          ref, atol=1e-4)


def test_generator_is_shared_not_per_image():
    """One theta for the whole dataset: the parameter count must not depend on
    how many images were pushed through."""
    gen = cg.ConditionalPatchGenerator(_cfg())
    before = gen.n_parameters()
    for _ in range(3):
        gen(image=torch.rand(4, 3, 32, 32), reference=torch.rand(4, 3, 32, 32),
            cam_global=torch.rand(4, 1, 32, 32),
            cam_local=torch.rand(4, 1, 32, 32))
    assert gen.n_parameters() == before


def test_different_images_get_different_patches():
    """The defining property of the threat model. A trained generator that
    ignores its input would silently be a universal patch."""
    gen = cg.ConditionalPatchGenerator(_cfg())
    with torch.no_grad():
        gen.head.weight.normal_(0, 1.0)
    imgs = torch.rand(2, 3, 32, 32)
    refs = torch.rand(2, 3, 32, 32)
    out = gen(image=imgs, reference=refs,
              cam_global=torch.rand(2, 1, 32, 32),
              cam_local=torch.rand(2, 1, 32, 32))
    assert not torch.allclose(out[0], out[1], atol=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
#  Placement
# ═════════════════════════════════════════════════════════════════════════════

def test_cam_placement_finds_the_hot_window():
    score = torch.zeros(40, 80)
    score[25:37, 60:72] = 1.0
    top, left = find_max_response_placement(score, p=12)
    assert (top, left) == (25, 60)


def test_constant_map_falls_back_to_centre_only_when_asked():
    """
    argmax over a flat tensor returns index 0 — the top-left corner — which
    reads as a deliberate placement and is not. A fully-suppressed Grad-CAM
    hits exactly this case, so the gradcam path opts in.

    The default stays OFF so find_semantic_placement's historical behaviour is
    reproduced bit for bit.
    """
    p = 12
    for m in (torch.zeros(40, 80), torch.full((40, 80), 0.5)):
        assert find_max_response_placement(m, p) == (0, 0)
        assert find_max_response_placement(m, p, centre_if_constant=True) \
            == (14, 34)


def test_semantic_placement_is_unchanged_by_the_refactor():
    """
    find_semantic_placement now delegates to find_max_response_placement.
    Existing behaviour must hold EXACTLY — including the degenerate case where
    the class covers the whole prediction and every window ties, which the old
    implementation resolved to (0,0).
    """
    from patchreach.patch.placement import find_semantic_placement
    pred = torch.zeros(40, 80, dtype=torch.long)
    pred[:18, :] = 2
    assert find_semantic_placement(pred, cls=0, p=12)[0] >= 18
    assert find_semantic_placement(pred, cls=17, p=12) == (14, 34)

    everywhere = torch.zeros(40, 80, dtype=torch.long)
    assert find_semantic_placement(everywhere, cls=0, p=12) == (0, 0)


def test_margin_keeps_the_window_off_the_border():
    """
    The measured failure: the sensitivity map's hottest ridge is the near-field
    road boundary along the BOTTOM of a dashcam frame, so the argmax pinned the
    window at top = H - p exactly. A patch on the border has roughly half its
    receptive field outside the image.
    """
    H, W, p = 40, 80, 12
    score = torch.zeros(H, W)
    score[H - p:, :p] = 1.0                       # hottest at the bottom-left

    assert find_max_response_placement(score, p) == (H - p, 0)   # flush
    top, left = find_max_response_placement(score, p, margin=6)
    assert top == H - p - 6 and left == 6
    assert 6 <= top <= H - p - 6 and 6 <= left <= W - p - 6


def test_margin_zero_is_the_unmargined_argmax():
    """Default must reproduce existing behaviour bit for bit."""
    torch.manual_seed(0)
    score = torch.rand(40, 80)
    assert (find_max_response_placement(score, 12)
            == find_max_response_placement(score, 12, margin=0))


def test_oversized_margin_degrades_rather_than_raising():
    """A margin larger than the image can accommodate must not crash."""
    score = torch.rand(40, 80)
    top, left = find_max_response_placement(score, 12, margin=500)
    assert 0 <= top <= 40 - 12 and 0 <= left <= 80 - 12


def test_batch_placement_passes_the_margin_through():
    cam = torch.zeros(1, 1, 40, 80)
    cam[0, 0, 28:, :12] = 1.0
    assert cg.resolve_batch_placement("gradcam", 40, 80, 12, cam=cam) == [(28, 0)]
    assert cg.resolve_batch_placement("gradcam", 40, 80, 12, cam=cam,
                                      margin=6) == [(22, 6)]


def test_window_reference_is_the_content_the_patch_replaces():
    """
    r_i sampled at the placement, not the centre. This removes the perspective
    mismatch: a centre crop of a dashcam frame is mid-distance content, while
    gradcam placement lands in the near field.
    """
    H, W, p = 64, 128, 16
    img = torch.zeros(1, 3, H, W)
    img[:, :, 40:56, 90:106] = 1.0                # mark ONLY the target window

    r = cg.window_reference(img, [(40, 90)], p, 32)
    assert torch.allclose(r, torch.ones_like(r), atol=1e-5)
    assert not r.requires_grad

    # the centre crop of the same image sees none of it
    assert float(cg.center_crop_reference(img, p, 32).max()) == 0.0


def test_window_reference_makes_baseline_a_a_literal_noop():
    """
    THE POINT OF THE OPTION. With reference='window', p_i = r_i reproduces
    exactly the pixels it covers, so the patched image equals the original and
    baseline A degrades nothing BY CONSTRUCTION. Every point of degradation is
    then attributable to the generator.

    Driven at size == p, which is the real configuration (128 == 128 at the
    default geometry). When they differ the crop makes a bilinear round trip
    p -> S -> p, so the reconstruction is close but not exact.

    The image is built by NORMALISING a valid [0,1] tensor rather than using
    randn directly: denormalise_batch clamps to [0,1], so a synthetic tensor
    with out-of-gamut pixels would not survive the round trip. Real dataloader
    output is in gamut, so this is the faithful case, not a convenient one.
    """
    model = _cam_model()
    labels = torch.randint(0, 19, (2, 64, 128))
    imgs = (torch.rand(2, 3, 64, 128) - MEAN) / STD

    cam = segmentation_cam.SegmentationCAM(model, adversarial.build("ce"),
                                           target="pred")
    attack = cg.ConditionalAttack(model, cam, None, MEAN, STD, scale=0.25,
                                  size=16, placement="gradcam",
                                  method="reference", reference="window")
    out = attack(imgs, labels)
    assert out["patch_side"] == 16                    # p == size, no resampling
    assert torch.allclose(out["patched"], imgs, atol=1e-5)
    cam.close()

    # and the centre-crop variant does NOT have that property
    cam2 = segmentation_cam.SegmentationCAM(model, adversarial.build("ce"),
                                            target="pred")
    centre = cg.ConditionalAttack(model, cam2, None, MEAN, STD, scale=0.25,
                                  size=16, placement="gradcam",
                                  method="reference", reference="center")
    assert not torch.allclose(centre(imgs, labels)["patched"], imgs, atol=1e-2)
    cam2.close()


def test_reference_mode_is_validated():
    with pytest.raises(ValueError, match="reference must be one of"):
        cg.ConditionalAttack(None, None, None, MEAN, STD, scale=0.25, size=32,
                             reference="centre")     # British spelling typo


def test_batch_placement_is_per_image():
    cam = torch.zeros(2, 1, 40, 80)
    cam[0, 0, 25:37, 60:72] = 1.0
    cam[1, 0, 0:12, 0:12] = 1.0
    places = cg.resolve_batch_placement("gradcam", 40, 80, 12, cam=cam)
    assert places == [(25, 60), (0, 0)]

    centred = cg.resolve_batch_placement("center", 40, 80, 12, cam=cam)
    assert centred == [(14, 34), (14, 34)]


# ═════════════════════════════════════════════════════════════════════════════
#  Segmentation Grad-CAM
# ═════════════════════════════════════════════════════════════════════════════

class _TinySeg(nn.Module):
    """
    Stand-in for WrappedSegModel: a `backbone` returning a TUPLE of multi-scale
    maps (mmseg's contract) and frozen parameters (the wrapper freezes every
    parameter, which is what makes the enable_grad/requires_grad_ dance in
    SegmentationCAM necessary).
    """

    class _BB(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, stride=4, padding=1)
            self.c2 = nn.Conv2d(8, 16, 3, stride=4, padding=1)

        def forward(self, x):
            a = torch.relu(self.c1(x))
            return (a, torch.relu(self.c2(a)))

    def __init__(self, K=19):
        super().__init__()
        self.backbone = self._BB()
        self.head = nn.Conv2d(16, K, 1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.head(self.backbone(x)[-1])


def _cam_model():
    torch.manual_seed(0)
    m = _TinySeg()
    # SegmentationCAM resolves dotted paths against `.model` when present,
    # mirroring WrappedSegModel. Expose it the same way.
    wrapper = nn.Module()
    wrapper.model = m
    wrapper.forward = m.forward
    return wrapper


def test_cam_is_normalised_detached_and_full_resolution():
    model = _cam_model()
    cam = segmentation_cam.SegmentationCAM(
        model, adversarial.build("cospgd"), layer=-1, target="pred")

    imgs = torch.randn(2, 3, 64, 128)
    labels = torch.randint(0, 19, (2, 64, 128))
    M, logits = cam(imgs, labels)

    assert M.shape == (2, 1, 64, 128)
    assert float(M.min()) >= 0.0 and float(M.max()) <= 1.0
    # GRADIENT SEPARATION: the map is a conditioning signal, so nothing may
    # flow from it back into the frozen segmentation model.
    assert not M.requires_grad and not logits.requires_grad
    assert all(not p.requires_grad for p in model.model.parameters())
    cam.close()


def test_cam_normalises_per_sample_not_across_the_batch():
    """A batch-wide min-max makes each map depend on which other images share
    the batch, so the same image would get a different patch at a different
    batch size."""
    model = _cam_model()
    cam = segmentation_cam.SegmentationCAM(model, adversarial.build("ce"),
                                           target="pred")
    imgs = torch.randn(3, 3, 64, 128)
    labels = torch.randint(0, 19, (3, 64, 128))
    M, _ = cam(imgs, labels)
    for i in range(3):
        assert float(M[i].max()) == pytest.approx(1.0, abs=1e-5)
    cam.close()


def test_cam_leaves_no_grad_on_frozen_parameters():
    """The model is frozen: the CAM's internal backward must not populate
    .grad on any parameter, or a later optimiser step would move the victim."""
    model = _cam_model()
    cam = segmentation_cam.SegmentationCAM(model, adversarial.build("cospgd"),
                                           target="pred")
    cam(torch.randn(1, 3, 64, 128), torch.randint(0, 19, (1, 64, 128)))
    assert all(p.grad is None for p in model.model.parameters())
    cam.close()


def test_cam_works_inside_no_grad():
    """Evaluation and export both call the CAM under no_grad. Without the
    internal enable_grad(), autograd.grad() raises there."""
    model = _cam_model()
    cam = segmentation_cam.SegmentationCAM(model, adversarial.build("cospgd"),
                                           target="pred")
    with torch.no_grad():
        M, _ = cam(torch.randn(1, 3, 64, 128),
                   torch.randint(0, 19, (1, 64, 128)))
    assert M.shape == (1, 1, 64, 128)
    cam.close()


def test_cam_bad_layer_index_raises_with_the_shapes():
    model = _cam_model()
    cam = segmentation_cam.SegmentationCAM(model, adversarial.build("ce"),
                                           layer=7, target="pred")
    with pytest.raises(IndexError, match="feature map"):
        cam(torch.randn(1, 3, 64, 128), torch.randint(0, 19, (1, 64, 128)))
    cam.close()


def test_cam_bad_module_path_lists_the_children():
    with pytest.raises(AttributeError, match="Available children"):
        segmentation_cam.SegmentationCAM(_cam_model(), adversarial.build("ce"),
                                         module="encoder")


# ═════════════════════════════════════════════════════════════════════════════
#  Token reshaping — sqrt(N) is WRONG at 512x1024
# ═════════════════════════════════════════════════════════════════════════════

def test_token_grid_respects_the_input_aspect_ratio():
    """A 2:1 image does not have a square token grid. Guessing sqrt(N) would
    transpose the map and every localisation conclusion with it."""
    a = torch.randn(2, (512 // 16) * (1024 // 16), 64)
    out = segmentation_cam.as_spatial(a, 512, 1024)
    assert out.shape == (2, 64, 32, 64)


def test_token_grid_drops_a_class_token():
    a = torch.randn(1, (64 // 8) * (128 // 8) + 1, 32)
    assert segmentation_cam.as_spatial(a, 64, 128).shape == (1, 32, 8, 16)


def test_unfactorisable_tokens_raise_rather_than_guess():
    with pytest.raises(ValueError, match="cannot factorise"):
        segmentation_cam.as_spatial(torch.randn(1, 37, 8), 64, 128)


def test_cnn_feature_map_passes_through():
    a = torch.randn(2, 16, 8, 16)
    assert segmentation_cam.as_spatial(a, 64, 128) is a


# ═════════════════════════════════════════════════════════════════════════════
#  LAP + checkpointing
# ═════════════════════════════════════════════════════════════════════════════

def test_lap_terms_are_zero_and_free_when_unweighted():
    """Weights default to 0 because L_rat/L_tv are SUMS while the attack loss
    is a per-pixel MEAN — 3-5 orders apart, per lap.py's warning."""
    out = cg.lap_terms(torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16))
    assert float(out["total"]) == 0.0
    assert float(out["rat"]) == 0.0 and float(out["tv"]) == 0.0


def test_lap_rat_is_scored_against_each_images_own_reference():
    """A per-image attack scored against a shared reference would be measuring
    the wrong thing entirely."""
    p = torch.zeros(2, 3, 8, 8)
    r = torch.zeros(2, 3, 8, 8)
    r[1] = 1.0                                   # only image 1 differs
    out = cg.lap_terms(p, r, alpha=1.0)
    expected = (0.0 + (3 * 8 * 8) ** 0.5) / 2    # mean over the batch
    assert float(out["rat"]) == pytest.approx(expected, rel=1e-4)


def test_lap_gradient_reaches_the_generator():
    gen = cg.ConditionalPatchGenerator(_cfg())
    with torch.no_grad():
        gen.head.weight.normal_(0, 0.5)
    ref = torch.rand(2, 3, 32, 32)
    patches = gen(image=torch.rand(2, 3, 32, 32), reference=ref,
                  cam_global=torch.rand(2, 1, 32, 32),
                  cam_local=torch.rand(2, 1, 32, 32))
    cg.lap_terms(patches, ref, alpha=1.0, beta=1.0)["total"].backward()
    assert float(gen.head.weight.grad.abs().sum()) > 0


def test_checkpoint_stores_theta_not_a_patch(tmp_path):
    """
    A single generated patch is an OUTPUT of this model, not the model. Saving
    one would make the artefact indistinguishable from a universal-patch
    checkpoint and lose the only thing that generalises.
    """
    gen = cg.ConditionalPatchGenerator(_cfg())
    opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
    path = tmp_path / "ck.pt"
    cg.save_checkpoint(path, gen, opt, 7, {"lr": 1e-4},
                       {"scale": 0.25, "size": 32})

    ck = torch.load(path, map_location="cpu")
    assert "param" not in ck
    assert set(("generator_state_dict", "optimizer_state_dict", "epoch",
                "train_config", "generator_config")) <= set(ck)
    assert ck["epoch"] == 7

    loaded, _ = cg.load_checkpoint(path, torch.device("cpu"))
    assert not loaded.training                     # eval() at test time
    for k, v in gen.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_patch_checkpoint_is_rejected_with_a_useful_message(tmp_path):
    """Loading a train.py patch here should say so, not fail on a missing key."""
    path = tmp_path / "patch.pt"
    torch.save({"param": torch.zeros(3, 8, 8), "config": {}}, path)
    with pytest.raises(ValueError, match="Patch.load"):
        cg.load_checkpoint(path, torch.device("cpu"))


# ═════════════════════════════════════════════════════════════════════════════
#  End to end
# ═════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_trains_theta_and_leaves_the_model_frozen():
    """
    The whole objective, one step:
        x -> M (detached) -> r -> p = G(x,r,M) -> x_adv -> f -> L -> theta

    Asserts the three things that make it the right objective: theta moves, the
    victim does not, and the attack loss is the repository's own.
    """
    torch.manual_seed(0)
    model = _cam_model()
    cam = segmentation_cam.SegmentationCAM(model, adversarial.build("cospgd"),
                                           target="pred")
    gen = cg.ConditionalPatchGenerator(_cfg(size=32, depth=2))
    attack = cg.ConditionalAttack(model, cam, gen, MEAN, STD, scale=0.25,
                                  size=32, placement="gradcam")

    imgs = torch.randn(2, 3, 64, 128)
    labels = torch.randint(0, 19, (2, 64, 128))
    adv_loss = adversarial.build("cospgd")
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)

    before = gen.head.weight.detach().clone()
    out = attack(imgs, labels)
    assert out["patches"].shape == (2, 3, 32, 32)
    assert out["patched"].shape == imgs.shape
    assert out["footprint"].shape == (2, 64, 128)
    assert not out["cam"].requires_grad          # conditioning is detached
    assert not out["references"].requires_grad   # r_i is never optimised

    # upsample_to before the loss, exactly as train_conditional_generator.py
    # does — the head emits at stride 16 and the loss is scored per pixel.
    logits = upsample_to(model(out["patched"]), labels.shape[-2:])
    adv_loss(logits, labels, out["footprint"], ~out["footprint"]).backward()
    opt.step()

    assert not torch.equal(before, gen.head.weight)          # theta moved
    assert all(p.grad is None for p in model.model.parameters())  # frozen
    cam.close()


def test_reference_method_is_the_generator_free_baseline():
    """Baseline A must run through IDENTICAL placement and compositing, so the
    A-vs-C delta is attributable to the generator alone."""
    model = _cam_model()
    cam = segmentation_cam.SegmentationCAM(model, adversarial.build("ce"),
                                           target="pred")
    attack = cg.ConditionalAttack(model, cam, None, MEAN, STD, scale=0.25,
                                  size=32, placement="gradcam",
                                  method="reference")
    out = attack(torch.randn(1, 3, 64, 128), torch.randint(0, 19, (1, 64, 128)))
    assert torch.equal(out["patches"], out["references"])
    cam.close()


def test_evaluation_noise_is_deterministic():
    """The claim is 'one frozen generator, one patch per image'. A resampled z
    would make the reported test-time patch irreproducible."""
    gen = cg.ConditionalPatchGenerator(_cfg(noise_dim=2))
    z = gen.sample_noise(3, torch.device("cpu"), deterministic=True)
    assert torch.equal(z, torch.zeros_like(z))
    assert not torch.equal(gen.sample_noise(3, torch.device("cpu")), z)
    assert cg.GeneratorConfig(size=32).noise_dim == 0      # off by default


def test_config_rejects_indivisible_size():
    with pytest.raises(ValueError, match="divisible"):
        cg.GeneratorConfig(size=100, depth=3).validate()
