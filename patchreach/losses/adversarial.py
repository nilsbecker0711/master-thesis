r"""
Adversarial objectives.

  ce            untargeted, -CE(logits, GT)
  cospgd        untargeted, cos(pred,GT).detach() * CE, maximised
                (Agnihotri et al., ICML 2024)
  ce_target     TARGETED, plain per-pixel CE toward target_class, minimised.
                No weighting. This is what IAP's adversarial half actually
                specifies (`nn.CrossEntropyLoss()` used plainly), so it is the
                one to use when reproducing IAP rather than comparing against
                this repository's own targeted objective.
  ipatch_cospgd TARGETED CosPGD — Agnihotri et al.'s own Eq. 8, not a local
                combination. Drives every scored pixel toward target_class.
                THE IDENTIFIER IS A MISNOMER AND IS KEPT ANYWAY: every
                config.json and directory tag on record spells it this way,
                and the repo does not move names past runs recorded. Cite it
                as targeted CosPGD.
  tsallis       untargeted, -L_q(p_y), maximised — CE carrying a per-pixel
                gradient weight p^(1-q), with q optionally scheduled across
                the run. Defined in tsallis.py because it is stateful; build()
                below dispatches it. (Matyasko et al., IJCNN 2026)
  margin        untargeted w.r.t. the CLEAN PREDICTION, CW hinge
                relu(z_ref - max_{k!=ref} z_k + kappa), minimised. The
                differentiable count of any_flip_rate: a pixel flipped by kappa
                contributes exactly zero gradient, so effort moves to the
                pixels that have not flipped yet. (Carlini & Wagner, S&P 2017)
  cos_margin    margin carrying CosPGD's cosine weight, against the same clean
                prediction: cos(pred, onehot(ref)).detach() * the hinge. The
                hinge still cuts off hard at kappa; the cosine releases a pixel
                SMOOTHLY as it crosses its own boundary.

All six take an optional `support` mask restricting which pixels are scored.
That is where patch-footprint exclusion and reach-restriction compose: the
caller intersects them and passes one mask, rather than each loss knowing about
both.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _reduce(per_pixel: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Masked mean with a guarded denominator.

    THE NaN BUG THIS PREVENTS: F.cross_entropy(reduction='mean') computes
    sum/count over non-ignored pixels. Cityscapes batches DO occur where void
    regions (ego-vehicle hood, borders) plus the patch footprint cover every
    pixel — that is 0/0 = NaN. The NaN propagates through backward(), poisons
    Adam's moment estimates permanently, and the patch freezes at its init
    value for the rest of training with no error raised. Cost: one 100-epoch
    run before it was noticed.
    """
    n = valid.sum().clamp(min=1)
    return (per_pixel * valid.float()).sum() / n


def ce_loss(logits, labels, support: Optional[torch.Tensor] = None):
    """Untargeted CE. CE is HIGH when wrong -> maximise -> return -CE."""
    valid = labels != 255
    if support is not None:
        valid = valid & support
    per_pixel = F.cross_entropy(logits, labels, reduction="none",
                                ignore_index=255)
    return -_reduce(per_pixel, valid)


def cospgd_loss(logits, labels, support: Optional[torch.Tensor] = None):
    r"""
    CosPGD (Agnihotri et al., ICML 2024):

        L = mean_i [ cos(softmax(f(x)_i), onehot(y_i)) * CE(f(x)_i, y_i) ]

    The cosine SCALES the pixel-wise CE; it is a weight, never the objective.
    Pixels still predicted correctly (cos~1) keep full weight; pixels already
    fooled (cos~0) are de-emphasised. The gradient comes from CE, which does
    not saturate the way raw cosine does near alignment.

    FIDELITY NOTE — state this in the writeup: the paper does NOT detach the
    cosine; in their Algorithm 1 the gradient flows through the whole product.
    Detaching is a deliberate deviation, common in patch-attack
    implementations, that removes gradient contributions through the weight.

    A second deviation: they use sign-SGD with epsilon-projection (it is PGD).
    A patch is not an epsilon-ball perturbation, so we use Adam on the raw
    gradient. Both deviations are correct for this threat model but must be
    declared.

    Untargeted: MAXIMISE weighted CE -> return -L.
    """
    C = logits.shape[1]
    valid = labels != 255
    if support is not None:
        valid = valid & support

    safe = labels.clone()
    safe[labels == 255] = 0
    ref = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()

    cos_w = F.cosine_similarity(F.softmax(logits, 1), ref, dim=1).detach()
    ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    return -_reduce(cos_w * ce, valid)


def ce_target_loss(logits, target_class: int,
                   footprint: Optional[torch.Tensor] = None,
                   support: Optional[torch.Tensor] = None):
    r"""
    PLAIN TARGETED CE, per pixel:

        L = mean_i CE( f(x)_i, t )

    ipatch_cospgd_loss below, minus the cosine weight. That is the whole
    difference, and it exists because an unweighted targeted CE is what several
    published attacks actually specify.

    WHO NEEDS IT. IAP (Dutta & Zhang, ICCV 2025) builds its adversarial half
    from `criterion = nn.CrossEntropyLoss()` used plainly on both terms —

        1.5*criterion(pred, target) - criterion(pred, class_id) + 3*reg

    — with no cosine anywhere, in the paper or in the released code. Porting
    that to segmentation means a per-pixel targeted CE, which is this, and a
    per-pixel CE away from the ground truth, which is ce_loss above (it already
    returns -CE, so IAP's explicit minus is built in). Reusing
    ipatch_cospgd_loss instead would silently substitute targeted CosPGD for
    IAP's own objective — defensible for comparability against the rest of this
    repository, but not a reproduction, and the choice has to be the caller's.
    See patchreach/patch/iap.py.

    THE SIGN AND THE FOOTPRINT follow ipatch_cospgd_loss exactly: minimised to
    attack, and `footprint` is as mandatory here as it is there, for the same
    reason — leave the patch region in a targeted loss and the cheapest
    minimiser is to paint the PATCH in the target class, which collapses the
    loss and moves nothing remote.

    WHAT IT GIVES UP against the cosine-weighted version: every pixel keeps
    full weight however converted it already is, so a scene where most pixels
    flip early keeps spending gradient on them instead of on the holdouts.
    That is a real difference in behaviour, not a formality, and it is the
    reason both exist.
    """
    B, C, H, W = logits.shape
    target = torch.full((B, H, W), target_class, dtype=torch.long,
                        device=logits.device)
    if footprint is not None:
        target[footprint] = 255                       # camouflage prevention

    valid = target != 255
    if support is not None:
        valid = valid & support
    if valid.sum() == 0:
        raise ValueError("ce_target loss has no valid pixels — check the "
                         "footprint and reach masks do not exclude everything")

    per_pixel = F.cross_entropy(logits, target, reduction="none",
                                ignore_index=255)
    return _reduce(per_pixel, valid)


def ipatch_cospgd_loss(logits, target_class: int,
                       footprint: Optional[torch.Tensor] = None,
                       support: Optional[torch.Tensor] = None):
    r"""
    TARGETED COSPGD — Agnihotri et al., ICML 2024, Eq. 8:

        L = mean_i [ (1 - cos(softmax(f(x)_i), onehot(t))) * CE(f(x)_i, t) ]

    CITE IT AS TARGETED COSPGD. This is the paper's own construct, introduced
    there as "the targeted CosPGD update step ... in analogy to Eqn 5": the
    cosine is taken against the TARGET one-hot rather than the ground truth,
    the weight is (1 - cos) rather than cos, and the step DESCENDS. An earlier
    version of this docstring described it as "IPatch objective + targeted
    CosPGD weighting", which understated it — the weight and the
    cross-entropy it scales are both Eq. 8, and neither is a local
    combination. The function name predates that identification; see the
    module header for why it is not being renamed.

    The IPatch connection is a true aside, not the attribution: IPatch's
    KL(onehot(Y_t), Y) reduces to CE(logits, t) exactly, because KL from a
    one-hot is -log p_t. So the objective is simultaneously IPatch's, which is
    worth one sentence when motivating a REMOTE targeted patch, but the loss
    needs no second citation.

    WHY TARGETED AT ALL, for segmentation: every scored pixel is pushed toward
    the SAME class, so per-pixel gradients ALIGN and add — unlike untargeted
    objectives where pixel A says "not-road", pixel B says "not-sky", and the
    two partly cancel. The (1 - cos) weight de-emphasises pixels already
    converted, so effort concentrates on the stubborn remainder.

    THREE DEVIATIONS FROM EQ. 8, all to be declared wherever this is quoted:
      * the cosine is DETACHED here; the paper lets the gradient flow through
        the whole product (same deviation cospgd_loss documents);
      * the paper SUMS over pixels and this MEANS over them. Under the
        paper's sign() update the scale is irrelevant; under Adam or IAP's
        multiplicative rule it is absorbed into lr, so an lr is not
        transferable between the two conventions;
      * no sign-SGD and no epsilon-ball — a patch is not an epsilon-ball
        perturbation. `footprint` exclusion below is also ours, not Eq. 8's.

    `footprint` is MANDATORY. Leave the patch region in the loss and the
    cheapest win is to make the PATCH look like target_class: loss collapses,
    patch becomes camouflage, remote effect stays exactly zero.

    Adam MINIMISES this -> predictions driven toward target_class.
    """
    B, C, H, W = logits.shape
    target = torch.full((B, H, W), target_class, dtype=torch.long,
                        device=logits.device)
    if footprint is not None:
        target[footprint] = 255                       # camouflage prevention

    valid = target != 255
    if support is not None:
        valid = valid & support
    if valid.sum() == 0:
        raise ValueError("ipatch loss has no valid pixels — check the "
                         "footprint and reach masks do not exclude everything")

    # Ignored positions map to a DUMMY class 0, NOT clamp(max=C-1). Clamping
    # sends 255 -> class C-1 and computes the weight against a real edge class.
    # These pixels are zeroed by `valid` below so class 0 never enters the loss,
    # but the weight stays well-defined everywhere — matching the untargeted
    # path and the paper's Eq 8.
    safe = target.clone()
    safe[~valid] = 0
    ref = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()

    weight = 1.0 - F.cosine_similarity(F.softmax(logits, 1), ref, dim=1).detach()
    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=255)
    return _reduce(weight * ce, valid)


def margin_loss(logits, ref, support: Optional[torch.Tensor] = None,
                kappa: float = 5.0):
    r"""
    CW margin against the CLEAN PREDICTION:

        L = mean_i relu( z_ref(i) - max_{k != ref(i)} z_k + kappa )

    `ref` is the clean argmax, NOT the ground truth, because any_flip_rate
    measures change against the clean prediction. Scoring against GT would
    leave pixels the model already gets wrong nearly unscored while the metric
    still counts them.

    WHY NOT COSPGD FOR FLIP RATE: CosPGD's weight p_y/||p|| only APPROACHES
    zero, and the CE gradient it multiplies GROWS as a pixel flips, so flipped
    pixels keep most of the gradient. After an untargeted run collapses onto
    one class, that majority keeps pushing the collapse class up everywhere —
    exactly what the pixels already predicting that class need reversed. Here a
    pixel flipped by kappa contributes EXACTLY zero, and every unflipped pixel
    gets a unit-size gradient however confident it is, so the holdouts own the
    whole gradient.

    kappa is in LOGIT units. It demands a real margin so flips survive small
    patch updates; too large and flipped pixels keep drawing effort, which is
    the CE behaviour this exists to avoid.

    ref == 255 marks void (set by the caller from the label), excluded exactly
    as any_flip_rate excludes it. Minimise.
    """
    if logits.shape[1] < 2:
        raise ValueError("margin loss needs at least two classes")
    valid = ref != 255
    if support is not None:
        valid = valid & support

    safe = ref.clone()
    safe[~valid] = 0                      # dummy index, zeroed by `valid`
    idx = safe.unsqueeze(1)
    z_ref = logits.gather(1, idx).squeeze(1)
    z_other = logits.scatter(1, idx, float("-inf")).amax(1)
    return _reduce(F.relu(z_ref - z_other + kappa), valid)


def cos_margin_loss(logits, ref, support: Optional[torch.Tensor] = None,
                    kappa: float = 5.0):
    r"""
    CosPGD's weight on the CW hinge, both against the CLEAN PREDICTION:

        L = mean_i [ cos(softmax(f(x)_i), onehot(ref_i)).detach()
                     * relu( z_ref(i) - max_{k != ref(i)} z_k + kappa ) ]

    WHAT THE COSINE CHANGES, measured on the two-class approximation where
    w = p_ref/||p|| and m = z_other - z_ref (m > 0 means flipped):

        m       -7     -3     -1     +1     +3     +6
        margin  1.00   1.00   1.00   1.00   1.00   0
        cos     1.00   1.00   0.94   0.35   0.05   0

    Unflipped pixels keep ~full weight, so the confident holdouts a collapsed
    run leaves behind are still driven at full strength — the property plain
    CosPGD lacks, because CE's own gradient sigma(m) has already gone flat
    there. What moves is the region BETWEEN the boundary and kappa: the cosine
    releases a pixel smoothly once it flips rather than holding it at weight 1
    until kappa, so this behaves like margin with a softer, smaller kappa while
    kappa still bounds the hinge.

    The weight is DETACHED, matching cospgd_loss and carrying the same fidelity
    caveat: the CosPGD paper does not detach.

    The cosine is taken against ref, NOT the GT, so weight and hinge agree on
    what "this pixel's class" means. Mixing them would weight a pixel by its GT
    alignment while scoring its clean-prediction margin, and on pixels the model
    already gets wrong those are different questions.
    """
    if logits.shape[1] < 2:
        raise ValueError("cos_margin loss needs at least two classes")
    C = logits.shape[1]
    valid = ref != 255
    if support is not None:
        valid = valid & support

    safe = ref.clone()
    safe[~valid] = 0                      # dummy index, zeroed by `valid`
    idx = safe.unsqueeze(1)
    z_ref = logits.gather(1, idx).squeeze(1)
    z_other = logits.scatter(1, idx, float("-inf")).amax(1)

    onehot = F.one_hot(safe, num_classes=C).permute(0, 3, 1, 2).float()
    w = F.cosine_similarity(F.softmax(logits, 1), onehot, dim=1).detach()
    return _reduce(w * F.relu(z_ref - z_other + kappa), valid)


def _margin_unbound(*_):
    raise ValueError(
        "loss_fn='margin'/'cos_margin' needs the clean prediction bound at "
        "build time — build(loss_fn, margin_ref=clean_argmax). It is wired "
        "through "
        "patch/optimise.attack_image only; placement=gradcam with "
        "--cam_objective attack cannot use it.")


def build(loss_fn: str, target_class: int = 8,
          tsallis_q: float = 0.0, tsallis_schedule: str = "const",
          tsallis_q_start: float = -2.0, tsallis_q_end: float = 1.0,
          tsallis_total_steps: int = 1,
          margin_ref: Optional[torch.Tensor] = None,
          margin_kappa: float = 5.0):
    """
    Returns f(logits, labels, footprint, support) -> scalar.

    ipatch ALWAYS receives the footprint regardless of the exclusion flag —
    camouflage prevention is not optional for a targeted objective.

    The tsallis_* arguments are read by the 'tsallis' branch ONLY. They carry
    defaults so every existing two-argument call site is unchanged, and no
    other branch reads them.

    The margin_* arguments are read by the 'margin' branch ONLY, on the same
    terms. margin_ref is the clean argmax with void set to 255; the returned
    callable IGNORES the `labels` slot and scores against margin_ref instead.
    Built without margin_ref it returns a callable that raises on use, so a
    call site that cannot supply the clean prediction fails loudly rather than
    silently scoring against GT.
    """
    if loss_fn in ("margin", "cos_margin"):
        if margin_ref is None:
            return _margin_unbound
        ref = margin_ref
        fn = margin_loss if loss_fn == "margin" else cos_margin_loss
        return lambda lg, lb, fp, sp: fn(lg, ref, sp, margin_kappa)
    if loss_fn == "tsallis":
        from .tsallis import TsallisCELoss
        return TsallisCELoss(q=tsallis_q, schedule=tsallis_schedule,
                             q_start=tsallis_q_start, q_end=tsallis_q_end,
                             total_steps=tsallis_total_steps)
    if loss_fn == "ce_target":
        return lambda lg, lb, fp, sp: ce_target_loss(lg, target_class, fp, sp)
    if loss_fn == "ipatch_cospgd":
        return lambda lg, lb, fp, sp: ipatch_cospgd_loss(lg, target_class,
                                                         fp, sp)
    if loss_fn == "cospgd":
        return lambda lg, lb, fp, sp: cospgd_loss(lg, lb, sp)
    if loss_fn == "ce":
        return lambda lg, lb, fp, sp: ce_loss(lg, lb, sp)
    raise ValueError(f"unknown loss_fn {loss_fn!r}")
