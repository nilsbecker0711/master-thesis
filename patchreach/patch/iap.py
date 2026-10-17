r"""
IAP — Invisible Adversarial Patch (Dutta & Zhang, ICCV 2025), ported to
semantic segmentation.

    paper  https://arxiv.org/abs/2507.06856
    code   https://github.com/subratkishoredutta/IAP  (src/IAP.py, src/utils.py)

WHY IT IS HERE
--------------
It is the closest published neighbour to this repository's csf mode: both
answer "make the patch hard to see" with a per-pixel visibility model rather
than with an epsilon-ball, and both start the patch FROM the content it
replaces. The difference is which visibility model. csf uses a contrast
sensitivity function over spatial frequency; IAP uses a local-texture
sensitivity map, pixel-local, no frequency axis at all. Having both means the
thesis compares two visibility models under one attack, one dataset and one set
of architectures — which is a result — rather than quoting IAP's ImageNet
classification numbers beside Cityscapes segmentation numbers, which is not.

WHAT IAP IS, in four parts. Every one is reproduced here.

  1. SENSITIVITY MAP          sens = 1 / (sqrt(min(std_x, std_y)) + 1e-3),
                              std over a 1x3 and a 3x1 neighbourhood, averaged
                              over RGB. HIGH where the image is smooth, because
                              that is where a perturbation shows.
  2. LOCALISATION             the p x p window maximising a Grad-CAM response.
  3. PERCEPTIBILITY-REGULARISED LOSS
                              w1 * (toward target) + w2 * (away from truth)
                              + w3 * sum( sens * |delta - base| ).
  4. COLOUR-CONSTANT UPDATE   delta <- delta - lr * mean_c(grad) * delta / sens.
                              Three multiplications, each load-bearing — see
                              colour_constant_step_.

and around them: delta initialised to the image region itself, a ladder of
learning rates restarted on failure, and an early stop on attack success.

WHAT CHANGES FOR SEGMENTATION, AND NOTHING ELSE CHANGES
-------------------------------------------------------
The losses in part 3. IAP's two adversarial terms are classification
cross-entropies on one logit vector; here they are this repository's own
segmentation objectives from losses/adversarial.py, scored over every remote
pixel. That substitution is the whole point of the port: the procedure is
IAP's, the objective is this thesis's, so a performance or stealthiness
difference against csf is attributable to the visibility model.

Parts 1, 2 and 4 are pixel-space operations on the patch and its neighbourhood.
They do not know what task the model performs and are ported unchanged.

DECLARED DEVIATIONS — every one of these belongs in the write-up
----------------------------------------------------------------
D1  THE PAPER AND THE RELEASED CODE DISAGREE ON THE LOCALISATION SCORE. The
    paper scores a window by J_y / Sens; utils.py computes exactly that
    (`perturbPrior`) and then calls find_location(image, mask, ...) with the
    RAW CAM, so perturbPrior is dead code and the released localisation is
    Grad-CAM alone. Both are available here via `score=`, and the DEFAULT is
    'cam' — the released behaviour, because that is the procedure that produced
    the published numbers. 'cam_over_sens' is the paper's text.

D2  GRAD-CAM. Classification Grad-CAM differentiates one logit; there is no
    such scalar for a segmentor. This uses the repository's SegmentationCAM
    (S_seg = -L, see patch/segmentation_cam.py) with the model's own argmax as
    the pseudo-label, which makes the score the mean log-probability of the
    prediction rather than a raw logit — the post-softmax Grad-CAM variant.
    The reference additionally passes eigen_smooth=True and aug_smooth=True to
    pytorch_grad_cam; neither is reproduced. aug_smooth averages the map over
    flips and scale jitter, which is six extra full-resolution forward AND
    backward passes per image at 1024x2048.

D3  THE REGULARISER IS A SUM over patch elements, and the patch grew from
    84x84 to 256x256 — 9.3x more elements. At equal mean |delta - base| the
    term is therefore 9.3x larger against adversarial losses that are per-pixel
    MEANS and have not changed scale. w3 = 3 is consequently NOT the same
    operating point it was in the paper. `reduce='sum'` is the released
    arithmetic and the default; `reduce='sum84'` rescales by (84/p)**2 so that
    w3 means at 256px what it meant at 84px. Sweep w3; do not inherit it.

D4  THE LADDER'S FAILURE BRANCH. The released code tries lr in
    (0.05, 0.09, 0.2), breaks on success, and on failure reports whatever
    happened to be in scope from the last rung (its `conf`, `ssim_monitor_*`
    and `pur` are stale or undefined when every rung fails). Here every rung is
    recorded and, absent a success, the rung with the largest remote mIoU drop
    is the headline. Segmentation has no binary success, so this is not a tidy-
    up of a bug but a necessary definition.

D5  THE EARLY STOP. "argmax == target and softmax >= 0.9" has no single-label
    analogue. The criteria here are built on the metrics this thesis already
    reports (metrics/miou.attack_rates) — see `progress_rate`. All of them are
    strictly harder to satisfy than a one-label flip, so the ladder usually
    runs every rung, which is the compute IAP budgets for a failed attack.

D6  CLIPPING. The paper says the update is clipped to the valid pixel range;
    the released code never clips, so delta can leave [0,1] and the saved
    "image" is not one. Clipping here is Patch.project() under
    pixel_param='direct' — the repository's own PGD projection — and it is ON
    by default. `clamp=False` reproduces the released code.

D7  THE SENSITIVITY MAP'S BORDER. The reference's windows are ANCHORED at
    (i, j), not centred, and the loop leaves the last two rows of std_x and the
    last two columns of std_y at zero, so sens = 1/1e-3 = 1000 in a two-pixel
    band at the right and bottom edges of the frame. Reproduced exactly: a
    patch placed flush against those edges has its last two rows/columns
    effectively frozen (1/sens = 1e-3 in the update). Use the placement margin
    if that matters; do not silently "fix" it, because the published numbers
    were produced with it.

D8  THE FOOTPRINT IS EXCLUDED from both adversarial terms. Classification has
    no analogue; segmentation must, or the cheapest minimiser of a targeted
    loss is to paint the patch itself in the target class — loss collapses,
    remote effect stays zero. See losses/adversarial.ipatch_cospgd_loss.

D9  IN THE RELEASED CODE THE REGULARISER HAS NO GRADIENT AT ALL, and this is
    the largest discrepancy in the port. IAP.py builds its reference patch as

        delta = torch.nn.Parameter(delta)
        temp  = delta.clone()

    and Tensor.clone() on a leaf that requires grad is DIFFERENTIABLE: temp
    carries grad_fn=CloneBackward0 back to delta. So for
    R = sum(sens * |delta - temp|) autograd walks two paths into one leaf,

        dR/d(delta) through delta = + sens * sign(delta - temp)
        dR/d(delta) through temp  = - sens * sign(delta - temp)

    and they cancel EXACTLY, elementwise, at every step. Verified numerically
    on the reference's own expression: with |delta - temp| held at 0.1 the term
    is worth 2542.7 and what reaches delta.grad is 0.000e+00.

    So the perceptibility regulariser changes the REPORTED loss — it is the
    paper's "human distance" column — and changes the OPTIMISATION not at all.
    Whatever invisibility the released code achieves comes from the 1/sens
    factor in the update rule and from starting at the image region, i.e. from
    parts 1 and 4, and not from part 3. A direct consequence worth stating in
    the write-up: under the released code w3 cannot affect the result, so a w3
    ablation against it is a null experiment by construction.

    THE DEFAULT HERE IS THE PAPER, not the released code. `base` is detached,
    so the gradient is live and equals w3 * sens * sign(delta - base). This is
    the one place the port deliberately prefers the paper, and the reason is
    that the alternative is to reproduce a method whose central regularisation
    term provably does nothing — which is not what "IAP" denotes in the
    literature and not how a reviewer would read the name. D1 goes the other
    way, to the released code, because there BOTH branches are coherent
    procedures; here one of them is a dropped term.

    reg_grad='released' reproduces the cancellation exactly, by adding the
    term's value to the loss detached: identical value, identical (zero)
    gradient. So the two can be measured against each other instead of
    argued about, and "the regulariser contributes nothing" becomes a row.

WHAT THIS MODULE IS NOT
-----------------------
It is not a PatchConfig mode. IAP's parameterisation IS this repository's
raw + pixel_param='direct' — the parameter is the pixel, projected back into
[0,1] after the step — so a new mode would be a second name for an existing
one. What IAP adds is a different STEP RULE and a different REGULARISER, and
those live here, driven by scripts/iap_attack.py.

pixel_param='sigmoid' is REFUSED by the attack loop, and that is not a
preference. The update multiplies by `delta` itself (colour constancy); under a
sigmoid parameterisation that factor is a LOGIT, which is negative for any
pixel below 0.5 and would flip the sign of the step for half the patch.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..data.cityscapes import upsample_to
from ..losses import adversarial
from ..metrics.miou import single_image_miou, attack_rates
from .placement import find_max_response_placement

# ── the released settings, named once ────────────────────────────────────────
# src/IAP.py: learning_rates = [0.05,0.09,0.2]; iterations = 1000; the loss
# weights are the literals 1.5, 1.0 (implicit) and 3 in its `loss = ...` line.
# src/utils.py: sens = 1/(stdimg+0.001).
IAP_LRS: Tuple[float, ...] = (0.05, 0.09, 0.2)
IAP_STEPS = 1000
IAP_W1, IAP_W2, IAP_W3 = 1.5, 1.0, 3.0
SENS_EPS = 1e-3
# 84x84 on 224x224 — the geometry every published IAP number was measured at.
# Only reduce='sum84' reads it; see D3.
REF_PATCH_SIDE = 84

REG_REDUCTIONS = ("sum", "sum84", "mean")
# 'center' is NOT part of IAP. It is the ablation that isolates part 2 from
# parts 3 and 4: the same loss and the same update rule, at the position every
# other arm of this thesis uses, so the difference between it and 'cam' IS the
# contribution of perceptibility-aware localisation. Yuan et al.'s Table 7 puts
# centre at the STRONGEST position, so this control can beat the real thing on
# attack strength while losing on invisibility — which is the trade IAP claims
# to make, stated as a measurement.
LOCALISE_SCORES = ("cam", "cam_over_sens", "center")
STOP_CRITERIA = ("none", "flip", "target", "drop")
# D9. 'live' is the paper's regulariser and the default; 'released' reproduces
# the released code's exact gradient cancellation, where the term is reported
# and optimises nothing.
REG_GRADS = ("live", "released")


# ═════════════════════════════════════════════════════════════════════════════
#  1. the sensitivity map
# ═════════════════════════════════════════════════════════════════════════════

def window_std(x: torch.Tensor, dh: int, dw: int) -> torch.Tensor:
    r"""
    [B,h,w] of channel-averaged standard deviations over a dh x dw window,
    h = H-dh+1, w = W-dw+1.

        out[b,i,j] = mean_c std( x[b, c, i:i+dh, j:j+dw] )

    with std the UNBIASED (n-1) estimator — torch's default, and what
    utils.py's `subset[0].std()` computes.

    THE REFERENCE IS A DOUBLE PYTHON LOOP over every pixel. At 224x224 that is
    50k iterations and merely slow; at 1024x2048 it is 2.1 MILLION iterations
    per direction, twice per image, which is not a port, it is an outage. This
    is the same arithmetic as a tensor op.

    Implemented by STACKING the dh*dw shifted views and calling torch.std on
    the stack, NOT by the sum-of-squares identity var = (S2 - S^2/n)/(n-1).
    The identity is one conv2d and would be faster, but on 3 samples drawn from
    a near-constant window it is catastrophic cancellation: S2 and S^2/n agree
    to six digits precisely where the map matters most (smooth regions, which
    is where sens is LARGE and the whole mechanism lives), and the fp32
    difference is noise. The stack costs dh*dw views — 3 — and is exact.
    """
    if dh < 1 or dw < 1:
        raise ValueError(f"window must be at least 1x1, got {dh}x{dw}")
    H, W = x.shape[-2:]
    h, w = H - dh + 1, W - dw + 1
    if h < 1 or w < 1:
        raise ValueError(
            f"a {dh}x{dw} window does not fit a {H}x{W} image")
    sl = torch.stack([x[:, :, i:i + h, j:j + w]
                      for i in range(dh) for j in range(dw)], dim=-1)
    if sl.shape[-1] < 2:
        # std of one sample is undefined; the reference never asks for it.
        raise ValueError("window must contain at least two pixels")
    return sl.std(dim=-1, unbiased=True).mean(dim=1)


def _anchor(std: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Place a [B,h,w] window statistic at the window's TOP-LEFT pixel and leave
    the remainder zero — utils.py's `stdten[i][j] = std` with i, j the loop
    variables. See D7 for what the zero band costs.
    """
    out = std.new_zeros(std.shape[0], H, W)
    out[:, :std.shape[-2], :std.shape[-1]] = std
    return out


def sensitivity(x01: torch.Tensor, eps: float = SENS_EPS) -> torch.Tensor:
    r"""
    [B,H,W] sensitivity, HIGH where a perturbation would be visible.

        sens = 1 / ( sqrt( min(std_x, std_y) ) + eps )

    std_x over a 1x3 (horizontal) and std_y over a 3x1 (vertical)
    neighbourhood. `min` is what makes it an EDGE detector rather than a
    texture detector: an axis-aligned edge has large variance across it and
    near-zero along it, the min takes the small one, and the edge is therefore
    scored as SENSITIVE. Perturbing along a clean edge is exactly what a human
    eye finds; perturbing inside foliage is not.

    `x01` must be in [0,1] — the scale the reference works on and the scale eps
    = 1e-3 is calibrated for. Pass a denormalised image, never a mean/std
    normalised one, or eps becomes meaningless and the 1/eps ceiling moves.
    """
    if x01.dim() != 4:
        raise ValueError(f"expected [B,3,H,W], got {tuple(x01.shape)}")
    H, W = x01.shape[-2:]
    x = x01.float()
    sx = _anchor(window_std(x, 1, 3), H, W)
    sy = _anchor(window_std(x, 3, 1), H, W)
    return 1.0 / (torch.minimum(sx, sy).sqrt() + eps)


# ═════════════════════════════════════════════════════════════════════════════
#  2. localisation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def localise(cam: torch.Tensor, sens: torch.Tensor, p: int,
             score: str = "cam", margin: int = 0,
             eps: float = SENS_EPS) -> Tuple[int, int]:
    r"""
    Top-left of the p x p window IAP attacks.

    cam  : [H,W] Grad-CAM response, non-negative. Unused under score='center'
           and may be None there.
    sens : [H,W] from sensitivity() above. Read only under
           score='cam_over_sens'.
    score: 'cam'           argmax_window sum(J).           THE RELEASED CODE.
           'cam_over_sens' argmax_window sum(J/(sens+eps)). THE PAPER'S TEXT.
           See D1 — they are not the same procedure and the two place the patch
           differently on real images.
           'center'        the image centre. NOT IAP — the localisation
           ablation; see LOCALISE_SCORES.

    `p` must be given in every case, including 'center', because the centre of
    the FRAME is not the top-left of the window.

    find_location() in utils.py maximises the window SUM; this delegates to
    placement.find_max_response_placement, which maximises the window MEAN. The
    window size is constant over the search, so the two argmaxes are identical,
    and ties break to the first position in both (`>` against an initial max of
    0 there, torch.argmax's first-occurrence here). Delegating rather than
    re-implementing is why --placement gradcam and IAP localisation cannot
    drift apart; a test pins the equality.
    """
    if score not in LOCALISE_SCORES:
        raise ValueError(f"score must be one of {LOCALISE_SCORES}, "
                         f"got {score!r}")
    if score == "center":
        H, W = sens.shape[-2:]
        return (H - p) // 2, (W - p) // 2
    if cam is None or cam.dim() != 2:
        raise ValueError(
            f"score={score!r} needs the Grad-CAM as [H,W], got "
            f"{None if cam is None else tuple(cam.shape)}")
    if score == "cam_over_sens":
        if sens is None:
            raise ValueError("score='cam_over_sens' needs the sensitivity map")
        field = cam / (sens + eps)
    else:
        field = cam
    return find_max_response_placement(field, p, centre_if_constant=True,
                                       margin=margin)


# ═════════════════════════════════════════════════════════════════════════════
#  3. the perceptibility regulariser
# ═════════════════════════════════════════════════════════════════════════════

def regulariser(delta: torch.Tensor, base: torch.Tensor,
                sens_patch: torch.Tensor, reduce: str = "sum",
                side_ref: int = REF_PATCH_SIDE) -> torch.Tensor:
    r"""
        R = REDUCE( sens * |delta - base| )

    the third term of IAP's loss, `3*(sens.detach()*torch.abs(delta-temp)).sum()`.

    `sens` is detached — it is a property of the CLEAN image and never of the
    patch, so a gradient through it would be a gradient through a constant that
    happens to be computed from a tensor.

    L1, not L2, and that is the point: an L2 penalty would spread a fixed
    perturbation budget evenly, while L1 lets the optimiser spend ALL of it on
    a few pixels and leave the rest at the base. Weighted by sens, those few
    pixels are the textured ones.

    reduce : 'sum'    the released arithmetic. Scale depends on p**2 — see D3.
             'sum84'  sum * (side_ref/p)**2, so w3 keeps its published meaning
                      at a patch size the paper never ran.
             'mean'   scale-free, and NOT what IAP does. Here only so the
                      size-dependence of 'sum' can be measured rather than
                      argued about.
    """
    if reduce not in REG_REDUCTIONS:
        raise ValueError(f"reduce must be one of {REG_REDUCTIONS}, "
                         f"got {reduce!r}")
    w = sens_patch.detach() * (delta - base).abs()
    if reduce == "mean":
        return w.mean()
    total = w.sum()
    if reduce == "sum84":
        p = delta.shape[-1]
        total = total * (float(side_ref) / float(p)) ** 2
    return total


# ═════════════════════════════════════════════════════════════════════════════
#  4. the colour-constant update
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def colour_constant_step_(param: torch.Tensor, sens_patch: torch.Tensor,
                          lr: float, eps: float = SENS_EPS) -> dict:
    r"""
    IAP's update, in place on `param`. Returns step diagnostics.

        param <- param - lr * mean_c(grad) * param * (1 / sens)

    src/IAP.py:
        delta.data = delta.data - learning_rate*delta.grad.mean(dim=1)
                     *delta.data*(1/sens)

    THREE FACTORS, EACH DOING ONE JOB.

    mean_c(grad)  every channel of a pixel gets the SAME update, so the pixel
                  moves along the grey axis and its RGB ratios — its hue — are
                  preserved. This is the "colour constancy" of the paper's
                  title claim and the reason it is not Adam: Adam updates each
                  channel independently and recolours the patch, which is the
                  single most visible thing a patch can do.

    * param       the update is RELATIVE. A dark pixel moves little in absolute
                  terms, a bright one moves more, which is Weber's law and also
                  keeps the step small near 0. Note the consequence: a pixel at
                  EXACTLY 0 is an absorbing state — its step is zero forever.
                  Starting from image content (where true black is rare) this
                  is a corner case, but it is why param must never be
                  initialised to zeros.

    * 1 / sens    = sqrt(min std) + eps, i.e. the LOCAL TEXTURE. Smooth
                  regions get a ~eps-sized step and are effectively frozen;
                  textured regions move freely. This is the invisibility
                  mechanism, and it is worth seeing what it does to the
                  regulariser: dR/d(delta) is proportional to sens, and this
                  factor divides by sens, so the regulariser's contribution to
                  the step is sens-INDEPENDENT — a uniform relative pull of
                  rate lr*w3 back toward the base. The adversarial gradient is
                  the only term the texture gates. Invisibility is therefore
                  not bought by penalising harder in smooth regions; it is
                  bought by not LISTENING to the attack there.

    `param` is the [3,p,p] parameter of a Patch in raw/direct mode, so the
    channel axis is -3 and the grad is read off `param.grad`. Clipping back
    into [0,1] is the CALLER's job, via Patch.project() — see D6 — because that
    is where this repository's projection lives and a second clamp here would
    fork the definition.
    """
    g = param.grad
    if g is None:
        raise RuntimeError(
            "IAP step with param.grad = None — the graph never reached the "
            "patch. Check that the composite was built from the parameter "
            "(Patch.apply) and that backward() ran.")
    if not torch.isfinite(g).all():
        raise RuntimeError("non-finite gradient at the IAP step")

    step = lr * g.mean(dim=-3, keepdim=True) * param.data * (1.0 / sens_patch)
    param.data.sub_(step)
    return {"step_rms": float(step.pow(2).mean().sqrt()),
            "step_absmax": float(step.abs().max()),
            "grad_absmean": float(g.abs().mean())}


# ═════════════════════════════════════════════════════════════════════════════
#  the loss
# ═════════════════════════════════════════════════════════════════════════════

class Objective:
    r"""
    IAP's three-term loss with this repository's segmentation losses in the
    first two slots.

        L = w1 * L_targeted + w2 * L_untargeted + w3 * R

    SIGNS LINE UP WITHOUT A NEGATION, which is worth stating because it looks
    like luck and is not. IAP minimises

        1.5*CE(pred, target) - 1.0*CE(pred, truth) + 3*R

    and every objective in losses/adversarial.py is already defined so that
    MINIMISING it strengthens the attack: 'ipatch_cospgd' returns +CE toward
    the target, 'ce' returns -CE against the ground truth. So IAP's second
    term, with its explicit minus, IS adversarial.ce_loss, and the port is a
    substitution rather than a rearrangement.

    tar : 'ipatch_cospgd' or 'none'. The targeted term. This repository has no
          plain targeted CE and one is not added here: ipatch_cospgd IS its
          targeted objective, every targeted number in the thesis is measured
          under it, and introducing a second targeted loss purely to match
          IAP's would make the IAP column incomparable to every other column.
          The deviation is the cosine weight (1 - cos to target), which
          de-emphasises pixels already converted.
    untar : 'ce', 'cospgd', 'margin', 'cos_margin' or 'none'. The term away
          from the truth. 'ce' is IAP's literal -CE(pred, truth).
    'none' on either slot drops it. tar='none' makes the attack UNTARGETED,
          which IAP never is — declare it as an extension, not as IAP.
    """

    def __init__(self, tar: str = "ipatch_cospgd", untar: str = "ce",
                 target_class: int = 8,
                 w1: float = IAP_W1, w2: float = IAP_W2, w3: float = IAP_W3,
                 reg_reduce: str = "sum", reg_grad: str = "live",
                 margin_ref: Optional[torch.Tensor] = None,
                 margin_kappa: float = 5.0):
        if reg_grad not in REG_GRADS:
            raise ValueError(f"reg_grad must be one of {REG_GRADS}, "
                             f"got {reg_grad!r}")
        if tar not in ("ipatch_cospgd", "none"):
            raise ValueError(
                f"iap_tar_loss must be 'ipatch_cospgd' or 'none', got {tar!r}")
        if untar not in ("ce", "cospgd", "margin", "cos_margin", "none"):
            raise ValueError(f"iap_untar_loss {untar!r} is not a loss in "
                             f"losses/adversarial.py")
        if tar == "none" and untar == "none":
            raise ValueError("both adversarial terms are 'none' — that run "
                             "optimises the regulariser alone and converges "
                             "back to the clean image region")
        if untar in ("margin", "cos_margin") and margin_ref is None:
            raise ValueError(f"iap_untar_loss {untar!r} scores against the "
                             f"CLEAN PREDICTION; pass margin_ref")
        self.tar_name, self.untar_name = tar, untar
        self.w1, self.w2, self.w3 = float(w1), float(w2), float(w3)
        self.reg_reduce = reg_reduce
        self.reg_grad = reg_grad
        self.margin_ref = margin_ref
        self._tar = (None if tar == "none"
                     else adversarial.build(tar, target_class))
        if untar == "none":
            self._untar = None
        elif untar in ("margin", "cos_margin"):
            self._untar = adversarial.build(untar, margin_ref=margin_ref,
                                            margin_kappa=margin_kappa)
        else:
            self._untar = adversarial.build(untar, target_class)

    def __repr__(self):
        g = "" if self.reg_grad == "live" else ",NO GRADIENT"
        return (f"IAP({self.w1:g}*{self.tar_name} + {self.w2:g}*"
                f"{self.untar_name} + {self.w3:g}*reg[{self.reg_reduce}{g}])")

    def on_step_begin(self, step: int, total: int):
        """
        Forward the progress counter to whichever term is stateful.

        NO TERM OFFERED HERE IS STATEFUL TODAY — tsallis is deliberately absent
        from the untargeted choices, because IAP's second term is a plain
        cross-entropy and a scheduled q would stack a second deviation on top
        of the port. The hook exists anyway, and that is what
        test_tsallis.test_every_optimisation_loop_installs_the_step_hook is
        for: a loop that calls .backward() without it would pin a scheduled
        loss at q_start for the whole run and report a wrong number with no
        error. Adding the term later must not also require remembering this.
        """
        for fn in (self._tar, self._untar):
            if hasattr(fn, "on_step_begin"):
                fn.on_step_begin(step, total)

    def __call__(self, logits, label, footprint, support,
                 delta, base, sens_patch) -> dict:
        """
        Every term separately, plus 'total'. Separately because a run whose
        regulariser is three orders of magnitude above its adversarial terms
        has not converged to an invisible patch, it has converged to the clean
        image — and the only way to see that is to log the parts. D3 is exactly
        that failure waiting at a patch size the paper never ran.
        """
        out = {}
        total = logits.new_zeros(())
        if self._untar is not None:
            # EVERY loss adversarial.build returns takes the same four slots,
            # f(logits, labels, footprint, support), and the margin branch
            # ignores the labels slot because it closed over margin_ref at
            # build time. So `label` is passed unconditionally and the
            # margin/cos_margin pair still scores against the clean
            # prediction — there is nothing to substitute here.
            lu = self._untar(logits, label, footprint, support)
            out["untar"] = float(lu)
            total = total + self.w2 * lu
        if self._tar is not None:
            lt = self._tar(logits, label, footprint, support)
            out["tar"] = float(lt)
            total = total + self.w1 * lt
        r = regulariser(delta, base, sens_patch, self.reg_reduce)
        out["reg"] = float(r)
        # D9. Detached, the term has the SAME VALUE and a ZERO gradient, which
        # is exactly what the released code's two cancelling paths produce. The
        # reported `reg` is the undetached value either way, so the two modes
        # are comparable on the column the paper calls "human distance".
        total = total + self.w3 * (r.detach() if self.reg_grad == "released"
                                   else r)
        out["total"] = total
        return out


# ═════════════════════════════════════════════════════════════════════════════
#  the early stop
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def progress_rate(clean_logits, adv_logits, label, footprint,
                  criterion: str = "target",
                  target_class: Optional[int] = None,
                  num_classes: int = 19,
                  clean_remote: Optional[float] = None) -> float:
    r"""
    One number in the unit the stop threshold is quoted in. See D5.

    'flip'   fraction of REMOTE pixels whose argmax changed      [0,1]
    'target' fraction of non-target remote pixels driven to it   [0,1]
    'drop'   remote mIoU points lost                             [0,100]
    'none'   0.0, i.e. never stops

    All three come from the metrics this thesis already reports —
    metrics/miou.attack_rates and single_image_miou — so a run that stopped
    early stopped on a quantity that appears in the results table, not on a
    private criterion invented for the stop.

    NOTE THE ASYMMETRY WITH CLASSIFICATION, which is D5's whole content: one
    label flipping is a success there, while 'target' at 0.9 asks for NINE
    TENTHS of a megapixel frame to be converted. These are not comparable
    bars, and an IAP run that never early-stops is not therefore a failed
    attack.
    """
    if criterion not in STOP_CRITERIA:
        raise ValueError(f"criterion must be one of {STOP_CRITERIA}, "
                         f"got {criterion!r}")
    if criterion == "none":
        return 0.0
    if criterion == "drop":
        if clean_remote is None:
            raise ValueError("criterion='drop' needs the clean remote mIoU")
        return clean_remote - single_image_miou(
            adv_logits, label, num_classes, exclude=footprint)
    if criterion == "target" and target_class is None:
        raise ValueError("criterion='target' needs a target_class")
    r = attack_rates(clean_logits, adv_logits, label, footprint,
                     target_class if criterion == "target" else None)
    key = "any_flip_rate" if criterion == "flip" else "target_hit_rate"
    return r[key] / 100.0


# ═════════════════════════════════════════════════════════════════════════════
#  invisibility
# ═════════════════════════════════════════════════════════════════════════════

def ssim_psnr(a: torch.Tensor, b: torch.Tensor,
              data_range: float = 1.0) -> Tuple[float, float]:
    """
    (SSIM, PSNR) for one [1,3,h,w] pair in [0,1], via torchmetrics with
    data_range=1.0 — the reference's `SSIM(data_range=1.0)` and
    `PSNR(data_range=1.0)`.

    torchmetrics rather than a local implementation because the reference uses
    it and SSIM has enough free parameters (kernel size, sigma, which channels,
    whether the mean is over windows or pixels) that two "SSIM"s routinely
    differ in the second decimal. The paper's broader invisibility suite —
    LPIPS, CLIPScore, UIQ, SRE — is NOT computed here: scripts/lpips_eval.py
    already owns perceptual scoring in its own environment, and iap_attack.py
    writes the a_clean.png / c_patched.png pair that script discovers. One
    implementation of LPIPS in this repository, not two.
    """
    try:
        from torchmetrics.image import (PeakSignalNoiseRatio,
                                        StructuralSimilarityIndexMeasure)
    except ImportError as e:                               # pragma: no cover
        raise ImportError(
            "IAP's invisibility metrics need torchmetrics (it is pinned in "
            "requirements.txt and in envs/iap.yml). "
            "pip install torchmetrics==0.11.4") from e
    dev = a.device
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(dev)
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(dev)
    return float(ssim(a, b)), float(psnr(a, b))


def invisibility(clean01: torch.Tensor, adv01: torch.Tensor,
                 box: Tuple[int, int, int]) -> dict:
    """
    SSIM and PSNR on the FOOTPRINT and on the WHOLE FRAME, which is the
    local/global split the paper reports every invisibility metric under.

    Both are needed and neither alone is honest. The frame figures are spatial
    means, so a 256px patch in a 1024x2048 frame is diluted 32x and any patch
    looks invisible; the footprint figures ignore a visible seam against the
    surround. The reference reports exactly this pair (ssim_patches /
    ssim_images).

    box : (top, left, p).
    """
    top, left, p = box
    cp = clean01[:, :, top:top + p, left:left + p]
    ap = adv01[:, :, top:top + p, left:left + p]
    s_loc, p_loc = ssim_psnr(cp, ap)
    s_glb, p_glb = ssim_psnr(clean01, adv01)
    return {"ssim_patch": s_loc, "psnr_patch": p_loc,
            "ssim_image": s_glb, "psnr_image": p_glb}


# ═════════════════════════════════════════════════════════════════════════════
#  the procedure
# ═════════════════════════════════════════════════════════════════════════════

def prepare(model, img, patch, *, mean_t, std_t, label, cam,
            score: str = "cam", margin: int = 0, log=print) -> dict:
    r"""
    Everything IAP does BEFORE its first optimisation step, in the one order
    that works:

        denormalise -> sensitivity map -> Grad-CAM -> localise -> init delta

    ORDERING IS LOAD-BEARING, exactly as it is in optimise.prepare(). The CAM
    needs the model to have seen the image; localisation needs the CAM and the
    sensitivity map; `base` is the image region AT the resolved location, so it
    cannot be read before the location exists. Done backwards, the patch
    initialises from the wrong region and every invisibility number is measured
    against content the patch never replaced.

    THE PIXEL SCALE. CityscapesSeg returns mean/std NORMALISED images, and
    every one of IAP's three mechanisms is defined on [0,1] pixels: eps = 1e-3
    in the sensitivity map, the multiplicative `* delta` in the update, and the
    [0,1] projection. So the frame is denormalised once here, the patch lives
    in [0,1] like every other pixel-mode patch in this repository, and
    Patch.apply re-normalises it on the way into the model.

    Returns the dict the attack loop needs, and leaves `patch` holding its
    placement and its initial parameter.
    """
    H, W = img.shape[-2:]
    p = patch.side(H, W)
    if p != patch.cfg.size:
        raise ValueError(
            f"IAP needs the patch pasted at its parameter resolution: "
            f"side(H,W) = {p} but --patch_size = {patch.cfg.size}. Any "
            f"resampling between the two interpolates the perturbation, so "
            f"the sensitivity map, the regulariser and the update rule would "
            f"all be computed on a different signal from the one pasted. At "
            f"{H}x{W} with --patch_scale {patch.cfg.scale:g} that means "
            f"--patch_size {p}.")
    if patch.cfg.mode != "raw" or patch.cfg.pixel_param != "direct":
        raise ValueError(
            f"IAP is --patch_mode raw --pixel_param direct (the parameter IS "
            f"the pixel), got mode={patch.cfg.mode!r} "
            f"pixel_param={patch.cfg.pixel_param!r}. Under a sigmoid "
            f"parameterisation the update's `* delta` factor multiplies by a "
            f"LOGIT, which is negative below 0.5 and flips the sign of the "
            f"step for half the patch.")

    x01 = (img * std_t + mean_t).clamp(0.0, 1.0)
    sens_full = sensitivity(x01)                                # [1,H,W]

    if score == "center":
        # The localisation ablation. No CAM is built, so no backward pass is
        # paid for a map nothing reads — and `cam` is allowed to be None,
        # rather than being built and ignored, so a run cannot claim a CAM it
        # never computed.
        with torch.no_grad():
            clean_logits = upsample_to(model(img), label.shape[-2:])
        cam_map = None
    else:
        if cam is None:
            raise ValueError(
                f"score={score!r} needs a SegmentationCAM. Build one with "
                f"segmentation_cam.build(model, ...), or pass score='center' "
                f"for the localisation ablation. Centring silently here would "
                f"report an IAP-localised run that was never localised.")
        # The CAM's forward pass returns the clean logits too, so the clean
        # pass costs ONE backward rather than an extra forward.
        cam_map, clean_logits = cam(img, label)
        clean_logits = upsample_to(clean_logits, label.shape[-2:])
    top, left = localise(None if cam_map is None else cam_map[0, 0],
                         sens_full[0], p, score=score, margin=margin)

    # RECORDED AS 'fixed', with the window's own centre, so that reloading the
    # checkpoint re-resolves to exactly this window instead of silently
    # centring. What CHOSE the window is recorded separately as iap_locate in
    # the result — placement_policy alone would under-report it.
    patch.cfg.placement = "fixed"
    patch.cfg.placement_xy = ((top + p / 2) / H, (left + p / 2) / W)
    patch.placement = (int(top), int(left))

    base = x01[0, :, top:top + p, left:left + p].clone()        # [3,p,p]
    sens_patch = sens_full[0, top:top + p, left:left + p].clone()   # [p,p]
    with torch.no_grad():
        patch.param.data.copy_(base)

    ctop, cleft = (H - p) // 2, (W - p) // 2
    log(f"[iap  ] localise  : score={score} -> (top,left)=({top},{left}), "
        f"{((top - ctop) ** 2 + (left - cleft) ** 2) ** 0.5:.0f}px from centre")
    log(f"[iap  ] sens patch: min {float(sens_patch.min()):.1f}  "
        f"median {float(sens_patch.median()):.1f}  "
        f"max {float(sens_patch.max()):.1f}   (ceiling 1/eps = "
        f"{1.0 / SENS_EPS:.0f}, a perfectly flat neighbourhood)")
    log(f"[iap  ] base      : the image region itself, {p}x{p} at "
        f"({top},{left})")
    return {"clean_logits": clean_logits, "x01": x01,
            "sens": sens_full, "sens_patch": sens_patch, "base": base,
            "box": (int(top), int(left), int(p))}


def _rung(model, img, label, patch, prepared, objective, lr, steps, *,
          stop, stop_thresh, target_class, num_classes, exclude_footprint,
          classes, clamp, clean_remote, log_every, verbose, log) -> dict:
    r"""
    ONE learning rate, from a fresh delta — the reference's inner
    `for t in range(iterations)` loop.

    `best_param` is kept in MEMORY, not written to disk: the ladder does not
    know which rung wins until it has run them all, and a per-rung best.pt in
    one directory would be overwritten by a worse rung. The caller writes the
    winner once.
    """
    hw = label.shape[-2:]
    base, sens_patch = prepared["base"], prepared["sens_patch"]
    clean_logits = prepared["clean_logits"]

    with torch.no_grad():
        patch.param.data.copy_(base)         # the reference re-inits per rung
    best_drop, best_param, history = -1e9, patch.param.detach().clone(), []
    stopped_at, last = None, {}

    for step in range(1, steps + 1):
        # `step` is 1-BASED here, so step-1 is the 0-based progress counter a
        # schedule is defined on — the same convention optimise.attack_image
        # uses, so a stateful loss cannot mean one thing in one loop and
        # another in the other.
        objective.on_step_begin(step - 1, steps)
        if patch.param.grad is not None:
            patch.param.grad = None
        patched, fp = patch.apply(img)
        logits = upsample_to(model(patched), hw)
        sup = (~fp) if exclude_footprint else None

        parts = objective(logits, label, fp, sup,
                          patch.render(), base, sens_patch)
        total = parts["total"]
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite IAP loss at step {step}: {parts}")

        # THE STOP IS CHECKED BEFORE THE UPDATE, which is the reference's own
        # ordering (its `if pred.max(...) == target: break` sits above
        # `loss.backward()`), so a run that stops reports the patch that
        # satisfied the criterion rather than one step past it.
        rate = progress_rate(clean_logits, logits.detach(), label, fp,
                             criterion=stop, target_class=target_class,
                             num_classes=num_classes,
                             clean_remote=clean_remote)
        last = {k: float(v) for k, v in parts.items() if k != "total"}
        last["total"] = float(total)
        last["rate"] = rate
        if stop != "none" and rate >= stop_thresh:
            stopped_at = step
            # THE STOPPING PATCH IS THE RESULT, overriding best-by-drop. The
            # criterion IS the success definition for this rung, so reporting
            # a different (earlier, higher-drop) patch would leave
            # iap_stopped_at naming a step whose patch is not the one saved —
            # the exact clean/attacked mismatch the 2026-10-05 best.pt entry
            # exists to prevent, one level up. The break happens BEFORE the
            # update, so param is still the patch that met the bar.
            best_param = patch.param.detach().clone()
            best_drop = clean_remote - single_image_miou(
                logits.detach(), label, num_classes, exclude=fp,
                classes=classes)
            if verbose:
                log(f"  step {step:4d}  STOP: {stop} = {rate:.4f} "
                    f">= {stop_thresh:g}   (this patch is the rung's result, "
                    f"drop {best_drop:+.2f})")
            break

        total.backward()
        st = colour_constant_step_(patch.param, sens_patch, lr)
        if clamp:
            patch.project()                   # [0,1] — PGD's own projection

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                cur = single_image_miou(logits.detach(), label, num_classes,
                                        exclude=fp, classes=classes)
                drift = float((patch.render() - base).abs().mean())
            row = {"step": step, "miou_remote": cur, "drift_mean_abs": drift,
                   **last, **st}
            history.append(row)
            if verbose:
                log(f"  step {step:4d}  remote={cur:6.2f} "
                    f"({clean_remote - cur:+6.2f})  "
                    + " ".join(f"{k}={row[k]:.4g}" for k in
                               ("tar", "untar", "reg") if k in row)
                    + f"  |d|={drift:.4f}  {stop}={row['rate']:.3f}")
            if clean_remote - cur > best_drop:
                best_drop = clean_remote - cur
                best_param = patch.param.detach().clone()

    # The rung's END state, measured once, outside the loop.
    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), hw)
        final_rem = single_image_miou(adv_logits, label, num_classes,
                                      exclude=fp, classes=classes)
    final_param = patch.param.detach().clone()
    if clean_remote - final_rem > best_drop:      # a rung that never logged
        best_drop, best_param = clean_remote - final_rem, final_param.clone()

    return {"lr": float(lr), "steps_run": stopped_at or steps,
            "stopped_at": stopped_at,
            "final_remote": final_rem,
            "drop_remote": clean_remote - final_rem,
            "best_drop_remote": best_drop,
            "last": last, "history": history,
            "best_param": best_param, "final_param": final_param}


def attack_image(model, img, label, patch, *, mean_t, std_t, cam,
                 target_class: int = 8,
                 tar_loss: str = "ipatch_cospgd",
                 untar_loss: str = "ce",
                 w1: float = IAP_W1, w2: float = IAP_W2, w3: float = IAP_W3,
                 reg_reduce: str = "sum", reg_grad: str = "live",
                 locate: str = "cam",
                 lrs: Sequence[float] = IAP_LRS,
                 steps: int = IAP_STEPS,
                 stop: str = "target", stop_thresh: float = 0.9,
                 margin_kappa: float = 5.0,
                 placement_margin: int = 0,
                 num_classes: int = 19,
                 exclude_footprint: bool = True,
                 classes: str = "gt",
                 clamp: bool = True,
                 log_every: int = 50,
                 out_dir=None,
                 verbose: bool = True,
                 log=print) -> dict:
    r"""
    The whole IAP procedure against ONE image. Returns the result record.

    The RECORD SCHEMA deliberately matches optimise.attack_image's for every
    key the two share — clean/final/drop, remote and all, the attack rates, the
    placement block — so analysis/build_index.py indexes an IAP run beside a
    csf run with no special case, and a table can put them in adjacent rows.
    Everything IAP-specific is prefixed `iap_`.

    THE LADDER. `lrs` is tried in order and stops at the first rung meeting the
    stop criterion; absent a success every rung runs and the one with the
    largest best remote drop is the headline. See D4 — the released code's
    failure branch reports stale values from the previous rung, and that cannot
    simply be ported.

    `patch` is left holding the WINNING rung's best parameter, so the caller's
    diagnostics, PNGs and invisibility numbers all describe the patch the
    record reports. That is the 2026-10-05 convention for the overfit block
    (best.pt, not final.pt) applied to a ladder.
    """
    if stop not in STOP_CRITERIA:
        raise ValueError(f"stop must be one of {STOP_CRITERIA}, got {stop!r}")
    if not len(lrs):
        raise ValueError("the lr ladder is empty")
    hw = label.shape[-2:]

    prepared = prepare(model, img, patch, mean_t=mean_t, std_t=std_t,
                       label=label, cam=cam, score=locate,
                       margin=placement_margin, log=log)
    clean_logits = prepared["clean_logits"]
    _, fp0 = patch.apply(img)
    clean_all = single_image_miou(clean_logits, label, num_classes,
                                  classes=classes)
    clean_rem = single_image_miou(clean_logits, label, num_classes,
                                  exclude=fp0, classes=classes)

    # margin/cos_margin score against the CLEAN PREDICTION, which only exists
    # once the clean forward has run. Void is set to 255 so the loss excludes
    # exactly the pixels any_flip_rate excludes. Same construction as
    # optimise.attack_image.
    margin_ref = None
    if untar_loss in ("margin", "cos_margin"):
        margin_ref = clean_logits.argmax(1)
        margin_ref[label == 255] = 255

    objective = Objective(tar=tar_loss, untar=untar_loss,
                          target_class=target_class, w1=w1, w2=w2, w3=w3,
                          reg_reduce=reg_reduce, reg_grad=reg_grad,
                          margin_ref=margin_ref, margin_kappa=margin_kappa)
    if verbose:
        log(f"[clean] all {clean_all:.2f} | remote {clean_rem:.2f}  "
            f"<- remote is the baseline that matters")
        log(f"[iap  ] loss      : {objective!r}")
        if reg_grad == "released":
            log(f"[iap  ] reg grad  : RELEASED — the regulariser is reported "
                f"and optimises NOTHING (D9).\n"
                f"                   w3 cannot affect this run; that is the "
                f"point of the arm.")
        log(f"[iap  ] ladder    : lr {', '.join(f'{x:g}' for x in lrs)}  "
            f"x {steps} steps, stop on {stop} >= {stop_thresh:g}")
        if reg_reduce == "sum":
            p = prepared["box"][2]
            log(f"[iap  ] reg scale : sum over {p}x{p}x3 elements, "
                f"{(p / REF_PATCH_SIDE) ** 2:.3g}x the element count the "
                f"published w3={IAP_W3:g} was tuned at (D3). --iap_reg_reduce "
                f"sum84 rescales it.")

    rungs = []
    for i, lr in enumerate(lrs):
        if verbose:
            log(f"\n{'-' * 66}\n  rung {i + 1}/{len(lrs)}   lr {lr:g}"
                f"\n{'-' * 66}")
        r = _rung(model, img, label, patch, prepared, objective, lr, steps,
                  stop=stop, stop_thresh=stop_thresh,
                  target_class=target_class, num_classes=num_classes,
                  exclude_footprint=exclude_footprint, classes=classes,
                  clamp=clamp, clean_remote=clean_rem, log_every=log_every,
                  verbose=verbose, log=log)
        rungs.append(r)
        if r["stopped_at"] is not None:
            break

    # The winner is tracked by INDEX, not by identity. rungs.index(win) would
    # compare dicts element-wise, and these dicts hold tensors, so the
    # comparison raises on the first `best_param`.
    i_win = next((i for i, r in enumerate(rungs)
                  if r["stopped_at"] is not None), None)
    if i_win is None:
        i_win = max(range(len(rungs)),
                    key=lambda i: rungs[i]["best_drop_remote"])
    win = rungs[i_win]
    with torch.no_grad():
        patch.param.data.copy_(win["best_param"])

    # ── the winner's own measurement, on the parameter now loaded ────────────
    with torch.no_grad():
        patched, fp = patch.apply(img)
        adv_logits = upsample_to(model(patched), hw)
        final_all = single_image_miou(adv_logits, label, num_classes,
                                      classes=classes)
        final_rem = single_image_miou(adv_logits, label, num_classes,
                                      exclude=fp, classes=classes)
        rates = attack_rates(clean_logits, adv_logits, label, fp,
                             target_class if tar_loss != "none" else None)
        adv01 = (patched * std_t + mean_t).clamp(0.0, 1.0)
        vis = invisibility(prepared["x01"], adv01, prepared["box"])
        reg_final = float(regulariser(patch.render(), prepared["base"],
                                      prepared["sens_patch"], reg_reduce))

    if out_dir is not None:
        from pathlib import Path
        from torchvision.utils import save_image
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        patch.save(out_dir / "best.pt")
        save_image(patch.render().detach().cpu(), out_dir / "best_patch.png")
        # final.pt is the WINNING RUNG's last parameter, not the best one, so
        # the pair stays interpretable: lpips_eval re-renders both and says
        # which one a panel PNG shows.
        with torch.no_grad():
            keep = patch.param.detach().clone()
            patch.param.data.copy_(win["final_param"])
            patch.save(out_dir / "final.pt")
            save_image(patch.render().detach().cpu(),
                       out_dir / "final_patch.png")
            patch.param.data.copy_(keep)
        # The sensitivity map is the one input to IAP that is not recoverable
        # from the checkpoint, and every localisation and invisibility claim
        # rests on it. 8-bit, inverted-log scaled for viewing only.
        s = prepared["sens_patch"]
        save_image((s / s.max()).detach().cpu().unsqueeze(0),
                   out_dir / "sensitivity_patch.png")

    top, left, p = prepared["box"]
    ctop, cleft = (hw[0] - p) // 2, (hw[1] - p) // 2
    H, W = hw
    record = {
        # ── the shared schema (see optimise.attack_image) ────────────────────
        "clean_all": clean_all, "clean_remote": clean_rem,
        "final_all": final_all, "final_remote": final_rem,
        "drop_all": clean_all - final_all,
        "drop_remote": clean_rem - final_rem,
        "best_drop_remote": win["best_drop_remote"],
        "last_loss": win["last"].get("total"),
        "placement": [top, left],
        "placement_policy": "fixed",
        "placement_dist_from_centre": float(
            ((top - ctop) ** 2 + (left - cleft) ** 2) ** 0.5),
        "placement_on_border": bool(top in (0, H - p) or left in (0, W - p)),
        "classes": classes,
        "optimiser": "iap_colour_constant",
        "pixel_param": patch.cfg.pixel_param,
        "lr_schedule": "iap_ladder",
        **rates,
        # ── IAP's own ────────────────────────────────────────────────────────
        "iap_locate": locate,
        "iap_lr": win["lr"],
        "iap_lr_ladder": [float(x) for x in lrs],
        "iap_rung_index": i_win,
        "iap_stopped_at": win["stopped_at"],
        "iap_stop": stop, "iap_stop_thresh": stop_thresh,
        "iap_steps_run": sum(r["steps_run"] for r in rungs),
        "iap_w1": w1, "iap_w2": w2, "iap_w3": w3,
        "iap_reg_reduce": reg_reduce, "iap_reg_grad": reg_grad,
        "iap_tar_loss": tar_loss, "iap_untar_loss": untar_loss,
        "iap_clamp": bool(clamp),
        "iap_reg_final": reg_final,
        "iap_sens_patch_median": float(prepared["sens_patch"].median()),
        "iap_drift_mean_abs": float(
            (patch.render() - prepared["base"]).abs().mean()),
        "iap_drift_absmax": float(
            (patch.render() - prepared["base"]).abs().max()),
        **vis,
        "iap_rungs": [{k: v for k, v in r.items()
                       if k not in ("best_param", "final_param")}
                      for r in rungs],
        "history": win["history"],
    }
    if verbose:
        # Built OUTSIDE the f-string: an expression spanning two lines inside
        # braces is a SyntaxError before Python 3.12, and the cluster is on 3.9.
        why = (f"stopped at step {win['stopped_at']}"
               if win["stopped_at"] is not None else "ran out, best by drop")
        log(f"\n{'=' * 66}")
        log(f"  winning rung      : lr {win['lr']:g} ({why})")
        log(f"  clean  all/remote : {clean_all:.2f} / {clean_rem:.2f}")
        log(f"  final  all/remote : {final_all:.2f} / {final_rem:.2f}")
        log(f"  drop   REMOTE     : {clean_rem - final_rem:+.2f}   <- "
            f"the result")
        for k in ("any_flip_rate", "target_hit_rate"):
            if k in rates:
                log(f"  {k:18s}: {rates[k]:.1f}%")
        log(f"  SSIM patch/image  : {vis['ssim_patch']:.4f} / "
            f"{vis['ssim_image']:.4f}")
        log(f"  PSNR patch/image  : {vis['psnr_patch']:.2f} / "
            f"{vis['psnr_image']:.2f} dB")
        log(f"{'=' * 66}")
    return record
