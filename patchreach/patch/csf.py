r"""
Human contrast sensitivity in the frequency domain.

WHAT THIS IS FOR
----------------
Every realism constraint in the repository so far is a PIXEL-SPACE statistic:
LAP's L_rat pulls toward a reference, L_tv penalises roughness, ASI/AGI/ADE
score texture regularity. None of them knows anything about the observer.

This module supplies the missing piece: a model of WHICH SPATIAL FREQUENCIES A
HUMAN CAN ACTUALLY SEE. That turns "make the patch look plausible" into a
falsifiable statement — a perturbation is invisible if its contrast at every
frequency sits below the detection threshold at that frequency.

The pay-off is an asymmetry worth attacking through: a segmentation network
has no contrast sensitivity function. It reads the whole band up to Nyquist at
full weight. If the network's decision is influenced by frequencies the eye
discards, an attack can live entirely inside that gap.

WHETHER THAT GAP EXISTS FOR THIS MODEL IS AN EXPERIMENT, NOT AN ASSUMPTION.
SegFormer's patch embedding has stride 4, so the very highest frequencies may
be attenuated before the first attention block ever sees them. Measuring that
is the point.

SOURCES
-------
Barten CSF, parameters and the cycles/pixel -> cycles/degree conversion follow
Galinska, Pogodzinski & Lenssen, "CSFlow: Aligning Flow Matching with Human
Contrast Sensitivity" (Appendices A and D), which in turn uses Barten (2003).
The Standard Spatial Observer is Watson & Ramirez (ModelFest, OSA 1999), with
the oblique effect; implementation follows Qiang Li's CSF toolbox.

UNITS ARE THE WHOLE GAME
------------------------
A CSF is defined over CYCLES PER DEGREE OF VISUAL ANGLE. An image spectrum is
in CYCLES PER PIXEL. The conversion depends on the physical pixel size and the
viewing distance, so EVERY claim about visibility is conditional on an assumed
viewing geometry. State it; never let it stay implicit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch

CSF_MODELS = ("barten", "sso")


# ═════════════════════════════════════════════════════════════════════════════
#  Viewing geometry
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ViewingGeometry:
    r"""
    The observer assumption that makes a CSF meaningful.

        theta_pix = 2 * arctan( p / 2d ) * 180/pi      degrees per pixel
        f_cpd     = f_cpp / theta_pix                  cycles per degree

    pixel_size_cm       physical size of one displayed pixel
    viewing_distance_cm observer distance to the display

    DEFAULTS reproduce the CSFlow paper (Appendix D): a 0.0114 cm pixel viewed
    from 50 cm, i.e. a normal laptop screen. They are NOT the driving threat
    model — a physical patch seen by a driver has completely different
    geometry, and the same perturbation would be far more or far less visible.
    Sweep this rather than trusting one setting.
    """
    pixel_size_cm: float = 0.0114
    viewing_distance_cm: float = 50.0

    @property
    def degrees_per_pixel(self) -> float:
        return (2.0 * math.atan(self.pixel_size_cm
                                / (2.0 * self.viewing_distance_cm))
                * 180.0 / math.pi)

    def to_cycles_per_degree(self, f_cpp: torch.Tensor) -> torch.Tensor:
        """cycles/pixel -> cycles/degree."""
        return f_cpp / self.degrees_per_pixel

    @property
    def nyquist_cpd(self) -> float:
        """
        The highest frequency the sampling grid can represent, in cpd.

        Worth printing: at the default geometry this is ~38 cpd, BELOW the
        ~50-60 cpd where human sensitivity actually vanishes. So the image
        cannot represent a truly invisible frequency — the budget at Nyquist
        is large but finite, and "invisible" always means "below threshold",
        never "outside the passband".
        """
        return 0.5 / self.degrees_per_pixel

    def describe(self, log=print):
        log(f"[csf ] geometry  : pixel {self.pixel_size_cm} cm at "
            f"{self.viewing_distance_cm} cm -> "
            f"{self.degrees_per_pixel:.5f} deg/px")
        log(f"[csf ] Nyquist   : {self.nyquist_cpd:.1f} cpd "
            f"(human cutoff is ~50-60 cpd, so the grid does not reach it)")


# ═════════════════════════════════════════════════════════════════════════════
#  Barten (2003)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BartenParams:
    """CSFlow Appendix A, Eqs 11-14. Do not change without saying why."""
    sigma: float = 0.5 / 60.0     # deg, optical point spread
    k: float = 3.0                # signal-to-noise ratio required
    T: float = 0.1                # s, integration time of the eye
    X_0: float = 60.0             # deg, object size
    X_max: float = 12.0           # deg, maximum integration area
    N_max: float = 15.0           # cycles, maximum number integrated
    n: float = 0.03               # quantum efficiency
    p: float = 1.2e6              # photons/(s deg^2 Td), conversion
    E: float = 500.0              # Td, retinal illuminance
    phi_0: float = 3e-8           # deg^2 s, neural noise
    u_0: float = 7.0              # cpd, lateral inhibition cutoff


def optical_mtf(f_cpd: torch.Tensor, sigma: float) -> torch.Tensor:
    """M_opt(f) = exp(-2 pi^2 sigma^2 f^2)   — CSFlow Eq 10."""
    return torch.exp(-2.0 * math.pi ** 2 * sigma ** 2 * f_cpd ** 2)


def barten_csf(f_cpd: torch.Tensor,
               params: BartenParams = BartenParams(),
               eps: float = 1e-12) -> torch.Tensor:
    r"""
    Contrast sensitivity at `f_cpd` cycles/degree — CSFlow Eq 9.

                     M_opt(f)  [ 2 (  1     1      f^2  ) (   1        phi_0        ) ]^-1/2
        CSF(f)  =    -------- *[ - ( ---- + ---- + ---- ) ( ----- + -------------- ) ]
                        k      [ T ( X_0^2  Xmax^2 Nmax^2) (  npE    1-exp(-(f/u0)^2)) ]

    Sensitivity is the RECIPROCAL of the threshold contrast: CSF(f) = 700 means
    a grating of contrast 1/700 is just detectable at that frequency.

    AT DC the lateral-inhibition term 1 - exp(-(f/u_0)^2) vanishes, the bracket
    diverges and CSF -> 0. That is physically correct — a uniform field has no
    contrast to detect — and it is handled explicitly rather than left to
    produce a NaN that would silently poison every downstream budget.
    """
    f = f_cpd.clamp(min=0.0)
    inhib = 1.0 - torch.exp(-((f / params.u_0) ** 2))

    integ = (2.0 / params.T) * (1.0 / params.X_0 ** 2
                                + 1.0 / params.X_max ** 2
                                + f ** 2 / params.N_max ** 2)
    noise = (1.0 / (params.n * params.p * params.E)
             + params.phi_0 / inhib.clamp(min=eps))

    csf = (optical_mtf(f, params.sigma) / params.k) / (integ * noise).sqrt()
    return torch.where(f > 0, csf, torch.zeros_like(csf))


# ═════════════════════════════════════════════════════════════════════════════
#  Standard Spatial Observer (Watson & Ramirez 1999)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SSOParams:
    g: float = 330.74             # overall gain
    fm: float = 7.28              # exponential decay
    l: float = 0.837              # low-frequency loss
    s: float = 1.809              # attenuation of that loss at high f
    w: float = 1.0                # oblique-effect weight; 0 disables
    os: float = 6.664             # oblique-effect scale


def sso_csf(fy_cpd: torch.Tensor, fx_cpd: torch.Tensor,
            params: SSOParams = SSOParams(),
            eps: float = 1e-12) -> torch.Tensor:
    r"""
        CSF_SSO(fx,fy) = CSF_Tyler(f) * OE(fx,fy)

        CSF_Tyler(f) = g ( exp(-f/fm) - l exp(-f^2/s^2) )
        OE(fx,fy)    = 1 - w ( 4 (1 - exp(-f/os)) fx^2 fy^2 ) / f^4

    ORIENTATION-DEPENDENT, which Barten is not. The oblique effect says
    diagonal gratings are harder to see than horizontal or vertical ones at the
    same radial frequency — OE is 1 on the axes and dips to its minimum at 45
    degrees. As a hiding place that is a real, free asymmetry: a diagonal
    carrier is less visible than an axis-aligned one of identical amplitude.

    Kept as an alternative model so the visibility claim can be shown not to
    depend on one particular CSF. If barten and sso disagree about a
    perturbation, that disagreement belongs in the writeup.
    """
    f2 = fy_cpd ** 2 + fx_cpd ** 2
    f = f2.clamp(min=eps).sqrt()

    tyler = params.g * (torch.exp(-f / params.fm)
                        - params.l * torch.exp(-f2 / params.s ** 2))
    oe = 1.0 - params.w * (4.0 * (1.0 - torch.exp(-f / params.os))
                           * fx_cpd ** 2 * fy_cpd ** 2) / f2.clamp(min=eps) ** 2

    csf = tyler * oe
    return torch.where(f2 > eps, csf, torch.zeros_like(csf)).clamp(min=0.0)


# ═════════════════════════════════════════════════════════════════════════════
#  Frequency grids
# ═════════════════════════════════════════════════════════════════════════════

def rfft_frequency_grid(H: int, W: int, device=None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (fy, fx) in cycles/pixel on the rfft2 half-spectrum, shape [H, W//2+1].

    rfft2 is used throughout rather than fft2: the patch is real, so the
    negative-frequency half is redundant, and a budget applied to the full
    spectrum would have to keep the two halves conjugate by hand.
    """
    fy = torch.fft.fftfreq(H, device=device).view(H, 1)
    fx = torch.fft.rfftfreq(W, device=device).view(1, -1)
    return fy.expand(H, fx.shape[-1]), fx.expand(H, fx.shape[-1])


def radial_frequency_cpp(H: int, W: int, device=None) -> torch.Tensor:
    """|f| in cycles/pixel on the rfft2 grid. Max is 0.5 (Nyquist)."""
    fy, fx = rfft_frequency_grid(H, W, device)
    return (fy ** 2 + fx ** 2).sqrt()


def csf_map(H: int, W: int,
            geometry: ViewingGeometry = ViewingGeometry(),
            model: str = "barten",
            device=None) -> torch.Tensor:
    """
    CSF evaluated at every rfft2 bin of an H x W image. Shape [H, W//2+1].

    'barten' is isotropic and depends only on |f|; 'sso' additionally applies
    the oblique effect and therefore varies with orientation.
    """
    if model not in CSF_MODELS:
        raise ValueError(f"csf model must be one of {CSF_MODELS}, got {model!r}")
    fy, fx = rfft_frequency_grid(H, W, device)
    if model == "barten":
        return barten_csf(geometry.to_cycles_per_degree(
            (fy ** 2 + fx ** 2).sqrt()))
    return sso_csf(geometry.to_cycles_per_degree(fy),
                   geometry.to_cycles_per_degree(fx))


# ═════════════════════════════════════════════════════════════════════════════
#  The budget
# ═════════════════════════════════════════════════════════════════════════════

# A DFT magnitude is not a Michelson contrast, and the gap is a constant that
# must be written down rather than absorbed silently.
#
#   a real cosine of peak amplitude A on mean mu has Michelson contrast A/mu
#   with norm="forward", rfft2 gives that component magnitude A/2
#   taking mu = 0.5 (mid grey) gives  contrast = 4 * |rfft magnitude|
#
# So detectability is 4*|d(f)|*CSF(f) > 1. The mean is assumed to be 0.5
# rather than measured locally, which is a simplification: a patch sitting on
# dark asphalt has a lower local mean and therefore HIGHER contrast for the
# same amplitude, so this under-estimates visibility there. tau absorbs the
# constant, but never report tau=1 as literally "one JND" without this caveat.
CONTRAST_SCALE = 4.0


def amplitude_budget(csf: torch.Tensor, threshold: float = 1.0,
                     max_amplitude: float = 1.0,
                     contrast_scale: float = CONTRAST_SCALE,
                     eps: float = 1e-8) -> torch.Tensor:
    r"""
    Per-frequency budget on the rfft magnitude:

        B(f) = min( tau / (contrast_scale * CSF(f)) , A_max )

    A component of contrast c at frequency f is detectable when c * CSF(f) > 1,
    so this keeps every component at or below `tau` just-noticeable
    differences. tau is THE knob of this attack family and it is exactly
    analogous to LAP's alpha: tau -> 0 gives an invisible, powerless patch,
    tau -> large gives a visible, powerful one, and the curve between them is
    the deliverable.

    `max_amplitude` caps the budget where CSF is near zero (DC, and beyond the
    optical cutoff), which would otherwise be unbounded.
    """
    return (threshold / (contrast_scale * csf.clamp(min=eps))
            ).clamp(max=max_amplitude)


def patch_budget(size: int, geometry: ViewingGeometry = ViewingGeometry(),
                 model: str = "barten", threshold: float = 1.0,
                 min_cycles: float = 2.0, max_amplitude: float = 0.25,
                 device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    (csf_values, budget) for a size x size PATCH, with the low-frequency band
    removed. Returns both because every downstream call needs both.

    WHY THE LOW BAND MUST GO — this is the correction that makes a full-field
    CSF applicable to a patch at all.

    The CSF falls to zero at DC, so 1/CSF diverges there and the naive budget
    hands the near-DC bins the LARGEST allowance of any frequency. Measured: a
    residual built that way put 3e-5 of its power in the lowest radial bin and
    1e-9 in the highest — the opposite of the intent, and dominated by what is
    effectively a uniform brightness offset.

    That is not a bug in the CSF; it is a category error in applying it. The
    low-frequency rolloff is measured for FULL-FIELD gratings, where the eye
    adapts to a slow luminance ramp. Inside a patch of side S there is no such
    grating: a component below 1/S cycles per pixel does not complete a single
    cycle, so it is not a grating but a brightness offset — and a patch whose
    mean differs from its surroundings is a visible rectangle no matter what
    the CSF says about DC. The patch border turns any offset into an edge, and
    edges are broadband.

    `min_cycles` therefore sets the lowest frequency allowed to carry energy,
    in cycles across the patch. 2.0 means "at least two full cycles inside the
    patch". Setting it to 0 restores the naive behaviour and is useful only to
    reproduce the failure.

    `max_amplitude` also drops to 0.25 here, since a quarter of the dynamic
    range in a single component is already far beyond anything the constraint
    is supposed to permit.
    """
    csf = csf_map(size, size, geometry, model, device)
    budget = amplitude_budget(csf, threshold, max_amplitude)
    if min_cycles > 0:
        f_min = min_cycles / float(size)          # cycles per pixel
        budget = torch.where(radial_frequency_cpp(size, size, device) >= f_min,
                             budget, torch.zeros_like(budget))
    return csf, budget


# ═════════════════════════════════════════════════════════════════════════════
#  The bounded residual — a reparameterisation, not a projection
# ═════════════════════════════════════════════════════════════════════════════

def bounded_residual(raw: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    r"""
    Map an unconstrained tensor to a residual whose spectrum respects `budget`.

        Z      = rfft2(raw)
        Zhat   = B(f) * Z / sqrt(1 + |Z|^2)          elementwise, complex
        delta  = irfft2(Zhat)

    so |Zhat(f)| = B(f) * |Z| / sqrt(1+|Z|^2) < B(f) for every bin, by
    construction and for any input.

    REPARAMETERISATION, NOT PROJECTION, and for the reason spec.py gives for
    preferring sigmoid over clamping: a hard projection zeroes the gradient of
    every component sitting on its bound, which is exactly where an adversarial
    component wants to live. Here the squash is smooth everywhere, the bound is
    never reached exactly, and no component can ever be stuck.

    The scaling is applied to the COMPLEX coefficient, so phase is preserved
    exactly and only magnitude is limited. That also avoids differentiating
    arg(Z), which is undefined at the origin.

    Because irfft2 of a half-spectrum is real by construction, the result needs
    no conjugate-symmetry bookkeeping.
    """
    spec = torch.fft.rfft2(raw, norm="forward")
    mag2 = spec.real ** 2 + spec.imag ** 2
    scale = budget / torch.sqrt(1.0 + mag2)
    return torch.fft.irfft2(spec * scale, s=raw.shape[-2:], norm="forward")


MINKOWSKI_BETA = 3.0


def visibility_index(delta: torch.Tensor, csf: torch.Tensor,
                     reduce: str = "minkowski",
                     beta: float = MINKOWSKI_BETA,
                     contrast_scale: float = CONTRAST_SCALE,
                     eps: float = 1e-12) -> torch.Tensor:
    r"""
    How visible is a perturbation, in just-noticeable-difference units?

        v(f) = contrast_scale * |rfft(delta)(f)| * CSF(f)

        minkowski :  V = ( sum_f v(f)^beta )^(1/beta)          <- default
        max       :  V = max_f v(f)

    WHY NOT max. Single-grating detection is decided by the most visible
    component, but a perturbation is not a single grating. The visual system
    SUMS evidence across frequency and orientation channels, so many
    individually sub-threshold components are jointly visible — probability
    summation, standardly modelled as a Minkowski norm with beta around 3-4.

    This is not academic. A residual whose every bin respected its own budget
    measured max-visibility 0.235 while reaching a pixel-space deviation of
    +/-7.65 — nominally "invisible" and in fact catastrophic. max is the
    beta -> infinity limit and therefore the most permissive criterion
    available; defaulting to it would have flattered every result.

    V <= 1 means at or below the detection threshold of Barten's average
    observer UNDER THE ASSUMED GEOMETRY. It does not say a human will fail to
    notice the patch. No claim of imperceptibility should rest on this number
    without a study with human subjects.
    """
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)
    B = delta.shape[0]
    spec = torch.fft.rfft2(delta, norm="forward").abs()
    v = contrast_scale * spec * csf.unsqueeze(0).unsqueeze(0)
    flat = v.reshape(B, -1)

    if reduce == "minkowski":
        return flat.clamp(min=0).pow(beta).sum(dim=1).clamp(min=eps).pow(1.0 / beta)
    if reduce == "max":
        return flat.max(dim=1).values
    if reduce == "mean":
        return flat.mean(dim=1)
    raise ValueError(f"reduce must be minkowski|max|mean, got {reduce!r}")


def csf_residual(raw: torch.Tensor, budget: torch.Tensor, csf: torch.Tensor,
                 threshold: float = 1.0, beta: float = MINKOWSKI_BETA,
                 eps: float = 1e-12) -> torch.Tensor:
    r"""
    The full reparameterisation: unconstrained tensor -> residual of exactly
    `threshold` JND of Minkowski visibility.

    Two stages with two distinct jobs:

      SHAPE   bounded_residual() applies the 1/CSF envelope, so the residual's
              energy is biased toward the bands the eye discards. This is the
              "camouflage in the frequency domain" step.
      SCALE   a global rescale sets the total perceived visibility to exactly
              tau. This is the "how visible is it allowed to be" step.

    The two must be separate. The envelope alone bounds each bin but says
    nothing about their sum; the rescale alone would put energy wherever the
    attack likes, including where the eye is most sensitive.

    Normalising to EXACTLY tau rather than clamping at it is deliberate. The
    attack always wants the largest perturbation it is allowed, so the
    constraint is active at every step anyway, and an equality is smooth where
    a min() would introduce a non-differentiable kink at the bound.

    Per-sample: each image's residual gets its own scale, so a batch cannot
    average one image's headroom into another's.
    """
    delta = bounded_residual(raw, budget)
    v = visibility_index(delta, csf, reduce="minkowski", beta=beta)
    return delta * (threshold / v.clamp(min=eps)).view(-1, 1, 1, 1)


def fit_to_range(delta: torch.Tensor, reference: torch.Tensor,
                 quantile: float = 0.001, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Shrink `delta` so that `reference + delta` mostly stays inside [0,1],
    WITHOUT clipping.

    WHY CLIPPING CANNOT BE USED HERE. A clamp is a hard nonlinearity, and a
    hard nonlinearity generates broadband harmonics — including energy at the
    low and mid frequencies where the CSF peaks. Measured: a residual built to
    exactly 1.0 JND rose to 5.0 JND after clamping against a real image, with
    only 8% of pixels actually clipped. The whole spectral guarantee is
    destroyed by the one operation that looked like a formality.

    A UNIFORM RESCALE IS THE FIX because it is the only operation that leaves
    the spectrum's SHAPE untouched: scaling delta by c scales every Fourier
    coefficient by c, so a CSF-shaped residual stays CSF-shaped and its
    visibility scales linearly, V(c*delta) = c*V(delta). The budget therefore
    remains satisfied, as an inequality rather than an equality.

    Per-pixel headroom is (1 - r) where delta pushes up and r where it pushes
    down. The largest safe scale is the minimum of headroom/|delta| over
    pixels. A strict minimum is used ONLY up to `quantile`: Cityscapes frames
    contain genuinely saturated pixels (blown sky, black shadow) where the
    headroom is zero, and one of them would otherwise force c = 0 and delete
    the entire perturbation. Allowing the lowest `quantile` of pixels to clip
    trades a tiny, measurable violation for a usable attack — and
    realised_visibility() reports what actually happened rather than what was
    requested.
    """
    allowed = torch.where(delta > 0, 1.0 - reference, reference).clamp(min=0.0)
    ratio = allowed / delta.abs().clamp(min=eps)
    B = delta.shape[0]
    flat = ratio.reshape(B, -1)
    q = torch.quantile(flat.float(), quantile, dim=1).clamp(0.0, 1.0)
    return delta * q.view(-1, 1, 1, 1)


def realised_visibility(patch: torch.Tensor, reference: torch.Tensor,
                        csf: torch.Tensor, beta: float = MINKOWSKI_BETA,
                        contrast_scale: float = CONTRAST_SCALE) -> torch.Tensor:
    """
    Visibility of the residual that SURVIVED compositing, not the one that was
    requested. This is the number to report: tau is an intent, this is an
    outcome, and the gap between them is the honest measure of how well the
    constraint held.
    """
    return visibility_index(patch - reference, csf, reduce="minkowski",
                            beta=beta, contrast_scale=contrast_scale)


# ═════════════════════════════════════════════════════════════════════════════
#  Spectral analysis
# ═════════════════════════════════════════════════════════════════════════════

def rapsd(x: torch.Tensor, n_bins: int = 64
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Radially averaged power spectral density — CSFlow Appendix C.1, Eq 25.

        RAPSD(f) = (1/|A_f|) sum_{(u,v) in A_f} |F(u,v)|^2

    Returns (bin_centres in cycles/pixel, power). Used for reporting WHERE a
    perturbation put its energy, which is the claim this attack family lives
    or dies on: a patch whose spectrum looks like the natural-image power law
    is hiding; one with a bump at mid frequencies is not.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    power = torch.fft.rfft2(x, norm="forward").abs().pow(2).mean(dim=1)  # [B,h,w]

    f = radial_frequency_cpp(H, W, x.device)
    edges = torch.linspace(0.0, 0.5, n_bins + 1, device=x.device)
    idx = torch.bucketize(f.reshape(-1), edges[1:-1], right=False)

    out = torch.zeros(B, n_bins, device=x.device)
    count = torch.zeros(n_bins, device=x.device)
    count.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
    for b in range(B):
        out[b].scatter_add_(0, idx, power[b].reshape(-1))
    out = out / count.clamp(min=1)
    return 0.5 * (edges[:-1] + edges[1:]), out


def report(H: int, W: int, geometry: ViewingGeometry = ViewingGeometry(),
           model: str = "barten", threshold: float = 1.0, log=print):
    """
    Print where the budget actually is, before anyone trusts it.

    The essential sanity check for this attack family: if the budget at high
    frequency is not orders of magnitude larger than at the CSF peak, there is
    no gap to hide in and the premise fails at step zero.
    """
    csf = csf_map(H, W, geometry, model)
    budget = amplitude_budget(csf, threshold)
    f_cpp = radial_frequency_cpp(H, W)
    geometry.describe(log)
    log(f"[csf ] model     : {model}  threshold tau = {threshold:g} JND")
    log(f"[csf ] peak CSF  : {float(csf.max()):.1f} at "
        f"{float(f_cpp.reshape(-1)[int(csf.argmax())]):.4f} cyc/px "
        f"({float(geometry.to_cycles_per_degree(f_cpp.reshape(-1)[int(csf.argmax())])):.2f} cpd)")
    log("[csf ] amplitude budget by radial frequency:")
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1),
                   (0.1, 0.2), (0.2, 0.35), (0.35, 0.5001)):
        m = (f_cpp >= lo) & (f_cpp < hi)
        if not bool(m.any()):
            continue
        log(f"        {lo:.2f}-{hi:.2f} cyc/px : CSF {float(csf[m].mean()):8.2f}"
            f"   budget {float(budget[m].mean()):.4f}")
    return {"geometry": asdict(geometry), "model": model,
            "threshold": threshold, "peak_csf": float(csf.max()),
            "nyquist_cpd": geometry.nyquist_cpd}
