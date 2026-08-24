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

    # Peak white of the display, in cd/m^2. ONLY read by the luminance-aware
    # path at the bottom of this module; the legacy mu = 0.5 convention never
    # touches it, so adding this field changes no existing number. 100 cd/m^2
    # is the sRGB reference white (IEC 61966-2-1) and the value CSFlow's
    # geometry implies. It matters because Barten's sensitivity depends on
    # RETINAL ILLUMINANCE, which needs an absolute luminance, and a relative
    # luminance in [0,1] is not one.
    display_peak_cd_m2: float = 100.0

    @property
    def degrees_per_pixel(self) -> float:
        return (2.0 * math.atan(self.pixel_size_cm
                                / (2.0 * self.viewing_distance_cm))
                * 180.0 / math.pi)

    # ── constructors ─────────────────────────────────────────────────────────
    #
    # ONLY THE RATIO p/d MATTERS. degrees_per_pixel uses p/(2d) and nothing
    # else, so pixel_size_cm and viewing_distance_cm are not two independent
    # knobs — they are one. Both constructors below exploit that: they compute
    # the angle the caller actually cares about and then pick any (p, d) pair
    # that realises it. That is also why a geometry can be quoted as a single
    # number, deg/px, in the writeup.

    @classmethod
    def from_screen(cls, diagonal_in: float, pixels_wide: int,
                    distance_cm: float,
                    aspect: Tuple[int, int] = (16, 9)) -> "ViewingGeometry":
        r"""
        Geometry of a real display, from the numbers written on the box.

            width_cm = 2.54 * diagonal_in * a / sqrt(a^2 + b^2)
            p        = width_cm / pixels_wide

        Exists because deriving p by hand is where the mistake happens: a
        15.6" 1080p laptop has a 0.0180 cm pixel, 1.6x the 0.0114 cm the
        defaults assume, which at 50 cm makes a residual built for the default
        geometry measure ~5x its requested visibility. Nothing in the code was
        wrong there — the geometry was simply describing a different monitor.

        aspect is (16, 9) for essentially every modern panel; pass (16, 10) or
        (21, 9) for the exceptions.
        """
        a, b = aspect
        width_cm = 2.54 * diagonal_in * a / math.hypot(a, b)
        return cls(width_cm / float(pixels_wide), distance_cm)

    @classmethod
    def from_physical(cls, width_m: float, distance_m: float, patch_px: int,
                      declare_distance_cm: float = 100.0) -> "ViewingGeometry":
        r"""
        Geometry of a PHYSICAL patch — the driving threat model.

            theta = 2 * arctan( width / 2*distance ) / patch_px      degrees

        A 0.5 m patch seen at 20 m and rendered on a 128 px grid gives
        0.01119 deg/px, i.e. Nyquist 44.7 cpd — a FINER geometry than the
        0.0114 cm / 50 cm default (38.3 cpd). So a real road patch has MORE
        room to hide in than a desk monitor suggests, which is the opposite of
        what inspecting PNGs on a laptop implies.

        `patch_px` is the grid the BUDGET is computed on — PatchConfig.size,
        i.e. --patch_size — not the rendered side p = int(H * scale). At the
        defaults they coincide (128 = 0.25 * 512); if you change --patch_scale
        so they differ, Patch.apply() resamples and attenuates exactly the high
        frequencies the budget spent its allowance on, so make them match or
        the declared geometry describes a spectrum that never reaches the model.

        `declare_distance_cm` is arbitrary and does not affect the result —
        only p/d matters — it just fixes which of the infinitely many
        equivalent (p, d) pairs gets stored.
        """
        theta = math.degrees(2.0 * math.atan(width_m / (2.0 * distance_m)))
        theta_per_px = theta / float(patch_px)
        p = 2.0 * declare_distance_cm * math.tan(math.radians(theta_per_px) / 2.0)
        return cls(p, declare_distance_cm)

    # ── equivalences, for reporting ──────────────────────────────────────────

    @property
    def ppi(self) -> float:
        """The display pixel density this pixel size corresponds to."""
        return 2.54 / self.pixel_size_cm

    def matched_distance_cm(self, ppi: float) -> float:
        r"""
        Distance at which a `ppi` display reproduces THIS geometry.

            d = (2.54 / ppi) / (2 tan(theta/2))        so  d ~ 1 / ppi

        The practical use is validation by eye. A patch built for one geometry
        can be inspected on any screen, PROVIDED it is viewed at the distance
        that reproduces that geometry — at the default, 79 cm on a 141 PPI
        laptop or 119 cm on a 93 PPI desktop panel. Viewing closer is not a
        stricter test of the same claim, it is a test of a different one.

        This is a CONSISTENCY check, not a perceptual validation: see the note
        on visibility_index() — no claim of imperceptibility should rest on
        these numbers without a study with human subjects.
        """
        return ((2.54 / ppi)
                / (2.0 * math.tan(math.radians(self.degrees_per_pixel) / 2.0)))

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
        # EVERY visibility number below is conditional on the line above, so
        # print what it means on real hardware. matched_distance_cm(1.0) is the
        # constant k in d = k / PPI, because d scales as 1/ppi.
        log(f"[csf ] equivalent: a {self.ppi:.0f} PPI display at "
            f"{self.viewing_distance_cm:g} cm")
        log(f"[csf ] verify by : viewing at {self.matched_distance_cm(1.0):.0f}"
            f"/PPI cm — {self.matched_distance_cm(141):.0f} cm on a 141 PPI "
            f"laptop, {self.matched_distance_cm(93):.0f} cm on a 93 PPI panel")
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

# MU_FLOOR guards 2/mu against a near-black window, where the ratio diverges.
MU_FLOOR = 0.02


def local_contrast_scale(base: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    2 / mu, per sample and per channel, with mu measured on `base`.

    The alternative to CONTRAST_SCALE's mu = 0.5 assumption, and MEASURABLY
    the more accurate one on this data. Cityscapes frames are dark: the local
    mean inside a patch window was measured at 0.19-0.34 across five scenes,
    not 0.5. Michelson contrast is A/mu, so assuming 0.5 where the truth is
    0.25 UNDER-STATES contrast by 2x, and every patch generated under that
    assumption is about twice as visible as its tau claims.

    Confirmed by the spectral probe: at the same nominal tau = 1, the fixed
    convention admitted rms 0.0264 where the local one admitted 0.0122 --
    a factor of 2.16.

    Per CHANNEL rather than per luminance, because visibility_index() takes the
    rfft of each colour channel separately; a single luminance mean would apply
    one channel's contrast to another's spectrum.
    """
    if base.dim() == 3:
        base = base.unsqueeze(0)
    if mask is None:
        mu = base.mean(dim=(2, 3), keepdim=True)
    else:
        m = mask.to(base.dtype)
        while m.dim() < base.dim():
            m = m.unsqueeze(0)
        m = m.expand_as(base)
        mu = ((base * m).sum(dim=(2, 3), keepdim=True)
              / m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0))
    return 2.0 / mu.clamp(min=MU_FLOOR)


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
                 mask: Optional[torch.Tensor] = None,
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
    # For a SHAPED patch only `delta * mask` is ever pasted, so that is what
    # the observer sees and what tau must describe. Normalising over the full
    # square would spend the budget on padding that is thrown away.
    v = visibility_index(_masked(delta, mask), csf, reduce="minkowski", beta=beta)
    return delta * (threshold / v.clamp(min=eps)).view(-1, 1, 1, 1)


def _masked(delta: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """delta restricted to the pasted region; unchanged when there is no mask."""
    if mask is None:
        return delta
    while mask.dim() < delta.dim():
        mask = mask.unsqueeze(0)
    return delta * mask.to(delta.dtype)


def fit_to_range(delta: torch.Tensor, reference: torch.Tensor,
                 quantile: float = 0.001, mask: Optional[torch.Tensor] = None,
                 min_headroom: float = 1e-3,
                 eps: float = 1e-8) -> torch.Tensor:
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

    `mask` RESTRICTS THE FIT TO PIXELS THAT ARE ACTUALLY PASTED, and omitting
    it caused a silent total failure. A shaped patch (--shape alpha) loads its
    cutout padded with TRANSPARENT BLACK, so for refs/cover_cut.png 45% of the
    square is RGB 0 and 95% of the region outside the silhouette is. Those
    pixels have zero headroom for a negative residual, they are never pasted,
    and they are irrelevant to the attack — yet unmasked they drove the
    quantile to 0.0000, scaled the residual to EXACTLY ZERO, and turned a
    five-epoch run into a patch that was the unmodified reference image. The
    same quantile computed inside the silhouette was 1.26, i.e. the residual
    would have fitted at full scale with room to spare.

    Excluded pixels are given an unreachable ratio rather than being dropped,
    so the tensor shape stays static and the quantile stays batched.
    """
    allowed = torch.where(delta > 0, 1.0 - reference, reference).clamp(min=0.0)
    ratio = allowed / delta.abs().clamp(min=eps)

    # SATURATED PIXELS ARE EXCLUDED, not merely down-weighted. A pixel already
    # at 0 or 1 has exactly zero headroom in one direction, so its ratio is 0
    # and it drags any quantile to 0 with it. Over a 128x128 patch a 0.1%
    # quantile tolerates ~49 such pixels; over a full 512x1024 frame it
    # tolerates 524, and a Cityscapes sky blow-out alone exceeds that. The
    # measured consequence: a full-image probe scaled EVERY band to exactly
    # zero and reported "range-limited" for all six, producing no measurement.
    #
    # Excluding them is also the physically correct choice. Where the base is
    # already 1.0 and the residual pushes up, clamping returns 1.0 — the pixel
    # was never going to move, so it should not be allowed to veto the scale of
    # every pixel that could.
    saturated = allowed <= min_headroom
    ratio = torch.where(saturated, torch.full_like(ratio, 1e6), ratio)
    if mask is not None:
        while mask.dim() < ratio.dim():
            mask = mask.unsqueeze(0)
        ratio = torch.where(mask.expand_as(ratio), ratio,
                            torch.full_like(ratio, 1e6))
    B = delta.shape[0]
    q = torch.quantile(ratio.reshape(B, -1).float(), quantile, dim=1
                       ).clamp(0.0, 1.0)
    return delta * q.view(-1, 1, 1, 1)


def realised_visibility(patch: torch.Tensor, reference: torch.Tensor,
                        csf: torch.Tensor, beta: float = MINKOWSKI_BETA,
                        contrast_scale: float = CONTRAST_SCALE,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Visibility of the residual that SURVIVED compositing, not the one that was
    requested. This is the number to report: tau is an intent, this is an
    outcome, and the gap between them is the honest measure of how well the
    constraint held.
    """
    return visibility_index(_masked(patch - reference, mask), csf,
                            reduce="minkowski", beta=beta,
                            contrast_scale=contrast_scale)


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
           model: str = "barten", threshold: float = 1.0,
           min_cycles: float = 2.0, log=print):
    """
    Print where the budget actually is, before anyone trusts it.

    The essential sanity check for this attack family: if the budget at high
    frequency is not orders of magnitude larger than at the CSF peak, there is
    no gap to hide in and the premise fails at step zero.
    """
    # patch_budget, NOT amplitude_budget. The naive budget has no low-frequency
    # cutoff, so 1/CSF diverges at DC and the lowest band is printed with the
    # LARGEST allowance of any frequency -- the exact inversion documented as a
    # failure mode above. A probe run printed 0.0770 for 0.00-0.02 cyc/px
    # against 0.0026 at Nyquist, i.e. the table said the opposite of what the
    # attack actually does.
    csf, budget = patch_budget(H, geometry, model, threshold, min_cycles)
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


# ═════════════════════════════════════════════════════════════════════════════
#  Luminance-aware budget  (ADDITIVE — nothing above this line reads it)
# ═════════════════════════════════════════════════════════════════════════════
#
# WHY THIS SECTION EXISTS
# -----------------------
# Everything above assumes mu = 0.5 (CONTRAST_SCALE = 4.0) and a fixed retinal
# illuminance (BartenParams.E = 500 Td). Both are wrong on Cityscapes and both
# are wrong in the SAME direction: the frames are dark, so the real Michelson
# denominator is smaller and the real retinal illuminance is lower.
#
# It is a SEPARATE, OPT-IN path rather than a correction in place, because the
# tau ladder already run under the legacy convention has to keep its meaning.
# A tau = 0.25 overfit result from csf_single.sh must still be a tau = 0.25
# result after this file changes. Nothing above this banner calls anything
# below it.
#
# TWO INDEPENDENT LUMINANCE EFFECTS, and they must not be conflated:
#
#   1. THE MICHELSON DENOMINATOR.  Contrast is A / Y_ref, so the amplitude that
#      buys one unit of contrast is proportional to Y_ref. EXACTLY linear, and
#      it is the large term.
#   2. BARTEN'S SENSITIVITY.  CSF(f) itself shifts with retinal illuminance,
#      via the pupil. Comparatively flat across the photopic range, and it
#      moves in the COMPENSATING direction — a darker background is both a
#      smaller denominator and a less sensitive observer.
#
# Reporting a single ratio hides that the two fight each other. Every function
# here keeps them separable so the Step 0 table can decompose them.
#
# SOURCES: Barten (1999) Ch. 2 for the pupil formula; CSFlow (arXiv 2606.08833)
# Appendix A for the CSF parameters and Appendix D for the cpp -> cpd
# conversion. sRGB transfer function is IEC 61966-2-1.

# Rec.709 / sRGB luminance weights, applied to LINEARISED channels. Applying
# them to gamma-encoded RGB is the commonest way to get this wrong, and it is
# what local_contrast_scale() above does — it means over encoded values.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

# Floor on relative luminance. Y = 1e-3 at a 100 cd/m^2 peak is 0.1 cd/m^2,
# already below the photopic range Barten's parameters are fitted for. Anything
# darker gets reported, not silently modelled.
LUMINANCE_FLOOR = 1e-3
PHOTOPIC_FLOOR_CD_M2 = 1.0


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """
    sRGB code value in [0,1] -> linear relative luminance contribution.

    IEC 61966-2-1. The linear segment below 0.04045 is not decoration: a naive
    c**2.4 diverges from the standard by tens of percent in the darkest codes,
    which is precisely the range Cityscapes asphalt occupies.
    """
    return torch.where(c <= 0.04045, c / 12.92,
                       ((c.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(y: torch.Tensor) -> torch.Tensor:
    """Inverse of srgb_to_linear. Needed to express a budget back in code units."""
    return torch.where(y <= 0.0031308, y * 12.92,
                       1.055 * y.clamp(min=0.0) ** (1.0 / 2.4) - 0.055)


def srgb_slope(c: torch.Tensor) -> torch.Tensor:
    r"""
    dY/dc at code value c — the local gain of the sRGB transfer function.

    THE UNIT BRIDGE, and the reason it is needed: the optimiser perturbs SRGB
    CODE VALUES, but contrast is defined on LINEAR LUMINANCE. A budget derived
    in linear units is not a bound on the tensor the optimiser touches until it
    is divided by this slope.

    Valid only for small perturbations — it is a first-order approximation to a
    nonlinear curve. At tau of order 1 the residual is far below one code step,
    so the linearisation is comfortable; at a large tau it is not, and there
    the realised-visibility measurement rather than the budget is the number to
    trust.
    """
    return torch.where(c <= 0.04045,
                       torch.full_like(c, 1.0 / 12.92),
                       (2.4 / 1.055) * ((c.clamp(min=0.0) + 0.055) / 1.055) ** 1.4)


def relative_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """
    [...,3,H,W] of sRGB code values -> [...,H,W] linear relative luminance.

    LINEARISE THEN WEIGHT, never the other way round. Averaging gamma-encoded
    channels over-states the luminance of a dark region, which would make the
    fixed-reference simplification look better than it is.
    """
    w = torch.tensor(LUMA_WEIGHTS, dtype=rgb.dtype, device=rgb.device)
    lin = srgb_to_linear(rgb)
    return (lin * w.view(*([1] * (rgb.dim() - 3)), 3, 1, 1)).sum(dim=-3)


def pupil_diameter_mm(L_cd_m2: float) -> float:
    """
    Barten (1999) Eq 2.9, after Le Grand:  d = 5 - 3 tanh(0.4 log10 L).

    Ranges from ~8 mm in near-darkness to ~2 mm in bright light. This is the
    entire mechanism by which background luminance enters Barten's CSF, and it
    is why the sensitivity term is flat: the pupil partly compensates.
    """
    return 5.0 - 3.0 * math.tanh(0.4 * math.log10(max(L_cd_m2, 1e-6)))


def retinal_illuminance_td(L_cd_m2: float) -> float:
    """Trolands: E = (pi d^2 / 4) * L, with d from pupil_diameter_mm."""
    d = pupil_diameter_mm(L_cd_m2)
    return math.pi * d * d / 4.0 * L_cd_m2


def barten_params_at_luminance(
        Y_ref: float,
        geometry: "ViewingGeometry" = None,
        base: BartenParams = BartenParams()):
    """
    (BartenParams with E set from Y_ref, absolute luminance in cd/m^2).

    Y_ref is RELATIVE luminance in [0,1]; the display peak in `geometry` turns
    it into the absolute value Barten's pupil formula needs. The absolute
    luminance comes back too, because a run that quietly modelled 0.4 cd/m^2
    with photopic parameters needs to say so in its log.
    """
    geometry = geometry if geometry is not None else ViewingGeometry()
    L = max(float(Y_ref), LUMINANCE_FLOOR) * geometry.display_peak_cd_m2
    return (BartenParams(**{**asdict(base), "E": retinal_illuminance_td(L)}), L)


def contrast_scale_at_luminance(Y_ref: float) -> float:
    r"""
    2 / Y_ref — the luminance-aware replacement for CONTRAST_SCALE = 4.0.

    Same derivation as the constant it replaces (a real cosine of peak
    amplitude A on mean mu has Michelson contrast A/mu, and rfft2 with
    norm="forward" reports A/2), with mu measured rather than assumed. At
    Y_ref = 0.5 this returns exactly 4.0, so the legacy convention is the
    special case and the two paths agree where they should.
    """
    return 2.0 / max(float(Y_ref), LUMINANCE_FLOOR)


def csf_map_at_luminance(H: int, W: int, Y_ref: float,
                         geometry: "ViewingGeometry" = None,
                         model: str = "barten", device=None) -> torch.Tensor:
    """
    csf_map(), but with Barten's retinal illuminance set from Y_ref.

    'sso' HAS NO LUMINANCE PARAMETER — Watson & Ramirez fit a single photopic
    observer — so under model='sso' only the Michelson denominator responds to
    Y_ref and this returns exactly what csf_map() returns. That is a limitation
    of the model, not an oversight, and it makes 'sso' a useful control: it
    isolates effect 1 from effect 2.
    """
    geometry = geometry if geometry is not None else ViewingGeometry()
    if model not in CSF_MODELS:
        raise ValueError(f"csf model must be one of {CSF_MODELS}, got {model!r}")
    if model == "sso":
        return csf_map(H, W, geometry, model, device)
    fy, fx = rfft_frequency_grid(H, W, device)
    params, _ = barten_params_at_luminance(Y_ref, geometry)
    return barten_csf(geometry.to_cycles_per_degree((fy ** 2 + fx ** 2).sqrt()),
                      params)


def patch_budget_at_luminance(size: int, Y_ref: float,
                              geometry: "ViewingGeometry" = None,
                              model: str = "barten", threshold: float = 1.0,
                              min_cycles: float = 2.0,
                              max_amplitude: float = 0.25,
                              units: str = "srgb",
                              device=None):
    r"""
    (csf_values, budget) at background luminance Y_ref — the luminance-aware
    twin of patch_budget().

        B(f) = min( tau * Y_ref / (2 * CSF(f; Y_ref)) , A_max )     [linear]
        B(f) = B_linear(f) / (dY/dc)|_{c(Y_ref)}                    [srgb]

    units='srgb' is the DEFAULT because that is the space the optimiser works
    in; units='linear' is exposed for reporting, where the physical quantity is
    the meaningful one. Mixing them silently is the failure this argument
    exists to prevent.

    `min_cycles` carries over unchanged from patch_budget, and for the same
    reason: below one cycle across the patch a component is a brightness
    offset, not a grating, and the patch border turns any offset into a
    broadband edge. Nothing about luminance changes that argument.
    """
    geometry = geometry if geometry is not None else ViewingGeometry()
    if units not in ("srgb", "linear"):
        raise ValueError(f"units must be srgb|linear, got {units!r}")
    csf = csf_map_at_luminance(size, size, Y_ref, geometry, model, device)
    budget = amplitude_budget(csf, threshold, max_amplitude,
                              contrast_scale=contrast_scale_at_luminance(Y_ref))
    if units == "srgb":
        c_ref = linear_to_srgb(torch.tensor(float(Y_ref)))
        budget = (budget / srgb_slope(c_ref).clamp(min=1e-6)
                  ).clamp(max=max_amplitude)
    if min_cycles > 0:
        f_min = min_cycles / float(size)
        budget = torch.where(radial_frequency_cpp(size, size, device) >= f_min,
                             budget, torch.zeros_like(budget))
    return csf, budget


# ═════════════════════════════════════════════════════════════════════════════
#  Universal mode: hard projection, and the calibrated visibility
# ═════════════════════════════════════════════════════════════════════════════

def projected_residual(raw: torch.Tensor, budget: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Hard per-bin magnitude projection. Returns (delta, frac_at_bound).

        Z     = rfft2(raw)
        Zhat  = Z * min(1, B(f)/|Z|)        elementwise, PHASE UNTOUCHED
        delta = irfft2(Zhat)

    THE DIFFERENCE FROM bounded_residual(), and why both exist. That function
    is a smooth reparameterisation, chosen so no component is ever stuck on its
    bound with a dead gradient. This one is a true projection: a bin at the
    bound has zero gradient through the min(), which is exactly the failure
    mode spec.py's logit_clip note describes in pixel space.

    It is used anyway for the universal mode because the bound then holds as an
    INEQUALITY on the parameter itself rather than asymptotically, which is
    what makes "project the parameter, not the render" checkable. The dead
    gradient is real, so `frac_at_bound` is returned rather than discarded:
    a run where that fraction climbs toward 1 has stopped optimising and is
    only re-phasing, and the caller must be able to see it.

    Only the MAGNITUDE is limited. Scaling the complex coefficient preserves
    phase exactly and avoids differentiating arg(Z), undefined at the origin.
    """
    spec = torch.fft.rfft2(raw, norm="forward")
    mag = spec.abs()
    scale = (budget / mag.clamp(min=1e-12)).clamp(max=1.0)
    at_bound = (scale < 1.0) & (budget > 0)
    delta = torch.fft.irfft2(spec * scale, s=raw.shape[-2:], norm="forward")
    return delta, at_bound.float().mean()


def normalise_budget_to_tau(budget: torch.Tensor, csf: torch.Tensor,
                            threshold: float = 1.0,
                            beta: float = MINKOWSKI_BETA,
                            contrast_scale: float = CONTRAST_SCALE,
                            channels: int = 3,
                            eps: float = 1e-12) -> torch.Tensor:
    r"""
    Rescale a 1/CSF envelope so that ALL-BINS-AT-BUDGET pools to exactly `tau`.

        v_full = ( sum_f (contrast_scale * B(f) * CSF(f))^beta )^(1/beta)
        B'     = B * tau / v_full

    WHY THIS IS NECESSARY, and it was measured rather than reasoned. The
    obvious construction — project each bin onto the raw 1/CSF budget, then
    rescale the result so pooled visibility equals tau — DOES NOT WORK. The
    projection clamps most bins down, so the pooled visibility afterwards is
    far below tau, and the rescale that restores it multiplies every bin back
    up: a smoke test put bins at 90x their own budget. Per-bin bound and pooled
    equality are not simultaneously satisfiable on one envelope, and asserting
    both would have silently reported a bound that was violated by two orders
    of magnitude.

    Normalising the ENVELOPE instead makes the per-bin bound the real
    constraint and pooled visibility an inequality, v <= tau, with equality
    only if every bin saturates. That is the honest direction for the
    inequality to point.

    `channels` IS NOT COSMETIC. visibility_index() takes the rfft of each
    colour channel separately and pools over all of them, so an envelope
    normalised on a single channel is 3^(1/beta) = 1.44x too generous and the
    bound it promises is not the bound that holds. Measured before the term was
    added: pooled 0.0541 against a nominal tau of 0.05, an 8% overshoot that
    every assertion in the training loop would have passed, because the
    assertion and the normaliser shared the mistake.

    THE COST is that a universal run at nominal tau is LESS visible than a
    per-image overfit run at the same nominal tau, where tau is enforced as an
    equality. The two are therefore compared at matched REALISED visibility as
    well as at matched nominal tau -- which is what the realised-tau
    distribution is reported for, and why it is not an optional extra here.
    """
    per_ch = (contrast_scale * budget * csf).clamp(min=0).pow(beta).sum()
    v_full = (channels * per_ch).clamp(min=eps).pow(1.0 / beta)
    return budget * (threshold / v_full)


def universal_residual(raw: torch.Tensor, budget: torch.Tensor,
                       csf: torch.Tensor, threshold: float = 1.0,
                       beta: float = MINKOWSKI_BETA,
                       contrast_scale: float = CONTRAST_SCALE,
                       mask: Optional[torch.Tensor] = None,
                       eps: float = 1e-12
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    (delta, frac_at_bound) for the universal mode.

    `budget` must ALREADY be normalised by normalise_budget_to_tau() -- the
    caller does that once at construction rather than every step, since it
    depends only on (size, geometry, model, tau, beta) and not on the
    parameter. Pooled visibility is then bounded by tau BY CONSTRUCTION, and
    the assertion the training loop makes is a check on that reasoning rather
    than a correction to it.

    Only the SHAPE of `raw` matters: the projection is invariant to scaling raw
    up (every bin ends at its bound) but not down, so the init magnitude sets
    how saturated the residual starts.
    """
    delta, at_bound = projected_residual(raw, budget)
    return delta, at_bound


def calibrated_contrast_scale(Y_ref: float) -> float:
    r"""
    The contrast scale that is actually correct: 2 * (dY/dc) / Y_ref.

    DERIVATION, because the factor is easy to get wrong by a power of gamma.
    rfft2 with norm="forward" reports A_code/2 for a cosine of peak amplitude
    A_code in CODE values. The luminance modulation it produces is
    A_lin = (dY/dc) * A_code, and Michelson contrast is A_lin / Y_ref. So

        contrast = 2 * |rfft(delta)| * (dY/dc) / Y_ref

    THE LEGACY CONSTANT IS NOT A SPECIAL CASE OF THIS ONE, and it is worth
    being exact about why. CONTRAST_SCALE = 4.0 is 2/mu at mu = 0.5 with the
    slope term ABSENT, i.e. it assumes code values ARE luminances. This
    function never drops the slope, so it cannot return 4.0 for any Y_ref;
    at Y_ref = 0.5 it returns ~6.07, because a mid-LUMINANCE grey is code 0.735
    where the transfer curve is steep. The two conventions differ by that slope
    factor, and that is the calibration error: measured over the Cityscapes
    train split the legacy budget is ~1.9x too permissive at mid and Nyquist
    frequencies and ~3x too permissive at low frequency.
    """
    Y = max(float(Y_ref), LUMINANCE_FLOOR)
    c = linear_to_srgb(torch.tensor(Y))
    return contrast_scale_at_luminance(Y) * float(srgb_slope(c))


def calibrated_visibility(delta: torch.Tensor, base: torch.Tensor,
                          geometry: "ViewingGeometry" = None,
                          model: str = "barten",
                          beta: float = MINKOWSKI_BETA,
                          mask: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    (tau_calibrated [B], Y_ref [B]) — visibility of `delta` against the TRUE
    local luminance of the content it sits on.

    THE HEADLINE METRIC FOR THE UNIVERSAL MODE. tau is set once, at a fixed
    reference luminance, because a universal patch cannot look at the image it
    lands on. This recomputes what that one residual actually costs on each
    val image at that image's own measured luminance, so the sentence
    "budget set at tau = X; realised visibility ranged from a to b" can be
    written from measurement rather than assumption.

    BOTH the CSF (through Barten's retinal illuminance) and the contrast
    denominator respond to Y_ref, so this is not a rescale of the nominal
    number — the per-image ordering can differ from the nominal one.

    `base` is the footprint content in [0,1] sRGB code values, [B,3,H,W].
    """
    geometry = geometry if geometry is not None else ViewingGeometry()
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)
    if base.dim() == 3:
        base = base.unsqueeze(0)
    if delta.shape[0] == 1 and base.shape[0] > 1:
        delta = delta.expand_as(base)

    m = None if mask is None else mask.to(base.dtype)
    lum = relative_luminance(base)                              # [B,H,W]
    if m is None:
        Y = lum.flatten(1).mean(dim=1)
    else:
        mm = m.expand_as(lum)
        Y = (lum * mm).flatten(1).sum(1) / mm.flatten(1).sum(1).clamp(min=1.0)

    out = []
    for i in range(base.shape[0]):
        csf = csf_map_at_luminance(base.shape[-2], base.shape[-1],
                                   float(Y[i]), geometry, model,
                                   device=base.device)
        out.append(visibility_index(
            _masked(delta[i:i + 1], mask), csf, reduce="minkowski", beta=beta,
            contrast_scale=calibrated_contrast_scale(float(Y[i]))))
    return torch.cat(out), Y


def radial_profile(x: torch.Tensor, n_bins: int = 32
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Radially averaged rfft MAGNITUDE (not power) per frequency bin.

    rapsd() above returns power and is the right thing for comparing against a
    natural-image power law. This returns amplitude, because the BUDGET is an
    amplitude bound and the quantity worth plotting for the universal mode is
    the ratio spent/allowed — which is only meaningful if both sides are in the
    same units.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    mag = torch.fft.rfft2(x, norm="forward").abs().mean(dim=1)
    f = radial_frequency_cpp(H, W, x.device)
    edges = torch.linspace(0.0, 0.5, n_bins + 1, device=x.device)
    idx = torch.bucketize(f.reshape(-1), edges[1:-1], right=False)
    out = torch.zeros(B, n_bins, device=x.device)
    cnt = torch.zeros(n_bins, device=x.device)
    cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
    for b in range(B):
        out[b].scatter_add_(0, idx, mag[b].reshape(-1))
    return 0.5 * (edges[:-1] + edges[1:]), out / cnt.clamp(min=1)


def budget_radial_profile(budget: torch.Tensor, n_bins: int = 32
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Radial mean of a budget map, on the same bins as radial_profile()."""
    H = budget.shape[-2]
    W = 2 * (budget.shape[-1] - 1)
    f = radial_frequency_cpp(H, W, budget.device)
    edges = torch.linspace(0.0, 0.5, n_bins + 1, device=budget.device)
    idx = torch.bucketize(f.reshape(-1), edges[1:-1], right=False)
    out = torch.zeros(n_bins, device=budget.device)
    cnt = torch.zeros(n_bins, device=budget.device)
    cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
    out.scatter_add_(0, idx, budget.reshape(-1))
    return 0.5 * (edges[:-1] + edges[1:]), out / cnt.clamp(min=1)
