"""Human contrast sensitivity in the frequency domain."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch

CSF_MODELS = ("barten", "sso")


@dataclass(frozen=True)
class ViewingGeometry:
    """The observer assumption that makes a CSF meaningful."""
    pixel_size_cm: float = 0.0114
    viewing_distance_cm: float = 50.0

    display_peak_cd_m2: float = 100.0

    @property
    def degrees_per_pixel(self) -> float:
        return (2.0 * math.atan(self.pixel_size_cm
                                / (2.0 * self.viewing_distance_cm))
                * 180.0 / math.pi)

    @classmethod
    def from_screen(cls, diagonal_in: float, pixels_wide: int,
                    distance_cm: float,
                    aspect: Tuple[int, int] = (16, 9)) -> "ViewingGeometry":
        """Geometry of a real display, from the numbers written on the box."""
        a, b = aspect
        width_cm = 2.54 * diagonal_in * a / math.hypot(a, b)
        return cls(width_cm / float(pixels_wide), distance_cm)

    @classmethod
    def from_physical(cls, width_m: float, distance_m: float, patch_px: int,
                      declare_distance_cm: float = 100.0) -> "ViewingGeometry":
        """Geometry of a PHYSICAL patch — the driving threat model."""
        theta = math.degrees(2.0 * math.atan(width_m / (2.0 * distance_m)))
        theta_per_px = theta / float(patch_px)
        p = 2.0 * declare_distance_cm * math.tan(math.radians(theta_per_px) / 2.0)
        return cls(p, declare_distance_cm)

    @property
    def ppi(self) -> float:
        """The display pixel density this pixel size corresponds to."""
        return 2.54 / self.pixel_size_cm

    def matched_distance_cm(self, ppi: float) -> float:
        """Distance at which a `ppi` display reproduces THIS geometry."""
        return ((2.54 / ppi)
                / (2.0 * math.tan(math.radians(self.degrees_per_pixel) / 2.0)))

    def to_cycles_per_degree(self, f_cpp: torch.Tensor) -> torch.Tensor:
        """cycles/pixel -> cycles/degree."""
        return f_cpp / self.degrees_per_pixel

    @property
    def nyquist_cpd(self) -> float:
        """The highest frequency the sampling grid can represent, in cpd."""
        return 0.5 / self.degrees_per_pixel

    def describe(self, log=print):
        log(f"[csf ] geometry  : pixel {self.pixel_size_cm} cm at "
            f"{self.viewing_distance_cm} cm -> "
            f"{self.degrees_per_pixel:.5f} deg/px")
        log(f"[csf ] equivalent: a {self.ppi:.0f} PPI display at "
            f"{self.viewing_distance_cm:g} cm")
        log(f"[csf ] verify by : viewing at {self.matched_distance_cm(1.0):.0f}"
            f"/PPI cm — {self.matched_distance_cm(141):.0f} cm on a 141 PPI "
            f"laptop, {self.matched_distance_cm(93):.0f} cm on a 93 PPI panel")
        log(f"[csf ] Nyquist   : {self.nyquist_cpd:.1f} cpd per axis, "
            f"{self.nyquist_cpd * 2 ** 0.5:.1f} cpd on the diagonal "
            f"(human cutoff ~50-60 cpd, so the corner is inside the passband)")


@dataclass(frozen=True)
class BartenParams:
    """CSFlow Appendix A, Eqs 11-14."""
    sigma: float = 0.5 / 60.0
    k: float = 3.0
    T: float = 0.1
    X_0: float = 60.0
    X_max: float = 12.0
    N_max: float = 15.0
    n: float = 0.03
    p: float = 1.2e6
    E: float = 500.0
    phi_0: float = 3e-8
    u_0: float = 7.0


MAX_CPP: float = 2 ** 0.5 / 2

RADIAL_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.02), (0.02, 0.05), (0.05, 0.10),
    (0.10, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.71))


def optical_mtf(f_cpd: torch.Tensor, sigma: float) -> torch.Tensor:
    """M_opt(f) = exp(-2 pi^2 sigma^2 f^2) — CSFlow Eq 10."""
    return torch.exp(-2.0 * math.pi ** 2 * sigma ** 2 * f_cpd ** 2)


def barten_csf(f_cpd: torch.Tensor,
               params: BartenParams = BartenParams(),
               eps: float = 1e-12) -> torch.Tensor:
    """Contrast sensitivity at `f_cpd` cycles/degree — CSFlow Eq 9."""
    f = f_cpd.clamp(min=0.0)
    inhib = 1.0 - torch.exp(-((f / params.u_0) ** 2))

    integ = (2.0 / params.T) * (1.0 / params.X_0 ** 2
                                + 1.0 / params.X_max ** 2
                                + f ** 2 / params.N_max ** 2)
    noise = (1.0 / (params.n * params.p * params.E)
             + params.phi_0 / inhib.clamp(min=eps))

    csf = (optical_mtf(f, params.sigma) / params.k) / (integ * noise).sqrt()
    return torch.where(f > 0, csf, torch.zeros_like(csf))


@dataclass(frozen=True)
class SSOParams:
    g: float = 330.74
    fm: float = 7.28
    l: float = 0.837
    s: float = 1.809
    w: float = 1.0
    os: float = 6.664


def sso_csf(fy_cpd: torch.Tensor, fx_cpd: torch.Tensor,
            params: SSOParams = SSOParams(),
            eps: float = 1e-12) -> torch.Tensor:
    """CSF_SSO(fx,fy) = CSF_Tyler(f) * OE(fx,fy)"""
    f2 = fy_cpd ** 2 + fx_cpd ** 2
    f = f2.clamp(min=eps).sqrt()

    tyler = params.g * (torch.exp(-f / params.fm)
                        - params.l * torch.exp(-f2 / params.s ** 2))
    oe = 1.0 - params.w * (4.0 * (1.0 - torch.exp(-f / params.os))
                           * fx_cpd ** 2 * fy_cpd ** 2) / f2.clamp(min=eps) ** 2

    csf = tyler * oe
    return torch.where(f2 > eps, csf, torch.zeros_like(csf)).clamp(min=0.0)


def rfft_frequency_grid(H: int, W: int, device=None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(fy, fx) in cycles/pixel on the rfft2 half-spectrum, shape [H, W//2+1]."""
    fy = torch.fft.fftfreq(H, device=device).view(H, 1)
    fx = torch.fft.rfftfreq(W, device=device).view(1, -1)
    return fy.expand(H, fx.shape[-1]), fx.expand(H, fx.shape[-1])


def radial_frequency_cpp(H: int, W: int, device=None) -> torch.Tensor:
    """|f| = sqrt(fy^2 + fx^2) in cycles/pixel on the rfft2 grid."""
    fy, fx = rfft_frequency_grid(H, W, device)
    return (fy ** 2 + fx ** 2).sqrt()


def csf_map(H: int, W: int,
            geometry: ViewingGeometry = ViewingGeometry(),
            model: str = "barten",
            device=None) -> torch.Tensor:
    """CSF evaluated at every rfft2 bin of an H x W image."""
    if model not in CSF_MODELS:
        raise ValueError(f"csf model must be one of {CSF_MODELS}, got {model!r}")
    fy, fx = rfft_frequency_grid(H, W, device)
    if model == "barten":
        return barten_csf(geometry.to_cycles_per_degree(
            (fy ** 2 + fx ** 2).sqrt()))
    return sso_csf(geometry.to_cycles_per_degree(fy),
                   geometry.to_cycles_per_degree(fx))


CONTRAST_SCALE = 4.0

MU_FLOOR = 0.02


def local_contrast_scale(base: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """2 / mu, per sample and per channel, with mu measured on `base`."""
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
    """Per-frequency budget on the rfft magnitude"""
    return (threshold / (contrast_scale * csf.clamp(min=eps))
            ).clamp(max=max_amplitude)


def band_limit(budget: torch.Tensor, size: int, min_cycles: float = 2.0,
               max_cpp: float = 0.0, device=None) -> torch.Tensor:
    """Zero the budget outside the radial band [min_cycles/size, max_cpp]."""
    f = radial_frequency_cpp(size, size, device)
    keep = torch.ones_like(budget, dtype=torch.bool)
    if min_cycles > 0:
        keep &= (f >= min_cycles / float(size))
    if max_cpp > 0:
        keep &= (f <= max_cpp)
    if not bool(keep.any()):
        raise ValueError(
            f"the band [{min_cycles}/{size}, {max_cpp}] c/px contains no rfft "
            f"bin at size={size}: every bin would be zeroed and "
            f"normalise_budget_to_tau would divide by ~0. Widen it.")
    return torch.where(keep, budget, torch.zeros_like(budget))


def patch_budget(size: int, geometry: ViewingGeometry = ViewingGeometry(),
                 model: str = "barten", threshold: float = 1.0,
                 min_cycles: float = 2.0, max_amplitude: float = 0.25,
                 device=None, max_cpp: float = 0.0
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(csf_values, budget) for a size x size PATCH, with the low-frequency band removed."""
    csf = csf_map(size, size, geometry, model, device)
    budget = amplitude_budget(csf, threshold, max_amplitude)
    return csf, band_limit(budget, size, min_cycles, max_cpp, device)


def bounded_residual(raw: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained tensor to a residual whose spectrum respects `budget`."""
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
    """How visible is a perturbation, in just-noticeable-difference units?"""
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
    """The full reparameterisation: unconstrained tensor -> residual of exactly `threshold` JND of Minkowski visibility."""
    delta = bounded_residual(raw, budget)
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
    """Shrink `delta` so that `reference + delta` mostly stays inside [0,1], WITHOUT clipping."""
    allowed = torch.where(delta > 0, 1.0 - reference, reference).clamp(min=0.0)
    ratio = allowed / delta.abs().clamp(min=eps)

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
    """Visibility of the residual that SURVIVED compositing, not the one that was requested."""
    return visibility_index(_masked(patch - reference, mask), csf,
                            reduce="minkowski", beta=beta,
                            contrast_scale=contrast_scale)


def fit_to_visibility(delta: torch.Tensor, base: torch.Tensor,
                      csf: torch.Tensor, threshold: float = 1.0,
                      beta: float = MINKOWSKI_BETA,
                      contrast_scale=CONTRAST_SCALE,
                      mask: Optional[torch.Tensor] = None,
                      iters: int = 12,
                      shared_scale: bool = False) -> torch.Tensor:
    """Scale `delta` so the residual that SURVIVES compositing satisfies tau."""
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)
    if base.dim() == 3:
        base = base.unsqueeze(0)
    B = base.shape[0]

    def realised(c: torch.Tensor) -> torch.Tensor:
        r = (base + c.view(-1, 1, 1, 1) * delta).clamp(0.0, 1.0) - base
        return visibility_index(_masked(r, mask), csf, beta=beta,
                                contrast_scale=contrast_scale)

    with torch.no_grad():
        one = torch.ones(B, device=delta.device, dtype=delta.dtype)
        ok = realised(one) <= threshold
        if bool(ok.all()):
            return delta
        lo = torch.where(ok, one, torch.zeros_like(one))
        hi = one.clone()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            good = realised(mid) <= threshold
            lo = torch.where(good, mid, lo)
            hi = torch.where(good, hi, mid)
        c = lo

    if shared_scale:
        return delta * c.min()
    return delta * c.view(-1, 1, 1, 1)


def rapsd(x: torch.Tensor, n_bins: int = 64
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Radially averaged power spectral density — CSFlow Appendix C.1, Eq 25."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    power = torch.fft.rfft2(x, norm="forward").abs().pow(2).mean(dim=1)

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
    """Print where the budget actually is, before anyone trusts it."""
    csf, budget = patch_budget(H, geometry, model, threshold, min_cycles)
    f_cpp = radial_frequency_cpp(H, W)
    geometry.describe(log)
    log(f"[csf ] model     : {model}  threshold tau = {threshold:g} JND")
    log(f"[csf ] peak CSF  : {float(csf.max()):.1f} at "
        f"{float(f_cpp.reshape(-1)[int(csf.argmax())]):.4f} cyc/px "
        f"({float(geometry.to_cycles_per_degree(f_cpp.reshape(-1)[int(csf.argmax())])):.2f} cpd)")
    log("[csf ] amplitude budget by radial frequency:")
    for lo, hi in RADIAL_BANDS:
        m = (f_cpp >= lo) & (f_cpp < hi)
        if not bool(m.any()):
            continue
        log(f"        {lo:.2f}-{hi:.2f} cyc/px : CSF {float(csf[m].mean()):8.2f}"
            f"   budget {float(budget[m].mean()):.4f}")
    return {"geometry": asdict(geometry), "model": model,
            "threshold": threshold, "peak_csf": float(csf.max()),
            "nyquist_cpd": geometry.nyquist_cpd}


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

LUMINANCE_FLOOR = 1e-3
PHOTOPIC_FLOOR_CD_M2 = 1.0


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """sRGB code value in [0,1] -> linear relative luminance contribution."""
    return torch.where(c <= 0.04045, c / 12.92,
                       ((c.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(y: torch.Tensor) -> torch.Tensor:
    """Inverse of srgb_to_linear."""
    return torch.where(y <= 0.0031308, y * 12.92,
                       1.055 * y.clamp(min=0.0) ** (1.0 / 2.4) - 0.055)


def srgb_slope(c: torch.Tensor) -> torch.Tensor:
    """dY/dc at code value c — the local gain of the sRGB transfer function."""
    return torch.where(c <= 0.04045,
                       torch.full_like(c, 1.0 / 12.92),
                       (2.4 / 1.055) * ((c.clamp(min=0.0) + 0.055) / 1.055) ** 1.4)


def relative_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """[...,3,H,W] of sRGB code values -> [...,H,W] linear relative luminance."""
    w = torch.tensor(LUMA_WEIGHTS, dtype=rgb.dtype, device=rgb.device)
    lin = srgb_to_linear(rgb)
    return (lin * w.view(*([1] * (rgb.dim() - 3)), 3, 1, 1)).sum(dim=-3)


def pupil_diameter_mm(L_cd_m2: float) -> float:
    """Barten (1999) Eq 2.9, after Le Grand: d = 5 - 3 tanh(0.4 log10 L)."""
    return 5.0 - 3.0 * math.tanh(0.4 * math.log10(max(L_cd_m2, 1e-6)))


def retinal_illuminance_td(L_cd_m2: float) -> float:
    """Trolands: E = (pi d^2 / 4) * L, with d from pupil_diameter_mm."""
    d = pupil_diameter_mm(L_cd_m2)
    return math.pi * d * d / 4.0 * L_cd_m2


def barten_params_at_luminance(
        Y_ref: float,
        geometry: "ViewingGeometry" = None,
        base: BartenParams = BartenParams()):
    """(BartenParams with E set from Y_ref, absolute luminance in cd/m^2)."""
    geometry = geometry if geometry is not None else ViewingGeometry()
    L = max(float(Y_ref), LUMINANCE_FLOOR) * geometry.display_peak_cd_m2
    return (BartenParams(**{**asdict(base), "E": retinal_illuminance_td(L)}), L)


def contrast_scale_at_luminance(Y_ref: float) -> float:
    """2 / Y_ref — the luminance-aware replacement for CONTRAST_SCALE = 4.0."""
    return 2.0 / max(float(Y_ref), LUMINANCE_FLOOR)


def csf_map_at_luminance(H: int, W: int, Y_ref: float,
                         geometry: "ViewingGeometry" = None,
                         model: str = "barten", device=None) -> torch.Tensor:
    """csf_map(), but with Barten's retinal illuminance set from Y_ref."""
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
                              device=None, max_cpp: float = 0.0):
    """(csf_values, budget) at background luminance Y_ref — the luminance-aware twin of patch_budget()."""
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
    return csf, band_limit(budget, size, min_cycles, max_cpp, device)


def projected_residual(raw: torch.Tensor, budget: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard per-bin magnitude projection."""
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
    """Rescale a 1/CSF envelope so that ALL-BINS-AT-BUDGET pools to exactly `tau`."""
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
    """(delta, frac_at_bound) for the universal mode."""
    delta, at_bound = projected_residual(raw, budget)
    return delta, at_bound


def calibrated_contrast_scale(Y_ref: float) -> float:
    """The contrast scale that is actually correct: 2 * (dY/dc) / Y_ref."""
    Y = max(float(Y_ref), LUMINANCE_FLOOR)
    c = linear_to_srgb(torch.tensor(Y))
    return contrast_scale_at_luminance(Y) * float(srgb_slope(c))


def calibrated_visibility(delta: torch.Tensor, base: torch.Tensor,
                          geometry: "ViewingGeometry" = None,
                          model: str = "barten",
                          beta: float = MINKOWSKI_BETA,
                          mask: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(tau_calibrated [B], Y_ref [B]) — visibility of `delta` against the TRUE local luminance of the content it sits on."""
    geometry = geometry if geometry is not None else ViewingGeometry()
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)
    if base.dim() == 3:
        base = base.unsqueeze(0)
    if delta.shape[0] == 1 and base.shape[0] > 1:
        delta = delta.expand_as(base)

    m = None if mask is None else mask.to(base.dtype)
    lum = relative_luminance(base)
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
    """Radially averaged rfft MAGNITUDE (not power) per frequency bin."""
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
