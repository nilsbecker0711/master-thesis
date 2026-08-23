r"""
SPECTRAL diagnostics — WHICH SPATIAL FREQUENCIES DOES THE NETWORK ACTUALLY READ?

WHY THIS EXISTS
---------------
csf.py bounds a perturbation by what the EYE can see. That is only half of the
attack's premise. The full claim is a GAP:

    exploitable band  =  (frequencies the eye discards)
                      INTERSECT
                         (frequencies the network reads)

The first set is a property of the CSF and the viewing geometry, and it is
IMAGE-INDEPENDENT by construction: patch_budget() takes a size, a geometry, a
model and a tau, and no image ever reaches it.

The second set is measured here, and nothing in the repository measured it
before. csf.py's own header names the risk explicitly: SegFormer's patch
embedding has stride 4, so the very highest frequencies may be attenuated
before the first attention block ever sees them. If that is true the gap is
narrow or empty, and the whole family is bounded by an architectural detail
rather than by tau.

THE METHOD, AND THE ONE CONTROL THAT MATTERS
--------------------------------------------
For each radial frequency band, inject BAND-LIMITED NOISE and record which
pixels change their argmax. Same instrument as geometric.receptive_field():
random stimulus, prediction change, no labels and no target class. That is what
makes it a control rather than a correlate, and it is why this probe runs on
ADE20K weights exactly as measure_erf.py does.

The control that decides whether the measurement means anything is that every
band is normalised to THE SAME PERCEPTUAL COST before it is injected:

    equal visibility   V(delta) = tau   for every band      <- default

Inject equal AMPLITUDE instead and the result is uninterpretable: high
frequencies are both less visible AND differently effective, and the two cannot
be separated afterwards. At equal visibility the question is sharp — PER UNIT
OF PERCEPTUAL COST, which band buys the most prediction change? Set
normalise='rms' to run the equal-amplitude version as a control; the ratio
between the two is itself informative.

WHAT COMES BACK, AND HOW TO READ IT
-----------------------------------
  flip rate rises toward Nyquist   the gap is real. Perturbation the eye
                                   discards still moves the network, and tau
                                   is the binding constraint rather than the
                                   architecture.
  flip rate collapses at high f    the network cannot read what the eye
                                   discards. The premise fails, and it fails
                                   for an architectural reason worth naming.
  peak band differs per image      the readable band is SCENE-DEPENDENT, and
                                   one fixed budget is the wrong object.
  peak band stable across images   the readable band is a MODEL property and
                                   transfers, exactly as the ERF does.

The last two are the direct test of whether the band is image-bound.
summarise() reports them explicitly rather than leaving them to be eyeballed
off a curve.

RANGE-LIMITED BANDS ARE A RESULT, NOT A NUISANCE
------------------------------------------------
A band the eye barely sees needs a large amplitude to reach tau JND, and a
large amplitude may not fit in [0,1] on top of real image content. fit_to_range
then shrinks it and the band never reaches its target cost. That is not a
measurement artefact to be hidden — it is a hard ceiling on how much signal the
attack can place in the invisible band, and it is reported per band as
`realised_visibility` against `target`. A band that cannot be driven to tau is
a band the attack cannot fully exploit however good the optimiser is.

MEAN LUMINANCE — A DELIBERATE DEPARTURE FROM THE ATTACK MODES
-------------------------------------------------------------
csf.CONTRAST_SCALE fixes the local mean at 0.5, and says so:

    "a patch sitting on dark asphalt has a lower local mean and therefore
     HIGHER contrast for the same amplitude, so this under-estimates
     visibility there. tau absorbs the constant"

tau absorbs a CONSTANT. It cannot absorb a quantity that varies per image,
which is exactly what the local mean does — and this probe's entire validity
rests on every band costing the same. Michelson contrast of a cosine of
amplitude A on mean mu is A/mu, and rfft2(norm='forward') reports A/2, so

    contrast_scale = 2 / mu          (= 4.0 exactly when mu = 0.5)

`contrast_mean='local'` is therefore the DEFAULT HERE, and it is a different
convention from the one spec.py and conditional_generator.py currently use.
The discrepancy is printed in every run and stored in the JSON. Pass
contrast_mean='fixed' to reproduce the attack modes' convention.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..data.cityscapes import upsample_to
from ..patch import csf as csf_mod
from ..patch.conditional_generator import (composite_batch, denormalise_batch,
                                           window_reference)

# The SAME radial banding csf.report() prints its budget table over, so a
# sensitivity row lines up with the budget row above it without rebinning.
DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.02), (0.02, 0.05), (0.05, 0.10),
    (0.10, 0.20), (0.20, 0.35), (0.35, 0.50))

NORMALISE_MODES = ("visibility", "rms")
CONTRAST_MEAN_MODES = ("local", "fixed")
REGION_MODES = ("patch", "full")

# Floor on the local mean. A window lying entirely in black shadow gives
# mu -> 0 and 2/mu -> infinity, which would silently scale that band's
# stimulus to nothing. 0.02 is below any real Cityscapes road surface, so the
# floor binds only on degenerate crops — and binding is counted and reported
# rather than absorbed.
MU_FLOOR = 0.02


# ═════════════════════════════════════════════════════════════════════════════
#  Band-limited stimulus
# ═════════════════════════════════════════════════════════════════════════════

def band_limited_noise(n: int, H: int, W: int, lo: float, hi: float,
                       device=None, channels: int = 3,
                       generator: Optional[torch.Generator] = None
                       ) -> torch.Tensor:
    r"""
    [n, channels, H, W] of white noise with all energy outside [lo, hi)
    cycles/pixel removed.

        Z = rfft2(white)
        Z[ |f| < lo  or  |f| >= hi ] = 0
        out = irfft2(Z)

    rfft2 rather than fft2 for the reason bounded_residual() gives: the result
    is real by construction, so there is no conjugate-symmetry bookkeeping and
    no imaginary residue can leak in.

    The band edges are HARD in the frequency domain, which makes the stimulus
    ring in the spatial domain. That is correct here and must not be "fixed"
    with a soft window: a soft edge leaks energy into neighbouring bands, and
    the whole measurement is about attributing an effect to ONE band. Ringing
    costs spatial locality, which this probe does not use.

    DC IS ALWAYS EXCLUDED, even when the requested band contains it. barten_csf
    returns EXACTLY zero at f=0, so a DC component costs exactly nothing under
    visibility_index — and normalising a stimulus to a fixed perceptual cost
    would then hand the lowest band an unbounded brightness offset for free,
    making its flip rate incomparable with every other band's. This is the same
    objection patch_budget() raises when it removes the low band: below one
    cycle across the patch there is no grating, only an offset, and an offset
    against the patch border is a visible edge rather than a sub-threshold
    signal. Near-DC bins remain in the sweep and are not free, only cheap;
    where that makes a band unreachable it is reported as range-limited rather
    than silently absorbed.
    """
    if not 0.0 <= lo < hi:
        raise ValueError(f"band must satisfy 0 <= lo < hi, got [{lo}, {hi})")
    white = torch.randn(n, channels, H, W, device=device, generator=generator)
    spec = torch.fft.rfft2(white, norm="forward")
    f = csf_mod.radial_frequency_cpp(H, W, device)
    keep = (f >= lo) & (f < hi) & (f > 0)
    return torch.fft.irfft2(spec * keep, s=(H, W), norm="forward")


def contrast_scale(base: torch.Tensor, mode: str = "local",
                   mask: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, int]:
    r"""
    (scale [B,C,1,1], n_floored) — the contrast_scale for visibility_index().

        local : 2 / mu, per sample and per channel, mu measured on `base`
        fixed : csf.CONTRAST_SCALE, the mu = 0.5 convention of the attack modes

    A [B,C,1,1] tensor broadcasts against the [B,C,h,w] spectrum inside
    visibility_index() with NO change to csf.py — that function already
    multiplies rather than assuming a scalar.

    Per CHANNEL, not per luminance, because visibility_index() takes the rfft
    of each colour channel separately and Minkowski-sums across all of them. A
    single luminance mean would apply one channel's contrast to another's
    spectrum.
    """
    if mode not in CONTRAST_MEAN_MODES:
        raise ValueError(f"contrast_mean must be one of {CONTRAST_MEAN_MODES}, "
                         f"got {mode!r}")
    B, C = base.shape[0], base.shape[1]
    if mode == "fixed":
        return (torch.full((B, C, 1, 1), csf_mod.CONTRAST_SCALE,
                           device=base.device, dtype=base.dtype), 0)

    if mask is None:
        mu = base.mean(dim=(2, 3), keepdim=True)
    else:
        m = mask.to(base.dtype)
        while m.dim() < base.dim():
            m = m.unsqueeze(0)
        m = m.expand_as(base)
        mu = ((base * m).sum(dim=(2, 3), keepdim=True)
              / m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0))

    n_floored = int((mu < MU_FLOOR).sum())
    return 2.0 / mu.clamp(min=MU_FLOOR), n_floored


def scale_to_cost(delta: torch.Tensor, csf: torch.Tensor, target: float,
                  mode: str = "visibility",
                  beta: float = csf_mod.MINKOWSKI_BETA,
                  cscale=csf_mod.CONTRAST_SCALE,
                  mask: Optional[torch.Tensor] = None,
                  eps: float = 1e-12) -> torch.Tensor:
    """
    Rescale each sample's residual to a fixed perceptual (or raw) cost.

    Uniform rescale, per sample, for the reason fit_to_range() gives: scaling
    leaves the spectrum's SHAPE untouched, so a band-limited stimulus stays
    band-limited and its cost scales linearly. Any other normalisation would
    move energy between bands and destroy the attribution this probe depends on.
    """
    if mode not in NORMALISE_MODES:
        raise ValueError(f"normalise must be one of {NORMALISE_MODES}, "
                         f"got {mode!r}")
    masked = csf_mod._masked(delta, mask)
    if mode == "visibility":
        cost = csf_mod.visibility_index(masked, csf, reduce="minkowski",
                                        beta=beta, contrast_scale=cscale)
    else:
        B = delta.shape[0]
        cost = masked.reshape(B, -1).pow(2).mean(dim=1).sqrt()
    return delta * (target / cost.clamp(min=eps)).view(-1, 1, 1, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  The probe
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def frequency_sensitivity(model, imgs: torch.Tensor, patch, mean_t, std_t,
                          bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
                          target: float = 0.25,
                          n_probes: int = 8,
                          region: str = "patch",
                          normalise: str = "visibility",
                          contrast_mean: str = "local",
                          csf_model: str = "barten",
                          geometry: Optional[csf_mod.ViewingGeometry] = None,
                          beta: float = csf_mod.MINKOWSKI_BETA,
                          log=print) -> List[Dict]:
    r"""
    Prediction-change rate per radial frequency band, at equal perceptual cost.

    `patch` supplies GEOMETRY ONLY — cfg.size, cfg.scale, the resolved
    placement and any shape mask. Its parameter is never read and never
    written, so this probe is safe to run on a trained patch without
    disturbing it. That is a deliberate difference from
    geometric.receptive_field(), which overwrites patch.param and restores it.

    region='patch'  the stimulus goes in the patch window and is composited
                    through composite_batch() — the SAME code path the
                    conditional generator uses, so placement, resizing and
                    normalisation match the real attack exactly. This is the
                    threat model.
    region='full'   the stimulus covers the whole frame. Cleaner as an
                    instrument (no receptive-field ceiling confounds the band
                    response) but it is not an attack. Run both: if they
                    disagree, the disagreement is the ERF, not the spectrum.

    ORDERING: pass a patch whose placement is already resolved. Unresolved,
    it falls back to centre exactly as Patch.apply() does.

    Returns one dict per band with per-image and per-probe detail retained, so
    summarise() can answer the image-boundness question rather than only the
    mean curve.
    """
    if region not in REGION_MODES:
        raise ValueError(f"region must be one of {REGION_MODES}, got {region!r}")
    geometry = geometry or csf_mod.ViewingGeometry()

    B, _, H, W = imgs.shape
    device = imgs.device
    clean = upsample_to(model(imgs), (H, W)).argmax(1)
    img01 = denormalise_batch(imgs, mean_t, std_t)

    if region == "patch":
        p = int(H * patch.cfg.scale)
        S = patch.cfg.size
        top, left = (patch.placement if patch.placement is not None
                     else ((H - p) // 2, (W - p) // 2))
        placements = [(int(top), int(left))] * B
        base = window_reference(img01, placements, p, S)
        shape_mask = patch.shape_mask
        gh = gw = S
    else:
        base = img01
        shape_mask = None
        gh, gw = H, W

    csf = csf_mod.csf_map(gh, gw, geometry, csf_model, device)
    cscale, n_floored = contrast_scale(base, contrast_mean, shape_mask)

    log(f"\n[spectral] band sweep — {region} region, {normalise}-normalised "
        f"to {target:g}"
        + (" JND" if normalise == "visibility" else " rms"))
    log(f"[spectral] contrast mean: {contrast_mean}"
        + (f"  (mu in [{float(2.0 / cscale.max()):.3f}, "
           f"{float(2.0 / cscale.min()):.3f}])" if contrast_mean == "local"
           else "  (mu = 0.5, matching the attack modes)"))
    if n_floored:
        log(f"[spectral] WARNING   : local mean hit the {MU_FLOOR} floor in "
            f"{n_floored} sample-channels — that window is near-black and its "
            f"visibility is under-stated.")
    log(f"[spectral] {'band (cyc/px)':<16s}{'flip_all':>10s}{'flip_rem':>10s}"
        f"{'realised':>10s}{'rms':>9s}")

    out = []
    for lo, hi in bands:
        flips_all, flips_rem, realised, rmss = [], [], [], []
        for _ in range(n_probes):
            raw = band_limited_noise(B, gh, gw, lo, hi, device)
            d = scale_to_cost(raw, csf, target, normalise, beta, cscale,
                              shape_mask)
            d = csf_mod.fit_to_range(d, base, mask=shape_mask)
            stim = (base + d).clamp(0.0, 1.0)

            # What SURVIVED the clamp, which is what the observer sees and what
            # the network is fed. csf.py makes the same distinction between an
            # intended tau and a realised one, for the same reason.
            eff = csf_mod._masked(stim - base, shape_mask)
            realised.append(float(csf_mod.visibility_index(
                eff, csf, reduce="minkowski", beta=beta,
                contrast_scale=cscale).mean()))
            rmss.append(float(eff.pow(2).mean().sqrt()))

            if region == "patch":
                patched, fp = composite_batch(imgs, stim, placements, p,
                                              mean_t, std_t)
            else:
                patched = (stim - mean_t) / std_t
                fp = torch.zeros(B, H, W, dtype=torch.bool, device=device)

            adv = upsample_to(model(patched), (H, W)).argmax(1)
            changed = adv != clean
            flips_all.append(float(changed.float().mean()) * 100.0)
            outside = ~fp
            flips_rem.append(float((changed & outside).sum())
                             / max(int(outside.sum()), 1) * 100.0)

        # std() over a single probe is NaN, which would silently reach the JSON
        # and then every plot's error bar. Zero is the honest value: one draw
        # carries no spread information.
        def _spread(xs):
            t = torch.tensor(xs)
            return float(t.std()) if t.numel() > 1 else 0.0

        row = {"lo": lo, "hi": hi,
               "centre": 0.5 * (lo + hi),
               "centre_cpd": geometry.to_cycles_per_degree(
                   torch.tensor(0.5 * (lo + hi))).item(),
               "flip_all": float(torch.tensor(flips_all).mean()),
               "flip_all_std": _spread(flips_all),
               "flip_remote": float(torch.tensor(flips_rem).mean()),
               "flip_remote_std": _spread(flips_rem),
               "realised": float(torch.tensor(realised).mean()),
               "rms": float(torch.tensor(rmss).mean()),
               "probes": flips_rem}
        out.append(row)
        short = "" if row["realised"] >= 0.9 * target else "  <- range-limited"
        log(f"           {lo:.2f}-{hi:<11.2f}{row['flip_all']:9.2f}%"
            f"{row['flip_remote']:9.2f}%{row['realised']:10.3f}"
            f"{row['rms']:9.4f}{short}")

    return out


def summarise(per_image: Sequence[Sequence[Dict]], target: float,
              normalise: str = "visibility",
              min_signal: float = 0.5, log=print) -> Dict:
    r"""
    Aggregate per-image band sweeps and answer the image-boundness question.

    THE TEST. If the readable band is a MODEL property it behaves like the ERF:
    the peak band is the same for every image and the spread across images is
    small. If it is SCENE-DEPENDENT the peak moves, and a single fixed budget is
    the wrong object to be optimising against.

    Reported rather than inferred:
      peak_band_per_image   which band flipped the most pixels, per image
      peak_agreement        fraction of images agreeing with the modal peak
      cv                    per-band coefficient of variation across images

    `min_signal` GUARDS EVERY VERDICT, and it is not a formality. At a small
    tau the residual can be well under 1/255 and no band moves the prediction
    at all — and then argmax over six near-zero rates still returns a band,
    every image still "agrees" with it, and the top band still "beats" the
    bottom one. A first run of this probe printed exactly that: `the gap is
    REAL: the top band flips 0.00% vs 0.00%`. Below `min_signal` percent of
    remote pixels the run is declared INCONCLUSIVE and no claim is made in
    either direction; raise --target and run it again.
    """
    n_img = len(per_image)
    if n_img == 0:
        raise ValueError("no per-image results to summarise")
    n_bands = len(per_image[0])

    rates = torch.tensor([[img[b]["flip_remote"] for b in range(n_bands)]
                          for img in per_image])                  # [n_img, n_bands]
    mean = rates.mean(dim=0)
    std = rates.std(dim=0) if n_img > 1 else torch.zeros(n_bands)
    cv = std / mean.clamp(min=1e-9)

    peaks = rates.argmax(dim=1).tolist()
    modal = max(set(peaks), key=peaks.count)
    agreement = peaks.count(modal) / n_img

    bands = [(per_image[0][b]["lo"], per_image[0][b]["hi"])
             for b in range(n_bands)]
    realised = torch.tensor([[img[b]["realised"] for b in range(n_bands)]
                             for img in per_image]).mean(dim=0)

    conclusive = float(mean.max()) >= min_signal

    log(f"\n{'=' * 72}")
    log(f" BAND SENSITIVITY — averaged over {n_img} images")
    log(f"{'=' * 72}")
    log(f"    {'band (cyc/px)':<16s}{'flip_remote':>13s}{'+/-':>8s}"
        f"{'CV':>8s}{'realised':>11s}")
    for b, (lo, hi) in enumerate(bands):
        flag = "" if realised[b] >= 0.9 * target else "  range-limited"
        cvs = f"{cv[b]:8.2f}" if mean[b] >= min_signal else f"{'-':>8s}"
        bar = "#" * int(mean[b])
        log(f"    {lo:.2f}-{hi:<11.2f}{mean[b]:12.2f}%{std[b]:8.2f}"
            f"{cvs}{realised[b]:11.3f}{flag}"
            + (f"   {bar}" if bar else ""))

    if not conclusive:
        log(f"\n  INCONCLUSIVE — the strongest band moved only "
            f"{float(mean.max()):.3f}% of remote pixels, below the "
            f"{min_signal:g}% floor.")
        log("  No band is distinguishable from any other at this cost, so no "
            "claim is made about")
        log("  the gap or about scene-dependence. The stimulus is too weak to "
            "measure, which is")
        log("  itself informative if the target was a realistic tau: raise "
            "--target and re-run,")
        log("  and report the smallest target that produces a measurable "
            "response.")
        return {"bands": [{"lo": lo, "hi": hi,
                           "flip_remote": float(mean[b]),
                           "std": float(std[b]), "cv": float(cv[b]),
                           "realised": float(realised[b])}
                          for b, (lo, hi) in enumerate(bands)],
                "peak_band_per_image": peaks,
                "modal_peak": None,
                "peak_agreement": None,
                "image_bound": None,
                "band_stability": "inconclusive",
                "verdict": "inconclusive",
                "max_flip_remote": float(mean.max()),
                "min_signal": min_signal}

    log(f"\n  peak band per image : {peaks}")
    log(f"  modal peak          : band {modal} "
        f"({bands[modal][0]:.2f}-{bands[modal][1]:.2f} cyc/px)")
    log(f"  agreement           : {100 * agreement:.0f}% of images")
    stable = agreement >= 0.8
    if stable:
        log("  -> the readable band is STABLE across scenes: it behaves like a")
        log("     MODEL property, as the ERF does, and one fixed budget is the")
        log("     right object to optimise against.")
    else:
        log("  -> the readable band MOVES between scenes: it is SCENE-DEPENDENT,")
        log("     and a single fixed budget is optimising against the wrong")
        log("     target for some images. Report the distribution, not a mean.")

    hi_band, lo_band = n_bands - 1, 0
    gap = mean[hi_band] > mean[lo_band]
    if gap:
        log(f"\n  -> the gap is REAL: the top band ({bands[hi_band][0]:.2f}-"
            f"{bands[hi_band][1]:.2f}) flips {mean[hi_band]:.2f}% vs "
            f"{mean[lo_band]:.2f}% at the bottom,")
        log("     at equal perceptual cost. Perturbation the eye discards "
            "still moves the network.")
    else:
        log(f"\n  -> NO GAP at the top: the highest band flips "
            f"{mean[hi_band]:.2f}% vs {mean[lo_band]:.2f}% at the bottom.")
        log("     The network does not read what the eye discards — check the "
            "backbone's input stride")
        log("     before attributing this to the CSF.")

    return {"bands": [{"lo": lo, "hi": hi,
                       "flip_remote": float(mean[b]),
                       "std": float(std[b]), "cv": float(cv[b]),
                       "realised": float(realised[b])}
                      for b, (lo, hi) in enumerate(bands)],
            "peak_band_per_image": peaks,
            "modal_peak": modal,
            "peak_agreement": agreement,
            "image_bound": not stable,
            "band_stability": "stable" if stable else "scene_dependent",
            "verdict": ("efficiency" if normalise == "rms"
                        else ("gap" if gap else "no_gap")),
            "normalise": normalise,
            "efficiency_ratio": (float(realised[lo_band])
                                 / max(float(realised[hi_band]), 1e-9)),
            "max_flip_remote": float(mean.max()),
            "min_signal": min_signal}


def plot_bands(summary: Dict, out_path, title="Frequency sensitivity",
               target: float = 0.25):
    """Standalone figure, own title and legend — thesis-ready without cropping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["bands"]
    x = [0.5 * (r["lo"] + r["hi"]) for r in rows]
    y = [r["flip_remote"] for r in rows]
    e = [r["std"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x, y, yerr=e, fmt="-o", color="#4c72b0", capsize=3,
                label="prediction change")
    limited = [i for i, r in enumerate(rows) if r["realised"] < 0.9 * target]
    if limited:
        ax.scatter([x[i] for i in limited], [y[i] for i in limited],
                   marker="x", s=90, color="#c44e52", zorder=5,
                   label="range-limited (could not reach target cost)")
    ax.set_xlabel("radial spatial frequency (cycles/pixel)")
    ax.set_ylabel("remote prediction-change rate (%)")
    ax.set_title(f"{title}\nband-limited noise at equal perceptual cost "
                 f"({target:g} JND) — label-free")
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
