r"""
Reporting for the universal CSF-bounded patch — where the value of this
experiment actually is.

The headline attack numbers (drop_remote, any_flip_rate) are computed by the
same evaluate() every other mode uses, and say nothing specific about this one.
What is specific is three questions, and a null result answers all three just as
informatively as a success does:

  REALISED TAU   tau is set ONCE, at a fixed reference luminance, because a
                 universal patch cannot look at the image it lands on. What did
                 that one residual actually cost on each val image, measured at
                 that image's own luminance? The sentence worth being able to
                 write is "the budget was set at tau = a using a fixed
                 reference; realised visibility across the val set ranged from
                 b to c". That spread IS the empirical case for content
                 dependence, or the evidence against it.

  SPECTRAL       Radially averaged amplitude of the converged residual against
  ALLOCATION     the budget it was allowed. The interesting quantity is the
                 RATIO -- where the optimiser chose to spend relative to what
                 it could have. The measured ~47x visibility-efficiency
                 advantage at Nyquist predicts it loads high frequencies; this
                 is what confirms or refutes that.

  CLIPPING       clip(x + delta) truncates where content is near 0 or 1, so the
                 realised perturbation is not the projected one and tau is
                 violated in the permissive direction. Non-negligible
                 frac_clipped invalidates the guarantee quietly, which is the
                 worst way for it to fail.

CONFIDENCE INTERVALS ON EVERYTHING, with n reported. A single val-set mean is a
point estimate over a small number of scenes with large scene-to-scene variance
-- overfit.py's --seeds note documents a 9.6-point swing from the same
invocation. A headline number without an interval invites a comparison the data
does not support.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..patch import csf as csf_mod


def mean_ci(x, confidence: float = 0.95) -> dict:
    r"""
    Mean with a normal-approximation confidence interval, and n.

    Deliberately the normal approximation rather than a t-interval or a
    bootstrap: n here is the val set (hundreds), where the three agree to well
    inside the width that matters, and an interval nobody can recompute by hand
    from mean/sd/n is worse than a slightly conservative one.

    n = 1 returns a null interval rather than a zero-width one. A zero-width
    interval reads as "measured exactly", which is the opposite of what one
    sample establishes.
    """
    t = torch.as_tensor(x, dtype=torch.float64).flatten()
    n = int(t.numel())
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci_lo": None, "ci_hi": None}
    mean = float(t.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "ci_lo": None, "ci_hi": None}
    sd = float(t.std(unbiased=True))
    z = 1.959963985 if abs(confidence - 0.95) < 1e-9 else 2.575829304
    half = z * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd,
            "ci_lo": mean - half, "ci_hi": mean + half}


def distribution(x) -> dict:
    """median / 5th / 95th / min / max, alongside mean_ci's mean and interval."""
    t = torch.as_tensor(x, dtype=torch.float64).flatten()
    if t.numel() == 0:
        return {"n": 0}
    q = torch.quantile(t, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    out = mean_ci(t)
    out.update({"median": float(q[1]), "p5": float(q[0]), "p95": float(q[2]),
                "min": float(t.min()), "max": float(t.max())})
    return out


@torch.no_grad()
def realised_tau(patch, loader, device, mean_t, std_t, max_images: int = 0
                 ) -> dict:
    r"""
    Per-image realised visibility of the SHARED residual, both conventions.

      tau_nominal_i     the legacy mu = 0.5 convention -- identical for every
                        image, because nothing in it depends on the image. It
                        is reported anyway, because it is the number every
                        existing run is quoted in and the one the budget was
                        enforced under.
      tau_calibrated_i  the same residual scored against image i's OWN measured
                        luminance, linearised properly, with Barten's retinal
                        illuminance following it. This is what the residual
                        really costs on that scene.

    The gap between them is the calibration error (~1.9x on this data); the
    SPREAD of tau_calibrated across images is the content-dependence.

    Also returns frac_clipped per batch, because a large realised tau and a
    large frac_clipped mean opposite things: the first is the residual costing
    more than intended, the second is it being truncated to less than intended.
    """
    cfg = patch.cfg
    if cfg.mode != "universal_csf":
        raise ValueError(f"realised_tau is universal_csf only, got {cfg.mode!r}")

    delta = patch.residual().detach()
    p = delta.shape[-1]
    top, left = patch.placement if patch.placement is not None else (0, 0)

    nominal = float(csf_mod.visibility_index(
        delta, patch._csf_values, beta=cfg.csf_beta,
        contrast_scale=patch._contrast_scale))

    taus, lums, clipped, seen = [], [], [], 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        img01 = (imgs * std_t + mean_t).clamp(0, 1)
        win = img01[:, :, top:top + p, left:left + p]

        t_cal, Y = csf_mod.calibrated_visibility(
            delta, win, patch._geometry, cfg.csf_model, cfg.csf_beta)
        taus.append(t_cal.cpu())
        lums.append(Y.cpu())

        raw_win = win + delta
        clipped.append(((raw_win < 0.0) | (raw_win > 1.0)
                        ).float().flatten(1).mean(1).cpu())

        seen += imgs.shape[0]
        if max_images and seen >= max_images:
            break

    taus = torch.cat(taus); lums = torch.cat(lums); clipped = torch.cat(clipped)
    return {"tau_requested": cfg.csf_threshold,
            "tau_nominal_legacy": nominal,
            "tau_calibrated": distribution(taus),
            "footprint_luminance_Y": distribution(lums),
            "frac_clipped": distribution(clipped),
            "calibration_factor_median": (float(taus.median()) / nominal
                                          if nominal > 0 else None),
            # THE RAW VALUES, not just the summary. plot() drew an empty axis
            # for one commit because it had only the quantiles to work with and
            # nothing to histogram -- an axis labelled "images" with no images
            # in it. They are also what lets anyone re-plot or re-test the
            # distribution later without re-running the attack.
            "per_image_tau_calibrated": taus.tolist(),
            "per_image_Y": lums.tolist()}


@torch.no_grad()
def spectral_allocation(patch, n_bins: int = 32) -> dict:
    r"""
    Radially averaged residual amplitude, the budget, and their RATIO.

    The ratio is the deliverable. An amplitude profile alone mostly reproduces
    the 1/CSF envelope it was projected onto and says little; the ratio says
    where, within what it was allowed, the optimiser actually chose to spend.

      ratio -> 1 across the band   the bound is active everywhere; the attack
                                   is limited by tau and not by the optimiser
      ratio peaked at high f       the predicted behaviour, and what the ~47x
                                   efficiency advantage at Nyquist implies
      ratio peaked at low f        the premise is wrong for this model -- the
                                   network is not reading the cheap band, so
                                   the attack pays for visibility it cannot use
    """
    delta = patch.residual().detach()
    f, amp = csf_mod.radial_profile(delta, n_bins)
    fb, bud = csf_mod.budget_radial_profile(patch._csf_budget, n_bins)
    amp = amp[0]
    live = bud > 0
    ratio = torch.zeros_like(amp)
    ratio[live] = amp[live] / bud[live]
    r = ratio[live]
    spread = float(r.max() - r.min())
    # A FLAT RATIO HAS NO PEAK, and argmax of a constant returns bin 0 -- which
    # reads as "the optimiser chose the lowest frequency" when in fact it chose
    # nothing. That happens whenever the run is pinned to the bound everywhere
    # (lr far too high, or the frozen-spectrum bug), and it is the single most
    # misleading thing this figure could say. Report the degeneracy instead.
    degenerate = spread < 0.02
    return {"f_cyc_per_px": f.tolist(),
            "residual_amplitude": amp.tolist(),
            "budget_amplitude": bud.tolist(),
            "spend_ratio": ratio.tolist(),
            "live_bins": int(live.sum()),
            "peak_spend_f": (None if degenerate
                             else float(f[int(ratio.argmax())])),
            "mean_spend_ratio": float(r.mean()),
            "spend_ratio_min": float(r.min()),
            "spend_ratio_max": float(r.max()),
            "spend_ratio_spread": spread,
            "degenerate": degenerate}


def plot(report: dict, alloc: dict, path, title: str = ""):
    """
    Three panels: the realised-tau DISTRIBUTION, amplitude, and the spend ratio.

    The ratio gets its own axis because on a log amplitude plot the residual
    and the budget lie on top of each other whenever the bound is active, which
    is most of the time -- two curves that coincide look like one curve and
    hide the only quantity worth reading.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    d = report["tau_calibrated"]

    # ── realised tau ─────────────────────────────────────────────────────────
    vals = report.get("per_image_tau_calibrated")
    if vals:
        ax[0].hist(vals, bins=min(30, max(8, len(vals) // 4)),
                   color="#3b6ea5", alpha=0.85)
    ax[0].axvline(report["tau_nominal_legacy"], color="#c44", ls="--", lw=1.5,
                  label=f"nominal (legacy) {report['tau_nominal_legacy']:.3f}")
    ax[0].axvline(d["median"], color="#222", ls="-", lw=1.4,
                  label=f"realised median {d['median']:.3f}")
    for k in ("p5", "p95"):
        ax[0].axvline(d[k], color="#888", ls=":", lw=1.2)
    ax[0].set_xlabel("realised tau on each val image (calibrated)")
    ax[0].set_ylabel("images")
    ax[0].set_title(f"n={d['n']}  median {d['median']:.3f}  "
                    f"5-95 pct {d['p5']:.3f}-{d['p95']:.3f}")
    ax[0].legend(fontsize=8)

    f = alloc["f_cyc_per_px"]
    ax[1].semilogy(f, alloc["budget_amplitude"], color="#888", lw=3,
                   alpha=0.55, label="budget (allowed)")
    ax[1].semilogy(f, alloc["residual_amplitude"], color="#3b6ea5", lw=1.4,
                   label="residual (spent)")
    ax[1].set_xlabel("radial frequency (cycles/pixel)")
    ax[1].set_ylabel("mean |rfft| amplitude")
    ax[1].set_title("amplitude — these coincide when the bound is active")
    ax[1].legend(fontsize=8)

    # ── the deliverable ──────────────────────────────────────────────────────
    ax[2].plot(f, alloc["spend_ratio"], color="#a5713b")
    ax[2].axhline(1.0, color="#888", ls="--", lw=1)
    ax[2].set_ylim(0, 1.1)
    ax[2].set_xlabel("radial frequency (cycles/pixel)")
    ax[2].set_ylabel("spent / allowed")
    if alloc.get("degenerate"):
        ax[2].set_title(f"FLAT (spread {alloc['spend_ratio_spread']:.3f}) — "
                        f"pinned to the bound, no allocation was chosen")
        msg = "no spectral preference expressed" + chr(10) +               "check frac_at_bound and lr"
        ax[2].text(0.5, 0.45, msg,
                   ha="center", va="center", transform=ax[2].transAxes,
                   fontsize=9, color="#c44")
    else:
        ax[2].set_title(f"peak spend at {alloc['peak_spend_f']:.3f} cyc/px  "
                        f"(range {alloc['spend_ratio_min']:.2f}-"
                        f"{alloc['spend_ratio_max']:.2f})")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def log_report(report: dict, alloc: dict, log=print):
    """The console summary. Read the spread, not the mean."""
    d = report["tau_calibrated"]
    log(f"\n[univ] tau requested   : {report['tau_requested']:g}")
    log(f"[univ] tau nominal     : {report['tau_nominal_legacy']:.4f} "
        f"(legacy mu=0.5, identical on every image by construction)")
    log(f"[univ] tau CALIBRATED  : median {d['median']:.4f}   "
        f"5-95 pct {d['p5']:.4f}-{d['p95']:.4f}   "
        f"range {d['min']:.4f}-{d['max']:.4f}   n={d['n']}")
    log(f"[univ]                   mean {d['mean']:.4f} "
        f"95% CI [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")
    log(f"[univ] calibration     : realised/nominal = "
        f"{report['calibration_factor_median']:.2f}x at the median")
    lum = report["footprint_luminance_Y"]
    log(f"[univ] footprint Y     : median {lum['median']:.4f}   "
        f"5-95 pct {lum['p5']:.4f}-{lum['p95']:.4f}")
    fc = report["frac_clipped"]
    log(f"[univ] frac_clipped    : median {fc['median']:.5f}   "
        f"max {fc['max']:.5f}")
    if fc["max"] > 0.01:
        log("[univ] WARNING        : clipping is active on >1% of footprint "
            "pixels somewhere in the val set.")
        log("        The realised residual differs from the projected one "
            "THERE, and tau is violated")
        log("        in the permissive direction. Compare against "
            "--csf_composite fit before quoting tau.")
    if alloc.get("degenerate"):
        log(f"[univ] spectral spend  : FLAT at {alloc['mean_spend_ratio']:.3f} "
            f"across all {alloc['live_bins']} live bins "
            f"(spread {alloc['spend_ratio_spread']:.4f})")
        log("[univ] WARNING        : the residual is pinned to its bound at "
            "every frequency, so NO spectral")
        log("        allocation was chosen and the peak is undefined. Either "
            "the learning rate is far")
        log("        too large -- check frac_at_bound in the epoch logs, "
            "pinned above ~0.9 means yes --")
        log("        or the attack genuinely wants the maximum everywhere. "
            "Lower lr and re-run before")
        log("        reading anything into the spectrum.")
    else:
        log(f"[univ] spectral spend  : peak at {alloc['peak_spend_f']:.3f} "
            f"cyc/px, mean ratio {alloc['mean_spend_ratio']:.3f} "
            f"(range {alloc['spend_ratio_min']:.2f}-"
            f"{alloc['spend_ratio_max']:.2f}) over "
            f"{alloc['live_bins']} live bins")
