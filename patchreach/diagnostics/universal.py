"""Reporting for the universal CSF-bounded patch — where the value of this experiment actually is."""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..patch import csf as csf_mod


def mean_ci(x, confidence: float = 0.95) -> dict:
    """Mean with a normal-approximation confidence interval, and n."""
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
    """Per-image realised visibility of the SHARED residual, both conventions."""
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
                                          if nominal > 0 else None)}


@torch.no_grad()
def spectral_allocation(patch, n_bins: int = 32) -> dict:
    """Radially averaged residual amplitude, the budget, and their RATIO."""
    delta = patch.residual().detach()
    f, amp = csf_mod.radial_profile(delta, n_bins)
    fb, bud = csf_mod.budget_radial_profile(patch._csf_budget, n_bins)
    amp = amp[0]
    live = bud > 0
    ratio = torch.zeros_like(amp)
    ratio[live] = amp[live] / bud[live]
    return {"f_cyc_per_px": f.tolist(),
            "residual_amplitude": amp.tolist(),
            "budget_amplitude": bud.tolist(),
            "spend_ratio": ratio.tolist(),
            "live_bins": int(live.sum()),
            "peak_spend_f": float(f[int(ratio.argmax())]),
            "mean_spend_ratio": float(ratio[live].mean())}


def plot(report: dict, alloc: dict, path, title: str = ""):
    """Two panels: the realised-tau distribution, and spectral allocation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

    d = report["tau_calibrated"]
    ax[0].axvline(report["tau_nominal_legacy"], color="#c44", ls="--", lw=1.5,
                  label=f"nominal (legacy) {report['tau_nominal_legacy']:.3f}")
    for k, c in (("p5", "#888"), ("median", "#222"), ("p95", "#888")):
        ax[0].axvline(d[k], color=c, ls=":", lw=1.2)
    ax[0].set_xlabel("realised tau on each val image (calibrated)")
    ax[0].set_ylabel("images")
    ax[0].set_title(f"n={d['n']}  median {d['median']:.3f}  "
                    f"5-95 pct {d['p5']:.3f}-{d['p95']:.3f}")
    ax[0].legend(fontsize=8)

    f = alloc["f_cyc_per_px"]
    ax[1].semilogy(f, alloc["budget_amplitude"], color="#888",
                   label="budget (allowed)")
    ax[1].semilogy(f, alloc["residual_amplitude"], color="#3b6ea5",
                   label="residual (spent)")
    ax[1].set_xlabel("radial frequency (cycles/pixel)")
    ax[1].set_ylabel("mean |rfft| amplitude")
    ax[1].set_title(f"spend/allowed peaks at {alloc['peak_spend_f']:.3f} cyc/px")
    ax[1].legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def log_report(report: dict, alloc: dict, log=print):
    """The console summary."""
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
    log(f"[univ] spectral spend  : peak at {alloc['peak_spend_f']:.3f} cyc/px, "
        f"mean ratio {alloc['mean_spend_ratio']:.3f} over "
        f"{alloc['live_bins']} live bins")
