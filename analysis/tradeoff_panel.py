#!/usr/bin/env python
"""The ATTACK-SUCCESS / VISIBILITY TRADE-OFF, one panel per architecture."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchreach.metrics.population import describe

CSF_MODES = ("csf", "universal_csf")
CEILING_MODES = ("raw",)

METRIC_LABEL = {
    "any_flip_rate": "remote pixels flipped (%)",
    "drop_remote": "remote mIoU drop (points)",
    "best_drop_remote": "best remote mIoU drop (points)",
    "target_hit_rate": "remote pixels hit target (%)",
    "final_visibility": "realised visibility (JND)",
    "final_visibility_local": "realised visibility, local contrast (JND)",
}

METRIC_SHORT = {
    "any_flip_rate": "remote flip rate",
    "drop_remote": "remote mIoU drop",
    "best_drop_remote": "best remote mIoU drop",
    "target_hit_rate": "target hit rate",
    "final_visibility": "realised visibility",
    "final_visibility_local": "realised visibility, local contrast",
}

AXES = ("loss_fn", "steps", "lr", "lr_schedule", "placement", "csf_param",
        "csf_enforce", "img_h", "img_w", "patch_scale", "image", "n_images",
        "sample_seed", "csf_lref")


def load_run(d: Path) -> dict | None:
    """(config, rows, kind) for one run directory, or None if it is not one."""
    cfg, rows, kind = None, None, None

    cfg_p = d / "config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text())

    s, r = d / "summary.json", d / "results.json"
    if s.exists():
        blob = json.loads(s.read_text())
        cfg = cfg or blob.get("config")
        if blob.get("records"):
            rows, kind = blob["records"], "population"
        elif blob.get("per_seed"):
            rows, kind = blob["per_seed"], "seeds"
    if rows is None and r.exists():
        blob = json.loads(r.read_text())
        cfg = cfg or blob.get("config")
        rows, kind = [blob], "single"

    if not rows or not cfg:
        return None
    return {"dir": d, "cfg": cfg, "rows": rows, "kind": kind}


def stat(rows: list, key: str, seed: int = 0) -> dict | None:
    """(mean, lo, hi, n) for one metric over a run's rows."""
    vals = [float(x[key]) for x in rows if x.get(key) is not None]
    if not vals:
        return None
    v = np.asarray(vals, dtype=np.float64)
    if v.size >= 8:
        d = describe(v, seed=seed)
        lo, hi = d["ci95_boot"]
        return {"mean": d["mean"], "lo": lo, "hi": hi, "n": int(v.size),
                "band": "bootstrap 95% CI"}
    if v.size > 1:
        m, sd = float(v.mean()), float(v.std(ddof=1))
        return {"mean": m, "lo": m - sd, "hi": m + sd, "n": int(v.size),
                "band": "+/- 1 sd over seeds"}
    return {"mean": float(v[0]), "lo": None, "hi": None, "n": 1,
            "band": "single run, no interval"}


def from_sweep(sweep_dir: Path, metrics: list, seed: int):
    """The tau ladder of a scripts/sweep_operating_point.py sweep, AT ITS OWN OPERATING POINT, plus any raw arm dropped into the same cells/ directory."""
    man_p = sweep_dir / "sweep.json"
    if not man_p.exists():
        return [], [], [(sweep_dir, "no sweep.json -- not a sweep directory")]
    man = json.loads(man_p.read_text())
    cells = sweep_dir / "cells"
    if not cells.is_dir():
        return [], [], [(sweep_dir, "no cells/ directory")]

    dirs = sorted(d for d in cells.iterdir() if d.is_dir())
    curve, ceiling, skipped = collect(dirs, metrics, seed)

    ops = {loss: d["operating_point"]
           for loss, d in (man.get("decisions") or {}).items()
           if isinstance(d, dict) and d.get("operating_point")}
    if not ops:
        print(f"  {sweep_dir.name}: the manifest records no operating point "
              f"(the lr/steps stages did not\n    finish), so every csf cell "
              f"is taken. If the sweep ran more than its tau stage,\n    the "
              f"curve now mixes learning rates -- check the warning below.")
        return curve, ceiling, skipped

    keep = []
    for p in curve:
        ax = p["axes"]
        for loss, op in ops.items():
            if (ax.get("loss_fn") == loss
                    and ax.get("lr") is not None
                    and abs(float(ax["lr"]) - float(op["lr"])) < 1e-12
                    and int(ax.get("steps", -1)) == int(op["steps"])
                    and ax.get("csf_enforce") == op.get("enforce")):
                keep.append(p)
                break
    dropped = len(curve) - len(keep)
    op_txt = "; ".join(f"{k}: lr {v['lr']:g}, {v['steps']} steps, "
                       f"enforce {v['enforce']}" for k, v in ops.items())
    print(f"  {sweep_dir.name}: {len(keep)} cells at the operating point "
          f"({op_txt});\n    {dropped} off-operating-point cells ignored.")
    return keep, ceiling, skipped


def collect(dirs: list, metrics: list, seed: int):
    """Every run directory, sorted into curve points and ceiling points."""
    curve, ceiling, skipped = [], [], []
    for d in dirs:
        run = load_run(d)
        if run is None:
            skipped.append((d, "no results.json / summary.json"))
            continue
        cfg, rows = run["cfg"], run["rows"]
        mode = cfg.get("patch_mode")
        pt = {"dir": str(d), "kind": run["kind"], "arch": cfg.get("arch"),
              "mode": mode, "tag": cfg.get("tag", ""),
              "axes": {k: cfg.get(k) for k in AXES if k in cfg},
              "n_rows": len(rows)}
        for m in set(metrics) | {"final_visibility", "drop_remote"}:
            pt[m] = stat(rows, m, seed=seed)

        if mode in CSF_MODES:
            pt["tau"] = float(cfg.get("csf_threshold"))
            pt["enforce"] = cfg.get("csf_enforce", "nominal")
            curve.append(pt)
        elif mode in CEILING_MODES:
            ceiling.append(pt)
        else:
            skipped.append((d, f"--patch_mode {mode} is neither csf nor raw"))
    return curve, ceiling, skipped


def knee(points: list, slack: float) -> dict:
    """The largest tau the run actually held, and the rule that decided it."""
    rows = sorted([p for p in points if p.get("final_visibility")],
                  key=lambda p: p["tau"])
    if not rows:
        return {"tau_max_tracking": None,
                "rule": "no realised visibility recorded; nothing to check"}

    modes = {p.get("enforce") for p in rows}
    ladder = [{"tau": p["tau"], "realised": p["final_visibility"]["mean"]}
              for p in rows]

    if modes == {"realised"}:
        ok = []
        for x in ladder:
            x["ratio"] = x["realised"] / max(x["tau"], 1e-12)
            x["tracking"] = abs(x["ratio"] - 1.0) <= slack
            if x["tracking"]:
                ok.append(x["tau"])
        return {"tau_max_tracking": max(ok) if ok else None, "ladder": ladder,
                "rule": f"realised visibility within {slack:.0%} of the "
                        f"request (--csf_enforce realised)"}

    held = ladder[0]["tau"]
    for prev, cur in zip(ladder, ladder[1:]):
        cur["grew"] = cur["realised"] > prev["realised"] * (1.0 + slack)
        if not cur["grew"]:
            break
        held = cur["tau"]
    modestr = "/".join(sorted(str(m) for m in modes))
    return {"tau_max_tracking": held, "ladder": ladder,
            "rule": f"last rung whose realised visibility grew by more than "
                    f"{slack:.0%} over the rung below (--csf_enforce "
                    f"{modestr})"}


def dedup_identical(points: list) -> tuple:
    """Collapse points that are the SAME CONFIGURATION at the same tau."""
    by_key: dict = {}
    for p in points:
        key = (p["tau"], json.dumps(p["axes"], sort_keys=True, default=str))
        prev = by_key.get(key)
        if prev is None or p["n_rows"] > prev["n_rows"]:
            by_key[key] = p
    kept = list(by_key.values())
    return kept, len(points) - len(kept)


def axis_drift(points: list) -> dict:
    """Fields that move between points of one curve."""
    drift = {}
    for k in AXES:
        vals = {json.dumps(p["axes"].get(k)) for p in points if k in p["axes"]}
        vals = {v for v in vals if v != "null"}
        if len(vals) > 1:
            drift[k] = sorted(json.loads(v) for v in vals)
    return drift


def draw(by_arch: dict, metrics: list, x_key: str, normalise: bool,
         slack: float, out: Path, title, ceiling_pick: str,
         share_y: bool) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = list(by_arch)
    nrow, ncol = len(metrics), len(archs)

    bands = {p[m]["band"] for d in by_arch.values() for p in d["curve"]
             for m in metrics if p.get(m)}
    n_ceil = {len(d["ceiling"]) for d in by_arch.values()}
    band_txt = f", {next(iter(bands))}" if len(bands) == 1 else ""
    ceil_txt = (f", strongest of {next(iter(n_ceil))}"
                if len(n_ceil) == 1 and next(iter(n_ceil)) > 1 else "")
    CSF_LABEL = f"csf (CSF-bounded residual){band_txt}"
    CEIL_LABEL = f"unconstrained (raw){ceil_txt}"
    SHADE_LABEL = "tau no longer held (range fit binds)"
    unnormalised: list = []
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             sharey=("row" if share_y else False),
                             figsize=(4.3 * ncol, 3.1 * nrow))
    notes = {}

    for j, arch in enumerate(archs):
        pts = sorted(by_arch[arch]["curve"], key=lambda p: p["tau"])
        ceil = by_arch[arch]["ceiling"]
        k = knee(pts, slack)
        notes[arch] = {"knee": k, "axis_drift": axis_drift(pts),
                       "n_curve_points": len(pts),
                       "n_ceiling_runs": len(ceil)}

        for i, m in enumerate(metrics):
            ax = axes[i][j]
            rows = [p for p in pts if p.get(m)]
            if x_key == "realised":
                rows = [p for p in rows if p.get("final_visibility")]
                xs = [p["final_visibility"]["mean"] for p in rows]
            else:
                xs = [p["tau"] for p in rows]

            cy = None
            cand = [c for c in ceil if c.get(m)]
            if cand:
                if ceiling_pick == "best":
                    best = max(cand, key=lambda c: c[m]["mean"])
                else:
                    best = max(cand, key=lambda c: c[m]["n"])
                cy = best[m]["mean"]
                den = cy if (normalise and abs(cy) > 1e-9) else 1.0
                ax.axhline(cy / den, color="#c44e52", lw=1.6, ls="--",
                           label=CEIL_LABEL, zorder=1)
                if best[m]["lo"] is not None:
                    ax.axhspan(best[m]["lo"] / den, best[m]["hi"] / den,
                               color="#c44e52", alpha=0.12, zorder=0)

            den = cy if (normalise and cy and abs(cy) > 1e-9) else 1.0
            if normalise and den == 1.0:
                unnormalised.append(f"{arch}/{m}")
            if rows:
                ys = np.array([p[m]["mean"] / den for p in rows])
                lo = np.array([(p[m]["lo"] if p[m]["lo"] is not None
                                else p[m]["mean"]) / den for p in rows])
                hi = np.array([(p[m]["hi"] if p[m]["hi"] is not None
                                else p[m]["mean"]) / den for p in rows])
                ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys],
                            marker="o", ms=5, lw=1.8, capsize=3,
                            color="#4c72b0", ecolor="#4c72b0", elinewidth=1,
                            zorder=3, label=CSF_LABEL)

            if x_key == "tau" and k.get("tau_max_tracking") and xs:
                tmax = k["tau_max_tracking"]
                if tmax < max(xs):
                    ax.axvspan(tmax, max(xs) * 1.15, color="#888", alpha=0.13,
                               zorder=0, label=SHADE_LABEL)

            if xs and min(xs) < 1.0 < max(xs) * 1.15:
                ax.axvline(1.0, color="#555", lw=1.0, ls=":", zorder=1,
                           label="1 JND (detection threshold)")

            ax.set_xscale("log")
            if i == 0:
                ns = {p[mm]["n"] for p in pts for mm in metrics if p.get(mm)}
                ax.set_title(arch + (f"   (n={next(iter(ns))})"
                                     if len(ns) == 1 else ""), fontsize=11)
            if j == 0:
                ax.set_ylabel(
                    f"{METRIC_SHORT.get(m, m)}\n(fraction of own ceiling)"
                    if normalise else METRIC_LABEL.get(m, m), fontsize=9)
            if i == nrow - 1:
                ax.set_xlabel("requested tau (JND)" if x_key == "tau"
                              else "realised visibility (JND)", fontsize=9)
            ax.grid(alpha=0.25, lw=0.6)
            ax.tick_params(labelsize=8, labelbottom=True)

            held = [p for p in rows
                    if k.get("tau_max_tracking") is not None
                    and p["tau"] <= k["tau_max_tracking"]]
            if cy and abs(cy) > 1e-9 and held:
                top = max(held, key=lambda p: p["tau"])
                frac = 100.0 * top[m]["mean"] / cy
                notes[arch].setdefault("frac_of_ceiling", {})[m] = {
                    "tau": top["tau"], "value": top[m]["mean"],
                    "ceiling": cy, "percent": frac}
                ax.annotate(f"{frac:.0f}% of ceiling\nat tau={top['tau']:g}",
                            xy=(0.97, 0.04), xycoords="axes fraction",
                            fontsize=8, ha="right", va="bottom", color="#333",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="none", alpha=0.75))

    if unnormalised:
        print(f"\n  WARNING: --normalise, but no unconstrained arm for "
              f"{', '.join(unnormalised)}.")
        print("    Those panels are in RAW UNITS while their neighbours are "
              "fractions, and the row")
        print("    shares one y axis. Run the raw arm, or drop --normalise.")

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    handles, labels = [], []
    for row in axes:
        for ax in row:
            for h, lb in zip(*ax.get_legend_handles_labels()):
                if lb not in labels:
                    handles.append(h)
                    labels.append(lb)
    if handles:
        fig.legend(handles, labels, fontsize=8, frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, 0.02),
                   ncol=min(len(handles), 4))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return notes


def main():
    p = argparse.ArgumentParser(
        description="Attack-success / visibility trade-off panel, one column "
                    "per architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("runs", nargs="*", default=[],
                   help="run directories (globs are fine). csf runs become "
                        "curve points at their --csf_threshold; raw runs "
                        "become the unconstrained ceiling.")
    p.add_argument("--sweep", action="append", default=[], metavar="DIR",
                   help="a scripts/sweep_operating_point.py directory (the "
                        "one holding sweep.json). Repeatable, once per "
                        "architecture. Its tau ladder is read AT THE "
                        "OPERATING POINT the manifest records -- do not glob "
                        "its cells/ by hand, see from_sweep().")
    p.add_argument("--out", default="figures/tradeoff_panel.png",
                   help="PNG path. A .pdf, a .csv of the plotted points and a "
                        ".json of the knees are written beside it.")
    p.add_argument("--metrics", default="any_flip_rate,drop_remote",
                   help="one row per metric. Add final_visibility to show the "
                        "budget saturating in the same figure.")
    p.add_argument("--x", default="tau", choices=["tau", "realised"],
                   dest="x_key",
                   help="tau: what was requested. realised: what was measured "
                        "(final_visibility) -- the honest axis.")
    p.add_argument("--archs", default=None,
                   help="comma-separated order/filter for the columns. "
                        "Default: every arch found, alphabetically.")
    p.add_argument("--normalise", action="store_true",
                   help="plot each curve as a fraction of its own "
                        "architecture's unconstrained ceiling")
    p.add_argument("--ceiling_pick", default="best",
                   choices=["best", "largest_n"],
                   help="which raw run is the ceiling when several exist. "
                        "'best' treats it as the upper bound it is meant to "
                        "be, and is best-of-N -- the count is printed in the "
                        "legend so it cannot be quoted without it.")
    p.add_argument("--slack", type=float, default=0.05,
                   help="fractional deviation still counted as tau being held")
    p.add_argument("--free_y", dest="share_y", action="store_false",
                   default=True,
                   help="let every panel autoscale its own y. The default "
                        "shares y across a row so the architectures are "
                        "actually comparable.")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--title", default=None)
    a = p.parse_args()

    dirs = sorted({Path(x).resolve() for x in a.runs if Path(x).is_dir()})
    sweeps = [Path(x).resolve() for x in a.sweep]
    if not dirs and not sweeps:
        print("no run directories matched, and no --sweep given",
              file=sys.stderr)
        return 2
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]

    curve, ceiling, skipped = collect(dirs, metrics, a.seed)
    for s in sweeps:
        c, ce, sk = from_sweep(s, metrics, a.seed)
        curve += c
        ceiling += ce
        skipped += sk
    for d, why in skipped:
        print(f"  [skip] {d.name}: {why}")
    if not curve:
        print("no csf runs among the directories given -- nothing to plot "
              "against tau", file=sys.stderr)
        return 2

    archs = sorted({p["arch"] for p in curve})
    if a.archs:
        want = [x.strip() for x in a.archs.split(",") if x.strip()]
        missing = [x for x in want if x not in archs]
        if missing:
            print(f"  no csf runs for: {', '.join(missing)}", file=sys.stderr)
        archs = [x for x in want if x in archs]
    by_arch = {}
    for arch in archs:
        pts, collapsed = dedup_identical(
            [p for p in curve if p["arch"] == arch])
        if collapsed:
            print(f"  {arch}: {collapsed} duplicate cell(s) collapsed "
                  f"(same tau, identical configuration)")
        by_arch[arch] = {"curve": pts,
                         "ceiling": [c for c in ceiling if c["arch"] == arch]}

    for arch, d in by_arch.items():
        taus = sorted(p["tau"] for p in d["curve"])
        print(f"\n  {arch}: {len(d['curve'])} csf points at tau "
              f"{', '.join(f'{t:g}' for t in taus)}")
        if not d["ceiling"]:
            print("    NO UNCONSTRAINED ARM. The panel will have no ceiling, "
                  "so it can show that\n    the attack is weak but not "
                  "whether the CONSTRAINT is what weakened it.\n"
                  "    Run --patch_mode raw on the same images.")
        dupes = sorted({t for t in taus if taus.count(t) > 1})
        if dupes:
            print(f"    WARNING: {len(dupes)} tau value(s) appear more than "
                  f"once ({', '.join(f'{t:g}' for t in dupes)}). The line "
                  f"will zig-zag through them.")
        drift = axis_drift(d["curve"])
        if drift:
            print(f"    WARNING: these points differ in more than tau -- "
                  f"{', '.join(drift)}.")
            for key, vals in drift.items():
                print(f"      {key}: {vals}")
            print("    The figure attributes those differences to tau. Fix "
                  "them, or say so in the caption.")

    out = Path(a.out)
    notes = draw(by_arch, metrics, a.x_key, a.normalise, a.slack, out,
                 a.title, a.ceiling_pick, a.share_y)

    rows = []
    for arch, d in by_arch.items():
        for pt in sorted(d["curve"], key=lambda q: q["tau"]) + d["ceiling"]:
            row = {"arch": arch, "mode": pt["mode"], "tau": pt.get("tau"),
                   "kind": pt["kind"], "n": pt["n_rows"], "dir": pt["dir"],
                   **pt["axes"]}
            for m in set(metrics) | {"final_visibility"}:
                if pt.get(m):
                    row[m] = pt[m]["mean"]
                    row[f"{m}_lo"] = pt[m]["lo"]
                    row[f"{m}_hi"] = pt[m]["hi"]
            rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".json").write_text(json.dumps(
        {"metrics": metrics, "x": a.x_key, "slack": a.slack,
         "normalised": a.normalise, "ceiling_pick": a.ceiling_pick,
         "per_arch": notes}, indent=2, default=str))

    print("")
    for arch, n in notes.items():
        k = n["knee"]
        print(f"  {arch}: tau held up to {k.get('tau_max_tracking')}  "
              f"({k['rule']})")
        for m, f in (n.get("frac_of_ceiling") or {}).items():
            print(f"    {m}: {f['percent']:.0f}% of the unconstrained ceiling "
                  f"at tau={f['tau']:g} "
                  f"({f['value']:.2f} vs {f['ceiling']:.2f})")
    print(f"\n  -> {out}\n  -> {out.with_suffix('.pdf')}"
          f"\n  -> {out.with_suffix('.csv')}\n  -> {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
