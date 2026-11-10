"""Statistics over a POPULATION of per-image attacks."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .miou import SegMetric, compare as miou_compare

SUCCESS_THRESHOLDS = (1.0, 5.0, 10.0)


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI for the MEAN."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def describe(values: Sequence[float], seed: int = 0) -> Dict:
    """Full distribution summary — never just a mean."""
    v = np.asarray(values, dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return {"n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    lo, hi = bootstrap_ci(v, seed=seed)
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {"n": n, "mean": mean, "std": std, "se": float(se),
            "ci95_boot": [lo, hi],
            "ci95_t": [mean - 1.96 * se, mean + 1.96 * se],
            "min": float(v.min()), "q1": q1, "median": med, "q3": q3,
            "max": float(v.max()),
            "success_rate": {str(t): float((v > t).mean() * 100.0)
                             for t in SUCCESS_THRESHOLDS}}


def images_needed(std: float, half_width: float = 1.0) -> int:
    """N required for a 95% CI of +/- `half_width` points, at the observed spread."""
    if std <= 0:
        return 1
    return int(np.ceil((1.96 * std / max(half_width, 1e-9)) ** 2))


class Population:
    """Per-image records plus the POOLED confusion matrices."""

    def __init__(self, num_classes: int = 19, device="cpu",
                 target_class: Optional[int] = None):
        self.K = num_classes
        self.target_class = target_class
        self.clean_all = SegMetric(num_classes, device=device)
        self.clean_rem = SegMetric(num_classes, device=device)
        self.adv_all = SegMetric(num_classes, device=device)
        self.adv_rem = SegMetric(num_classes, device=device)
        self.records: List[Dict] = []
        self._flipped = 0
        self._remote = 0
        self._hit = 0
        self._not_target = 0

    @torch.no_grad()
    def update(self, clean_logits, adv_logits, label, footprint,
               record: Dict) -> Dict:
        """Fold one image into the pooled metrics and keep its record."""
        from ..data.cityscapes import upsample_to
        hw = label.shape[-2:]
        pc = upsample_to(clean_logits, hw).argmax(1)
        pa = upsample_to(adv_logits, hw).argmax(1)

        self.clean_all.update(pc, label)
        self.adv_all.update(pa, label)
        self.clean_rem.update(pc, label, exclude=footprint)
        self.adv_rem.update(pa, label, exclude=footprint)

        remote = (label != 255) & (~footprint)
        self._remote += int(remote.sum())
        self._flipped += int((remote & (pc != pa)).sum())
        if self.target_class is not None:
            nt = remote & (pc != self.target_class)
            self._not_target += int(nt.sum())
            self._hit += int((nt & (pa == self.target_class)).sum())

        self.records.append(record)
        return record

    def state_dict(self) -> Dict:
        """Everything needed to continue a run that hit the walltime."""
        return {"clean_all": self.clean_all.cm.cpu(),
                "clean_rem": self.clean_rem.cm.cpu(),
                "adv_all": self.adv_all.cm.cpu(),
                "adv_rem": self.adv_rem.cm.cpu(),
                "flipped": self._flipped, "remote": self._remote,
                "hit": self._hit, "not_target": self._not_target,
                "records": self.records,
                "num_classes": self.K, "target_class": self.target_class}

    def load_state_dict(self, st: Dict):
        if st.get("num_classes") != self.K:
            raise ValueError(
                f"checkpoint has num_classes={st.get('num_classes')} but this "
                f"run uses {self.K} — the confusion matrices are not "
                f"compatible and pooling them would be meaningless")
        if st.get("target_class") != self.target_class:
            raise ValueError(
                f"checkpoint has target_class={st.get('target_class')} but "
                f"this run uses {self.target_class} — the hit-rate denominator "
                f"differs, so the pooled rate would mix two definitions")
        dev = self.clean_all.cm.device
        self.clean_all.cm = st["clean_all"].to(dev)
        self.clean_rem.cm = st["clean_rem"].to(dev)
        self.adv_all.cm = st["adv_all"].to(dev)
        self.adv_rem.cm = st["adv_rem"].to(dev)
        self._flipped, self._remote = st["flipped"], st["remote"]
        self._hit, self._not_target = st["hit"], st["not_target"]
        self.records = list(st["records"])
        return self

    @property
    def done_images(self) -> set:
        return {r["image"] for r in self.records if "image" in r}

    def pooled(self, classes: str = "gt") -> Dict:
        """Dataset-level mIoU from one confusion matrix over every image, under BOTH class sets."""
        ca = miou_compare(self.clean_all, self.adv_all)
        cr = miou_compare(self.clean_rem, self.adv_rem)
        sfx = "" if classes == "gt" else "_union"
        out = {"clean_all": ca["clean" + sfx], "adv_all": ca["adv" + sfx],
               "drop_all": ca["drop" + sfx],
               "clean_remote": cr["clean" + sfx], "adv_remote": cr["adv" + sfx],
               "drop_remote": cr["drop" + sfx],
               "classes": classes,
               "drop_all_union": ca["drop_union"],
               "drop_remote_union": cr["drop_union"],
               "n_classes_gt": cr["n_classes_gt"],
               "n_classes_clean_union": cr["n_classes_clean_union"],
               "n_classes_adv_union": cr["n_classes_adv_union"],
               "class_set_moved": bool(cr["n_classes_clean_union"]
                                       != cr["n_classes_adv_union"]),
               "any_flip_rate": 100.0 * self._flipped / max(self._remote, 1),
               "remote_pixels": self._remote}
        if self.target_class is not None:
            out["target_hit_rate"] = (100.0 * self._hit
                                      / max(self._not_target, 1))
        return out

    def summarise(self, key: str = "drop_remote", seed: int = 0,
                  log=print) -> Dict:
        """The collective statistic."""
        vals = [r[key] for r in self.records if r.get(key) is not None]
        dist = describe(vals, seed=seed)
        classes = (self.records[0].get("classes", "gt")
                   if self.records else "gt")
        pool = self.pooled(classes)
        n = dist.get("n", 0)

        log(f"\n{'=' * 72}")
        log(f" POPULATION — {n} images, one patch optimised per image")
        log(f"{'=' * 72}")

        if n == 0:
            log("  no results")
            return {"n": 0, "pooled": pool}

        log(f"\n  PER-IMAGE {key} — the per-image threat model's headline")
        log(f"    mean          : {dist['mean']:+7.2f}  "
            f"+/- {dist['se']:.2f} (SE)")
        log(f"    95% CI (boot) : [{dist['ci95_boot'][0]:+.2f}, "
            f"{dist['ci95_boot'][1]:+.2f}]")
        log(f"    95% CI (t)    : [{dist['ci95_t'][0]:+.2f}, "
            f"{dist['ci95_t'][1]:+.2f}]"
            + ("   <- disagrees with the bootstrap; trust the bootstrap, the "
               "distribution is skewed"
               if abs(dist['ci95_t'][0] - dist['ci95_boot'][0]) > 0.5
               else ""))
        log(f"    std           : {dist['std']:7.2f}")
        log(f"    min q1 med q3 max : {dist['min']:+.2f}  {dist['q1']:+.2f}  "
            f"{dist['median']:+.2f}  {dist['q3']:+.2f}  {dist['max']:+.2f}")
        log(f"\n    images with a drop greater than:")
        for t in SUCCESS_THRESHOLDS:
            log(f"      {t:5.1f} pts : "
                f"{dist['success_rate'][str(t)]:5.1f}%")

        log(f"\n  POOLED DATASET mIoU — the benchmark-comparable number.")
        log(f"  NOT the mean of the per-image drops above, and not "
            f"interchangeable with it:")
        log(f"    clean  all / remote : {pool['clean_all']:6.2f} / "
            f"{pool['clean_remote']:6.2f}")
        log(f"    adv    all / remote : {pool['adv_all']:6.2f} / "
            f"{pool['adv_remote']:6.2f}")
        log(f"    drop   REMOTE       : {pool['drop_remote']:+6.2f}   "
            f"(classes={pool['classes']})")
        log(f"    drop   REMOTE union : {pool['drop_remote_union']:+6.2f}   "
            f"(pre-fix convention, for comparison only)")
        if pool["class_set_moved"]:
            log(f"    [!] class set moved : union counts "
                f"{pool['n_classes_clean_union']} clean vs "
                f"{pool['n_classes_adv_union']} adv over {pool['n_classes_gt']} "
                f"with GT —")
            log(f"        drop_remote_union={pool['drop_remote_union']:+.2f} is "
                f"an ARTEFACT; drop_remote={pool['drop_remote']:+.2f} is the "
                f"number.")
        log(f"    any_flip_rate       : {pool['any_flip_rate']:6.2f}%  "
            f"over {pool['remote_pixels']:,} remote px")
        if "target_hit_rate" in pool:
            log(f"    target_hit_rate     : {pool['target_hit_rate']:6.2f}%")

        need1 = images_needed(dist["std"], 1.0)
        need2 = images_needed(dist["std"], 2.0)
        log(f"\n  SAMPLE SIZE, judged against the spread actually observed:")
        log(f"    this run    : n = {n}, 95% CI half-width "
            f"{1.96 * dist['se']:.2f} pts")
        log(f"    for +/-2.0  : n = {need2}")
        log(f"    for +/-1.0  : n = {need1}")
        if n < need2:
            log(f"    -> UNDERPOWERED for a +/-2 point claim. Quote the "
                f"interval, not the mean,")
            log(f"       or raise --n_images to {need2}.")
        else:
            log(f"    -> adequate for a +/-2 point claim.")

        degraded = sum(1 for r in self.records
                       if r.get("degraded_after_peak"))
        if degraded:
            log(f"\n  WARNING: {degraded}/{n} runs degraded after their peak "
                f"(saturation collapse).")
            log(f"           Their FINAL patch is worse than their BEST. "
                f"Compare best_drop_remote,")
            log(f"           or lower --lr / check frac_at_clip.")

        return {"n": n, "key": key, "distribution": dist, "pooled": pool,
                "images_needed_pm1": need1, "images_needed_pm2": need2,
                "underpowered_pm2": bool(n < need2),
                "degraded_after_peak": degraded}


SELECT_MODES = ("best", "worst", "median", "spread")


def select(records: Sequence[Dict], k: int = 3, mode: str = "spread",
           key: str = "drop_remote") -> List[Dict]:
    """Pick the k records that get panels and the full diagnostic suite."""
    if mode not in SELECT_MODES:
        raise ValueError(f"mode must be one of {SELECT_MODES}, got {mode!r}")
    rs = [r for r in records if r.get(key) is not None]
    if not rs or k <= 0:
        return []
    ordered = sorted(rs, key=lambda r: r[key], reverse=True)
    n = len(ordered)
    k = min(k, n)

    if mode == "best":
        return ordered[:k]
    if mode == "worst":
        return ordered[-k:][::-1]
    if mode == "median":
        mid = n // 2
        lo = max(0, mid - k // 2)
        return ordered[lo:lo + k]

    n_best = k - 2 * (k // 3) if k % 3 else k // 3
    n_med, n_worst = k // 3, k // 3
    mid = n // 2
    picked, seen = [], set()
    for r in (ordered[:n_best]
              + ordered[max(0, mid - n_med // 2): max(0, mid - n_med // 2) + n_med]
              + (ordered[-n_worst:][::-1] if n_worst else [])):
        rid = id(r)
        if rid not in seen:
            seen.add(rid)
            picked.append(r)
    for r in ordered:
        if len(picked) >= k:
            break
        if id(r) not in seen:
            seen.add(id(r))
            picked.append(r)
    return picked[:k]


def paired_compare(a_records: Sequence[Dict], b_records: Sequence[Dict],
                   key: str = "drop_remote", label_a: str = "A",
                   label_b: str = "B", seed: int = 0, log=print) -> Dict:
    """Compare two conditions image by image, not mean against mean."""
    a_by = {r["image"]: r for r in a_records if "image" in r}
    b_by = {r["image"]: r for r in b_records if "image" in r}
    common = sorted(set(a_by) & set(b_by))
    dropped = (len(a_by) - len(common), len(b_by) - len(common))

    if not common:
        log(f"\n  no images in common between {label_a} and {label_b} — "
            f"were they run with the same --sample_seed?")
        return {"n": 0, "label_a": label_a, "label_b": label_b}

    av = [a_by[i][key] for i in common]
    bv = [b_by[i][key] for i in common]
    diffs = [b - a for a, b in zip(av, bv)]
    d = describe(diffs, seed=seed)

    better = sum(1 for x in diffs if x > 0)
    worse = sum(1 for x in diffs if x < 0)
    tie = len(diffs) - better - worse
    conclusive = not (d["ci95_boot"][0] <= 0.0 <= d["ci95_boot"][1])

    log(f"\n{'=' * 72}")
    log(f" PAIRED: {label_b}  vs  {label_a}      ({key})")
    log(f"{'=' * 72}")
    log(f"  {len(common)} images in common"
        + (f"   (dropped {dropped[0]} from {label_a}, {dropped[1]} from "
           f"{label_b} — the runs did not cover the same images)"
           if any(dropped) else ""))
    log(f"\n  {label_a:<22s}: {float(np.mean(av)):+7.2f}")
    log(f"  {label_b:<22s}: {float(np.mean(bv)):+7.2f}")
    log(f"  difference (paired)   : {d['mean']:+7.2f}  "
        f"95% CI [{d['ci95_boot'][0]:+.2f}, {d['ci95_boot'][1]:+.2f}]")
    log(f"  median difference     : {d['median']:+7.2f}")
    log(f"  {(label_b + ' better on'):<22s}: {better}/{len(common)} images "
        f"({100*better/len(common):.0f}%)   worse on {worse}, tied {tie}")

    if conclusive:
        direction = "STRONGER" if d["mean"] > 0 else "WEAKER"
        log(f"\n  -> {label_b} is {direction} than {label_a}. The 95% CI on the "
            f"paired difference excludes zero.")
    else:
        log(f"\n  -> NO RESOLVABLE DIFFERENCE. The 95% CI on the paired "
            f"difference spans zero.")
        log(f"     Quote the interval, not the point estimate. n = "
            f"{images_needed(d['std'], abs(d['mean']) or 1.0)} images would be "
            f"needed to resolve an effect this size.")

    return {"n": len(common), "key": key,
            "label_a": label_a, "label_b": label_b,
            "images": common, "dropped": list(dropped),
            "mean_a": float(np.mean(av)), "mean_b": float(np.mean(bv)),
            "difference": d, "n_better": better, "n_worse": worse,
            "n_tie": tie, "conclusive": conclusive}


def plot_paired(a_records: Sequence[Dict], b_records: Sequence[Dict], out_path,
                key: str = "drop_remote", label_a: str = "A",
                label_b: str = "B", title: str = "Paired comparison"):
    """Scatter of B against A, one point per image, with the identity line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_by = {r["image"]: r for r in a_records if "image" in r}
    b_by = {r["image"]: r for r in b_records if "image" in r}
    common = sorted(set(a_by) & set(b_by))
    if not common:
        return
    av = np.array([a_by[i][key] for i in common], dtype=np.float64)
    bv = np.array([b_by[i][key] for i in common], dtype=np.float64)
    diffs = bv - av
    d = describe(diffs)

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11, 4.6))

    lim = [min(av.min(), bv.min()), max(av.max(), bv.max())]
    pad = 0.05 * (lim[1] - lim[0] + 1e-9)
    lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, color="#888", lw=1, ls="--", label="no change")
    ax.scatter(av, bv, s=22, alpha=0.75,
               c=["#55a868" if x > 0 else "#c44e52" for x in diffs])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"{label_a}   ({key})")
    ax.set_ylabel(f"{label_b}   ({key})")
    ax.set_title(f"{title}\n{len(common)} images, paired")
    ax.legend(fontsize=8)

    axd.hist(diffs, bins=max(8, min(30, int(np.sqrt(diffs.size) * 2))),
             color="#4c72b0", alpha=0.85, edgecolor="white", linewidth=0.6)
    axd.axvline(0, color="#888", lw=1, ls="--")
    axd.axvspan(d["ci95_boot"][0], d["ci95_boot"][1], color="#c44e52",
                alpha=0.18, label="95% CI (bootstrap)")
    axd.axvline(d["mean"], color="#c44e52", lw=2,
                label=f"mean {d['mean']:+.2f}")
    axd.set_xlabel(f"{label_b} - {label_a}  (mIoU points)")
    axd.set_ylabel("images")
    axd.set_title("Paired difference\nCI excluding zero is the claim")
    axd.legend(fontsize=8)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(records: Sequence[Dict], out_path,
                      key: str = "drop_remote",
                      title: str = "Per-image attack strength",
                      highlight: Sequence[int] = ()):
    """Histogram + per-image strip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.asarray([r[key] for r in records if r.get(key) is not None],
                      dtype=np.float64)
    if vals.size == 0:
        return
    d = describe(vals)

    fig, (ax, axs) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    bins = max(8, min(30, int(np.sqrt(vals.size) * 2)))
    ax.hist(vals, bins=bins, color="#4c72b0", alpha=0.85,
            edgecolor="white", linewidth=0.6)
    ax.axvline(d["mean"], color="#c44e52", lw=2,
               label=f"mean {d['mean']:+.2f}")
    ax.axvspan(d["ci95_boot"][0], d["ci95_boot"][1], color="#c44e52",
               alpha=0.18, label="95% CI (bootstrap)")
    ax.axvline(d["median"], color="#55a868", lw=1.6, ls="--",
               label=f"median {d['median']:+.2f}")
    ax.set_ylabel("images")
    ax.set_title(f"{title}\nn = {d['n']} images, one patch optimised per image")
    ax.legend(fontsize=8)

    ids = [r.get("image") for r in records if r.get(key) is not None]
    hl = set(highlight)
    colours = ["#c44e52" if i in hl else "#4c72b0" for i in ids]
    axs.scatter(vals, np.random.default_rng(0).uniform(-1, 1, vals.size),
                c=colours, s=18, alpha=0.75)
    axs.set_yticks([])
    axs.set_ylim(-2.5, 2.5)
    axs.set_xlabel(f"{key} (mIoU points)")
    axs.grid(axis="x", alpha=0.25)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
