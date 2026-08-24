r"""
Diagnostics POOLED ACROSS A POPULATION — the suite report.py runs on one image,
accumulated over all of them.

WHY THIS EXISTS
---------------
overfit_population.py already reports a distribution of mIoU drops, and it runs
report.run() on --n_panels images. Those are the two ends of a range with a hole
in the middle:

  the distribution   one scalar per image. Says how MUCH the attack achieved,
                     says nothing about what it did.
  the panel suite    everything about what it did — on three images out of n.
                     Every mechanism claim in the thesis then rests on three
                     anecdotes, and every one of those claims is about the
                     ATTACK ("the untargeted loss is an implicit class
                     selector", "influence extends past effect") rather than
                     about those three images.

This module closes it. Everything report.run() computes from logits — the
confusion flows, the reach curve, the ring profiles, contestability — is pure
post-processing on tensors the population loop has already computed, so running
it on EVERY image costs a handful of [H,W] reductions per image against a
1000-step optimisation. Only the two genuinely expensive probes stay
panel-only: the Grad-CAM (a hook and a backward) and the ERF probe (n_probes
forward passes, and a MODEL property that does not vary by image anyway).

The panels then stop carrying the argument. They illustrate it, and the pooled
figures are what the claim is quoted from — which is also what makes
--panel_select best defensible: cherry-picking is only a problem when the
cherries ARE the evidence.

POOLED, NOT AVERAGED
--------------------
Every rate here is a POOLED ratio: numerator and denominator summed over images
and divided once. A mean of per-image rates weights a class present in 300
pixels of one image equally with the same class covering 200k pixels of
another, and for flip rates that gap is two orders of magnitude. The per-image
spread is reported ALONGSIDE the pooled value (the band on the reach curve, the
box plot on contestability) rather than instead of it, for the same reason
population.py reports a distribution next to the pooled mIoU: they answer
different questions and neither substitutes for the other.

DISTANCE BINS ARE ABSOLUTE PIXELS, deliberately. curves.untargeted_reach uses
n_bins spanning each image's own max distance, which is right for one image and
wrong for pooling: under --placement gradcam the footprint moves, the max
distance moves with it, and bin 7 would mean 420px in one image and 680px in
another. Fixed edges make bin k the same physical ring in every image, at the
cost of trailing bins that only off-centre placements populate.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data.cityscapes import PALETTE, class_name, upsample_to
from ..losses.reach import centroid, distance_map
from .untargeted import RINGS

# Fixed absolute rings, in pixels from the footprint centroid. 50px steps out to
# 1200: at 512x1024 a centred patch reaches 572px to the far corner and an
# edge-placed one ~1140, so this covers both without either being rebinned.
BIN_W = 50
N_BINS = 24
BIN_EDGES = [(k * BIN_W, (k + 1) * BIN_W) for k in range(N_BINS)]
BIN_CENTRES = np.array([(lo + hi) / 2 for lo, hi in BIN_EDGES])

# RINGS is imported rather than redefined so a pooled ring number and a panel
# ring number are the same measurement and can be quoted in one sentence.
CONTEST_MIN_DIST = 300.0
CONTEST_TOL = 3.0


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    _plt().close(fig)


def _iqr_band(rows: np.ndarray):
    """(median, q1, q3) down axis 0. All-nan columns give nan, not a warning."""
    if rows.size == 0:
        empty = np.array([])
        return empty, empty, empty
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (np.nanmedian(rows, axis=0),
                np.nanpercentile(rows, 25, axis=0),
                np.nanpercentile(rows, 75, axis=0))


# ═════════════════════════════════════════════════════════════════════════════
#  Accumulator
# ═════════════════════════════════════════════════════════════════════════════

class PopulationDiagnostics:
    r"""
    Fold one image's clean/adversarial logits into the pooled diagnostics.

    Mirrors metrics.population.Population: update() per image, state_dict() for
    resume, summarise() at the end. It is a SEPARATE object with a SEPARATE
    checkpoint file rather than fields on Population, because Population's
    state_dict is what every existing population run on disk was written with,
    and load_state_dict() would have to grow optional-key handling for runs that
    never had these tensors. A second file is resumable, backward compatible,
    and simply absent when the aggregate suite is off.

    Everything accumulates on CPU. The tensors are small (one K x K long matrix
    and a few length-24 vectors) and keeping them off-device means a resumed run
    reloads them without a device round-trip.
    """

    def __init__(self, num_classes: int = 19, n_bins: int = N_BINS):
        self.K = num_classes
        self.n_bins = n_bins

        # what flipped, and into what
        self.flip_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.pred_total = torch.zeros(num_classes, dtype=torch.long)

        # reach: pooled counts, plus one row per image for the spread band
        self.reach_changed = torch.zeros(n_bins, dtype=torch.long)
        self.reach_total = torch.zeros(n_bins, dtype=torch.long)
        self.per_image_reach: List[List[float]] = []

        # ring profiles: sums and counts, so the pooled mean is pixel-weighted
        nr = len(RINGS)
        self.margin_clean = torch.zeros(nr, dtype=torch.float64)
        self.margin_adv = torch.zeros(nr, dtype=torch.float64)
        self.ring_n = torch.zeros(nr, dtype=torch.float64)
        self.ent_clean = torch.zeros(nr, dtype=torch.float64)
        self.ent_adv = torch.zeros(nr, dtype=torch.float64)

        # a clean-image property, so it is per image and stays per image
        self.contest: List[List[float]] = []

        # convergence traces, already downsampled by the caller's --log_every
        self.curves: List[Dict] = []
        self.images: List[int] = []

    # ── per image ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, clean_logits, adv_logits, label, footprint,
               record: Optional[Dict] = None):
        """
        One image. `clean_logits`/`adv_logits` may be at any resolution; both
        are upsampled to the LABEL resolution, which is the resolution every
        other measurement in the repository is taken at.
        """
        K = self.K
        hw = label.shape[-2:]
        H, W = hw
        cl = upsample_to(clean_logits, hw)
        al = upsample_to(adv_logits, hw)
        pc, pa = cl.argmax(1)[0], al.argmax(1)[0]

        fp = footprint[0]
        remote = (label[0] != 255) & (~fp)
        changed = (pc != pa) & remote

        # ── flip flows, pooled ───────────────────────────────────────────────
        # The denominator is remote pixels of each CLEAN-PREDICTED class, not of
        # each ground-truth class: the question is what the attack MOVED, and
        # what it moved is a prediction.
        self.pred_total += torch.bincount(
            pc[remote].clamp(0, K - 1).flatten().cpu(), minlength=K).long()
        if bool(changed.any()):
            s = pc[changed].clamp(0, K - 1).flatten()
            d = pa[changed].clamp(0, K - 1).flatten()
            cm = torch.zeros(K * K, dtype=torch.long, device=s.device)
            cm.scatter_add_(0, s * K + d, torch.ones_like(s))
            self.flip_cm += cm.view(K, K).cpu()

        # ── reach, on fixed absolute rings ───────────────────────────────────
        dist = distance_map(H, W, *centroid(footprint), cl.device)
        outside = ~fp
        row: List[float] = []
        for k, (lo, hi) in enumerate(BIN_EDGES[:self.n_bins]):
            ring = outside & (dist >= lo) & (dist < hi)
            # `changed` is already restricted to labelled pixels, so the
            # denominator has to be too or an unlabelled ring would depress the
            # rate without ever being able to contribute to the numerator.
            n = int((remote & ring).sum())
            if n == 0:
                row.append(float("nan"))
                continue
            c = int((changed & ring).sum())
            self.reach_total[k] += n
            self.reach_changed[k] += c
            row.append(100.0 * c / n)
        self.per_image_reach.append(row)

        # ── margin and entropy by coarse ring ────────────────────────────────
        def margin(lg):
            t2 = lg[0, :K].topk(2, dim=0).values
            return t2[0] - t2[1]

        def ent(lg):
            p = F.softmax(lg[:, :K], 1)[0]
            return -(p * (p + 1e-10).log()).sum(0) / float(np.log(K))

        mc, ma = margin(cl), margin(al)
        ec, ea = ent(cl), ent(al)
        for j, (lo, hi) in enumerate(RINGS):
            ring = outside & (dist >= lo) & (dist < hi)
            n = int(ring.sum())
            if n == 0:
                continue
            self.ring_n[j] += n
            self.margin_clean[j] += float(mc[ring].sum())
            self.margin_adv[j] += float(ma[ring].sum())
            self.ent_clean[j] += float(ec[ring].sum())
            self.ent_adv[j] += float(ea[ring].sum())

        # ── far-field contestability, from the CLEAN logits ──────────────────
        far = outside & (dist >= CONTEST_MIN_DIST)
        if int(far.sum()) > 0:
            lg = cl[0, :K]
            top1 = lg.max(0).values
            self.contest.append(
                [float((lg[c][far] > top1[far] - CONTEST_TOL).float().mean()
                       * 100.0) for c in range(K)])
        else:
            self.contest.append([float("nan")] * K)

        # ── convergence trace ────────────────────────────────────────────────
        if record is not None:
            self.images.append(int(record.get("image", -1)))
            hist = record.get("history") or []
            base = record.get("clean_remote")
            if hist and base is not None:
                self.curves.append({
                    "image": int(record.get("image", -1)),
                    "step": [int(h["step"]) for h in hist],
                    # the DROP, not the absolute mIoU: images start from
                    # different clean scores, and a band over absolute mIoU
                    # would be dominated by that offset rather than by the
                    # attack's progress.
                    "drop": [float(base - h["miou_remote"]) for h in hist],
                    "visibility": [float(h["visibility"]) for h in hist
                                   if "visibility" in h]})

    # ── resume ───────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {"num_classes": self.K, "n_bins": self.n_bins,
                "flip_cm": self.flip_cm, "pred_total": self.pred_total,
                "reach_changed": self.reach_changed,
                "reach_total": self.reach_total,
                "per_image_reach": self.per_image_reach,
                "margin_clean": self.margin_clean,
                "margin_adv": self.margin_adv, "ring_n": self.ring_n,
                "ent_clean": self.ent_clean, "ent_adv": self.ent_adv,
                "contest": self.contest, "curves": self.curves,
                "images": self.images}

    def load_state_dict(self, st: Dict):
        if st.get("num_classes") != self.K or st.get("n_bins") != self.n_bins:
            raise ValueError(
                f"aggregate checkpoint has num_classes={st.get('num_classes')},"
                f" n_bins={st.get('n_bins')} but this run uses {self.K}, "
                f"{self.n_bins} — the pooled tensors are not compatible and "
                f"adding them together would be meaningless")
        for k in ("flip_cm", "pred_total", "reach_changed", "reach_total",
                  "margin_clean", "margin_adv", "ring_n", "ent_clean",
                  "ent_adv"):
            setattr(self, k, st[k])
        self.per_image_reach = list(st["per_image_reach"])
        self.contest = list(st["contest"])
        self.curves = list(st["curves"])
        self.images = list(st["images"])
        return self

    @property
    def n_images(self) -> int:
        return len(self.per_image_reach)

    @property
    def done_images(self) -> set:
        return {i for i in self.images if i >= 0}

    # ── reporting ────────────────────────────────────────────────────────────

    def flows(self, top: int = 12) -> List[Dict]:
        total = int(self.flip_cm.sum())
        if total == 0:
            return []
        pairs = [(int(self.flip_cm[s, d]), s, d)
                 for s in range(self.K) for d in range(self.K)
                 if s != d and self.flip_cm[s, d] > 0]
        pairs.sort(reverse=True)
        return [{"src": class_name(s), "dst": class_name(d), "px": n,
                 "pct_of_flips": 100.0 * n / total}
                for n, s, d in pairs[:top]]

    def flip_rates(self, min_px: int = 50_000) -> Dict[str, Dict]:
        """
        Pooled flip rate per clean-predicted class.

        min_px is a POOLED floor, so it is far larger than the per-image 50 that
        untargeted.confusion uses: at n=100 a class covering 500px per image
        accumulates 50k, and below that the pooled rate is decided by whichever
        handful of images happened to contain the class at all.
        """
        out = {}
        for c in range(self.K):
            tot = int(self.pred_total[c])
            if tot < min_px:
                continue
            fl = int(self.flip_cm[c].sum())
            out[class_name(c)] = {"rate": 100.0 * fl / tot,
                                  "flipped": fl, "total": tot}
        return out

    def summarise(self, tau: Optional[float] = None,
                  records: Sequence[Dict] = (), log=print) -> Dict:
        """The pooled report. Returns the dict that goes into aggregate.json."""
        n = self.n_images
        res: Dict = {"n_images": n, "distance_bin_width_px": BIN_W}

        log(f"\n{'=' * 72}")
        log(f" AGGREGATED DIAGNOSTICS — pooled over {n} images")
        log(f" Every rate below is a POOLED ratio, not a mean of per-image "
            f"rates.")
        log(f"{'=' * 72}")
        if n == 0:
            return res

        res.update(self._report_flows(log))
        res.update(self._report_reach(log))
        res.update(self._report_rings(log))
        res["contestability"] = self._report_contest(log)
        res["visibility"] = self._report_visibility(tau, records, log)
        res["convergence"] = self._report_convergence(log)
        return res

    # ── the individual sections ──────────────────────────────────────────────

    def _report_flows(self, log) -> Dict:
        total_flips = int(self.flip_cm.sum())
        total_remote = int(self.pred_total.sum())
        fl = self.flows()
        res = {"total_remote_px": total_remote,
               "total_flipped_px": total_flips,
               "pooled_flip_rate": 100.0 * total_flips / max(total_remote, 1),
               "flows": fl, "flip_rate_by_class": self.flip_rates()}

        log(f"\n  FLIP FLOWS — pooled over {total_remote:,} remote px, "
            f"{total_flips:,} flipped ({res['pooled_flip_rate']:.2f}%)")
        for f in fl[:8]:
            log(f"    {f['src']:12s} -> {f['dst']:12s} : {f['px']:12,} px  "
                f"({f['pct_of_flips']:5.1f}%)")
        if fl and fl[0]["pct_of_flips"] > 50:
            log(f"    -> STRUCTURED ACROSS THE POPULATION: one channel carries "
                f"{fl[0]['pct_of_flips']:.0f}% of every flip in the sample.")
            log(f"       The untargeted loss is an implicit class selector — "
                f"and this is that claim at")
            log(f"       n={self.n_images} rather than at n=1.")
        elif fl:
            log(f"    -> DIFFUSE: the top channel carries only "
                f"{fl[0]['pct_of_flips']:.0f}% of flips. Whatever a single "
                f"image showed,")
            log(f"       the population does not support a "
                f"single-channel-collapse reading.")
        return res

    def _report_reach(self, log) -> Dict:
        tot = self.reach_total.numpy().astype(np.float64)
        ch = self.reach_changed.numpy().astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled = np.where(tot > 0, 100.0 * ch / np.maximum(tot, 1), np.nan)
        med, q1, q3 = _iqr_band(np.asarray(self.per_image_reach,
                                           dtype=np.float64))
        keep = tot > 0
        rows = [{"centre_px": float(BIN_CENTRES[k]),
                 "pooled_pct": float(pooled[k]), "median_pct": float(med[k]),
                 "q1_pct": float(q1[k]), "q3_pct": float(q3[k]),
                 "px": int(tot[k])}
                for k in range(self.n_bins) if keep[k]]
        collapse = next((float(BIN_CENTRES[k]) for k in range(self.n_bins)
                         if keep[k] and pooled[k] < 5.0), None)

        log(f"\n  REACH — flip rate vs distance from the footprint centroid")
        log(f"      ring     pooled     median [q1, q3] across images")
        for k in range(self.n_bins):
            if not keep[k] or tot[k] < 1000:
                continue
            log(f"    {BIN_CENTRES[k]:5.0f}px  {pooled[k]:6.2f}%   "
                f"{med[k]:6.2f}% [{q1[k]:5.2f}, {q3[k]:5.2f}]")
        if collapse is not None:
            log(f"    -> the pooled flip rate falls below 5% at "
                f"~{collapse:.0f}px. That is the GEOMETRIC limit")
            log(f"       measured on the attack itself, not on the "
                f"random-noise ERF probe.")
        return {"reach": rows, "collapse_px": collapse}

    def _report_rings(self, log) -> Dict:
        nrn = self.ring_n.numpy()
        out = []
        log(f"\n  WINNER MARGIN and ENTROPY by ring "
            f"(pixel-weighted over all images)")
        log(f"      ring(px)       margin clean -> adv          entropy "
            f"clean -> adv")
        for j, (lo, hi) in enumerate(RINGS):
            if nrn[j] == 0:
                continue
            mc = float(self.margin_clean[j]) / nrn[j]
            ma = float(self.margin_adv[j]) / nrn[j]
            ec = float(self.ent_clean[j]) / nrn[j]
            ea = float(self.ent_adv[j]) / nrn[j]
            out.append({"lo": lo, "hi": hi, "px": int(nrn[j]),
                        "margin_clean": mc, "margin_adv": ma,
                        "entropy_clean": ec, "entropy_adv": ea})
            log(f"    {lo:5d}-{hi:<5d}    {mc:6.2f} -> {ma:6.2f} "
                f"({ma - mc:+6.2f})     {ec:6.3f} -> {ea:6.3f} "
                f"({ea - ec:+6.3f})")
        log(f"    Margin eroding in rings where the flip rate has already "
            f"collapsed IS the")
        log(f"    influence-outruns-effect claim, now measured on the whole "
            f"sample.")
        return {"rings": out}

    def _report_contest(self, log) -> Dict:
        cts = np.asarray(self.contest, dtype=np.float64)
        if cts.size == 0:
            return {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(cts, axis=0)
            q1 = np.nanpercentile(cts, 25, axis=0)
            q3 = np.nanpercentile(cts, 75, axis=0)

        log(f"\n  FAR-FIELD CONTESTABILITY (clean logits, dist >= "
            f"{CONTEST_MIN_DIST:.0f}px, within {CONTEST_TOL} logits of top-1)")
        log(f"      the SEMANTIC factor — a property of the SCENE, not of the "
            f"attack")
        out = {}
        for c in np.argsort(-np.nan_to_num(med)):
            c = int(c)
            # nan means NO image had a far field at all — at this resolution or
            # this placement every remote pixel sits inside CONTEST_MIN_DIST.
            # That is a missing measurement, not a contestability of zero.
            if np.isnan(med[c]) or (med[c] < 0.05 and q3[c] < 0.05):
                continue
            out[class_name(c)] = {"median": float(med[c]), "q1": float(q1[c]),
                                  "q3": float(q3[c])}
            log(f"    {c:2d} {class_name(c):12s}: median {med[c]:5.1f}%  "
                f"[q1 {q1[c]:5.1f}, q3 {q3[c]:5.1f}]")
        if not out:
            log(f"    no image has a far field beyond "
                f"{CONTEST_MIN_DIST:.0f}px — the semantic factor is not "
                f"measurable at this resolution.")
        if out:
            spread = {k: v["q3"] - v["q1"] for k, v in out.items()}
            worst = max(spread, key=spread.get)
            log(f"    -> {worst} spans {spread[worst]:.1f} points between q1 "
                f"and q3 across images.")
            log(f"       Report it as a distribution; the single-image number "
                f"is not a model property.")
        return out

    def _report_visibility(self, tau, records, log) -> Dict:
        r"""
        Realised visibility against the budget it was optimised under.

        tau is an INTENT — the constraint the renderer enforces, under a mu=0.5
        contrast convention. The realised index is the OUTCOME, measured on the
        rendered patch against the content it actually landed on, and the two
        are not the same number: local contrast on Cityscapes windows runs well
        below 0.5, so the locally-measured index is systematically the larger of
        the two. A CSF result quoted at its nominal tau without this ratio
        beside it is quoting the intent and calling it the measurement.
        """
        vals = {k: [r[k] for r in records if r.get(k) is not None]
                for k in ("final_visibility", "final_visibility_local",
                          "final_resid_rms", "final_resid_absmax")}
        vals = {k: v for k, v in vals.items() if v}
        if not vals:
            return {}

        out: Dict = {"tau": tau}
        log(f"\n  REALISED VISIBILITY — the outcome, against tau = "
            f"{tau if tau is not None else '?'} (the intent)")
        for k, v in vals.items():
            a = np.asarray(v, dtype=np.float64)
            out[k] = {"mean": float(a.mean()),
                      "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                      "median": float(np.median(a)),
                      "q1": float(np.percentile(a, 25)),
                      "q3": float(np.percentile(a, 75)),
                      "min": float(a.min()), "max": float(a.max()),
                      "n": int(a.size)}
            log(f"    {k:24s}: mean {a.mean():7.4f}   median "
                f"{np.median(a):7.4f}   [{a.min():.4f}, {a.max():.4f}]")
        if tau:
            for k in ("final_visibility", "final_visibility_local"):
                if k not in out:
                    continue
                out[k]["median_over_tau"] = float(out[k]["median"] / tau)
                out[k]["frac_over_tau"] = float(
                    np.mean(np.asarray(vals[k]) > tau) * 100.0)
                log(f"    {k:24s}: median / tau = "
                    f"{out[k]['median_over_tau']:5.2f}x   "
                    f"{out[k]['frac_over_tau']:5.1f}% of images exceed tau")
            log(f"    -> Quote the MEASURED index. A ratio above 1 is the "
                f"budget being spent past its")
            log(f"       own bound on real content, and here it is a "
                f"population number rather than")
            log(f"       a one-image anecdote.")
        return out

    def _report_convergence(self, log) -> Dict:
        """Did the run converge, or did the step counter simply run out?"""
        if not self.curves:
            return {}
        steps = self.curves[0]["step"]
        rows = np.full((len(self.curves), len(steps)), np.nan)
        for i, c in enumerate(self.curves):
            m = min(len(steps), len(c["drop"]))
            rows[i, :m] = c["drop"][:m]
        med, q1, q3 = _iqr_band(rows)
        out = {"step": [int(s) for s in steps],
               "median_drop": [float(x) for x in med],
               "q1_drop": [float(x) for x in q1],
               "q3_drop": [float(x) for x in q3],
               "n_curves": len(self.curves)}
        if len(steps) < 5:
            return out

        # THE RUN-LENGTH TEST. If the last fifth of the run still moves the
        # median by more than a point, the reported number is where the step
        # counter stopped rather than where the attack converged.
        k = max(1, len(steps) // 5)
        tail = float(med[-1] - med[-1 - k])
        out["tail_gain"] = tail
        out["tail_steps"] = int(steps[-1] - steps[-1 - k])
        out["converged"] = bool(abs(tail) < 1.0)

        log(f"\n  CONVERGENCE — median drop over {len(self.curves)} images, "
            f"{steps[-1]} steps")
        log(f"    median drop at step {steps[-1]:4d} : {med[-1]:+6.2f}  "
            f"[q1 {q1[-1]:+.2f}, q3 {q3[-1]:+.2f}]")
        log(f"    gain over the last {out['tail_steps']} steps : "
            f"{tail:+.2f} pts")
        if out["converged"]:
            log(f"    -> CONVERGED. The run length is not what set the number.")
        else:
            log(f"    -> NOT CONVERGED: the attack was still improving when "
                f"the run ended, so the")
            log(f"       reported drop is a LOWER BOUND and --steps needs "
                f"raising before it is quoted.")
        return out

    # ── figures ──────────────────────────────────────────────────────────────

    def write_figures(self, out_dir, tau: Optional[float] = None,
                      records: Sequence[Dict] = (), title: str = ""):
        """Every pooled figure, each standalone with its own title and legend."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, fn in (("flip_flows.png", self._fig_flows),
                         ("flip_rate_by_class.png", self._fig_flip_rates),
                         ("reach.png", self._fig_reach),
                         ("rings.png", self._fig_rings),
                         ("contestability.png", self._fig_contest),
                         ("convergence.png", self._fig_convergence)):
            if fn(out_dir / name, title):
                written.append(name)
        if self._fig_visibility(out_dir / "visibility.png", tau, records,
                                title):
            written.append("visibility.png")
        return written

    def _fig_flows(self, path, title):
        fl = self.flows(12)
        if not fl:
            return False
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(fl))))
        y = np.arange(len(fl))[::-1]
        ax.barh(y, [f["pct_of_flips"] for f in fl], color="#4c72b0", alpha=0.9)
        for k, f in enumerate(fl):
            ax.text(f["pct_of_flips"] + 0.4, y[k], f"{f['px']:,} px",
                    va="center", fontsize=7, color="#444")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{f['src']} -> {f['dst']}" for f in fl],
                           fontsize=8)
        ax.set_xlabel("% of all remote flips in the population")
        ax.set_title(f"Pooled prediction flows — {self.n_images} images"
                     + (f"\n{title}" if title else ""))
        ax.grid(axis="x", alpha=0.25)
        _save(fig, path)
        return True

    def _fig_flip_rates(self, path, title):
        rates = self.flip_rates()
        if not rates:
            return False
        plt = _plt()
        idx = [c for c in range(self.K) if class_name(c) in rates]
        vals = [rates[class_name(c)]["rate"] for c in idx]
        fig, ax = plt.subplots(figsize=(max(8, len(idx) * 0.7), 4.2))
        ax.bar(np.arange(len(idx)), vals,
               color=[PALETTE[c].tolist() for c in idx], edgecolor="#333",
               linewidth=0.5)
        for i, c in enumerate(idx):
            ax.text(i, vals[i] + 1.5,
                    f"{rates[class_name(c)]['total'] / 1e6:.1f}M",
                    ha="center", fontsize=6.5, color="#555")
        ax.set_xticks(np.arange(len(idx)))
        ax.set_xticklabels([class_name(c) for c in idx], rotation=40,
                           ha="right")
        ax.set_ylabel("% of that class's remote px that flipped")
        ax.set_ylim(0, 105)
        ax.set_title("Pooled flip rate by clean-predicted class"
                     + (f" — {title}" if title else "")
                     + "\nbar label = pooled pixels in that class's "
                       "denominator")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, path)
        return True

    def _fig_reach(self, path, title):
        tot = self.reach_total.numpy().astype(np.float64)
        if tot.sum() == 0:
            return False
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled = np.where(tot > 0,
                              100.0 * self.reach_changed.numpy()
                              / np.maximum(tot, 1), np.nan)
        rows = np.asarray(self.per_image_reach, dtype=np.float64)
        med, q1, q3 = _iqr_band(rows)
        keep = tot > 0
        # BIN_CENTRES is the full ladder; this accumulator may hold a prefix of
        # it, and indexing the full array with a short mask is an IndexError
        # rather than a wrong plot only by luck.
        x = BIN_CENTRES[:self.n_bins][keep]

        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for r in rows:
            ax.plot(x, np.asarray(r)[keep], color="#999", lw=0.5, alpha=0.18)
        ax.fill_between(x, q1[keep], q3[keep], color="#4c72b0", alpha=0.22,
                        label="per-image IQR")
        ax.plot(x, med[keep], color="#4c72b0", lw=1.6, ls="--",
                label="per-image median")
        ax.plot(x, pooled[keep], color="#c44e52", lw=2.4,
                label="pooled (pixel-weighted)")
        ax.axhline(5.0, color="#555", lw=0.9, ls=":", label="5% collapse line")
        ax.set_xlabel("distance from footprint centroid (px)")
        ax.set_ylabel("% of ring pixels whose argmax changed")
        ax.set_title(f"Attack reach, pooled over {self.n_images} images"
                     + (f"\n{title}" if title else ""))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        _save(fig, path)
        return True

    def _fig_rings(self, path, title):
        nrn = self.ring_n.numpy()
        keep = [j for j in range(len(RINGS)) if nrn[j] > 0]
        if not keep:
            return False
        plt = _plt()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(len(keep))
        lbl = [f"{RINGS[j][0]}-{RINGS[j][1]}" for j in keep]
        panels = [(self.margin_clean, self.margin_adv,
                   "winner margin (logits)"),
                  (self.ent_clean, self.ent_adv, "normalised entropy")]
        for ax, (cs, advs, name) in zip(axes, panels):
            c = np.array([float(cs[j]) / nrn[j] for j in keep])
            a = np.array([float(advs[j]) / nrn[j] for j in keep])
            ax.bar(x - 0.19, c, 0.38, label="clean", color="#4c72b0",
                   alpha=0.9)
            ax.bar(x + 0.19, a, 0.38, label="patched", color="#dd8452",
                   alpha=0.9)
            for i in range(len(keep)):
                ax.annotate(f"{a[i] - c[i]:+.2f}", (x[i], max(c[i], a[i])),
                            ha="center", va="bottom", fontsize=7,
                            color="#c44e52")
            ax.set_xticks(x)
            ax.set_xticklabels(lbl, rotation=20, ha="right", fontsize=8)
            ax.set_xlabel("ring (px from footprint centroid)")
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Confidence by ring, pixel-weighted over "
                     f"{self.n_images} images"
                     + (f" — {title}" if title else ""))
        plt.tight_layout()
        _save(fig, path)
        return True

    def _fig_contest(self, path, title):
        cts = np.asarray(self.contest, dtype=np.float64)
        if cts.size == 0:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(cts, axis=0)
        keep = [int(c) for c in np.argsort(-np.nan_to_num(med))[:12]
                if not np.isnan(med[c]) and med[c] > 0.05]
        if not keep:
            return False
        data = [cts[~np.isnan(cts[:, c]), c] for c in keep]
        plt = _plt()
        fig, ax = plt.subplots(figsize=(max(8, len(keep) * 0.75), 4.4))
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops={"color": "#222", "lw": 1.4},
                        flierprops={"marker": ".", "ms": 3, "alpha": 0.5})
        for patch, c in zip(bp["boxes"], keep):
            patch.set_facecolor(PALETTE[c].tolist())
            patch.set_alpha(0.85)
            patch.set_edgecolor("#333")
        ax.set_xticks(np.arange(1, len(keep) + 1))
        ax.set_xticklabels([class_name(c) for c in keep], rotation=40,
                           ha="right")
        ax.set_ylabel(f"% of far-field px within {CONTEST_TOL} logits of top-1")
        ax.set_title(f"Far-field contestability across {self.n_images} images "
                     f"(dist >= {CONTEST_MIN_DIST:.0f}px)"
                     + (f" — {title}" if title else "")
                     + "\nthe SEMANTIC factor, measured on the CLEAN logits")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, path)
        return True

    def _fig_convergence(self, path, title):
        if not self.curves:
            return False
        steps = np.asarray(self.curves[0]["step"], dtype=np.float64)
        rows = np.full((len(self.curves), len(steps)), np.nan)
        for i, c in enumerate(self.curves):
            m = min(len(steps), len(c["drop"]))
            rows[i, :m] = c["drop"][:m]
        med, q1, q3 = _iqr_band(rows)

        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for r in rows:
            ax.plot(steps, r, color="#999", lw=0.5, alpha=0.18)
        ax.fill_between(steps, q1, q3, color="#4c72b0", alpha=0.22,
                        label="IQR across images")
        ax.plot(steps, med, color="#c44e52", lw=2.2, label="median")
        ax.set_xlabel("optimiser step")
        ax.set_ylabel("remote mIoU drop (pts)")
        ax.set_title(f"Convergence over {len(self.curves)} images"
                     + (f" — {title}" if title else "")
                     + "\na flat tail is the evidence that --steps did not set "
                       "the number")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        _save(fig, path)
        return True

    def _fig_visibility(self, path, tau, records, title):
        keys = [k for k in ("final_visibility", "final_visibility_local")
                if any(r.get(k) is not None for r in records)]
        if not keys:
            return False
        colours = {"final_visibility": "#4c72b0",
                   "final_visibility_local": "#dd8452"}
        labels = {"final_visibility": "realised index (mu=0.5 convention)",
                  "final_visibility_local": "realised index (local contrast)"}
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for k in keys:
            v = np.asarray([r[k] for r in records if r.get(k) is not None],
                           dtype=np.float64)
            ax.hist(v, bins=max(8, min(30, int(np.sqrt(v.size) * 2))),
                    alpha=0.6, color=colours[k], edgecolor="white",
                    linewidth=0.5,
                    label=f"{labels[k]} — median {np.median(v):.3f}")
        if tau:
            ax.axvline(tau, color="#c44e52", lw=2.2,
                       label=f"tau = {tau:g}  (the INTENT)")
        ax.set_xlabel("realised visibility index at the end of the run")
        ax.set_ylabel("images")
        ax.set_title("Realised visibility vs the budget it was optimised under"
                     + (f" — {title}" if title else "")
                     + "\nmass right of tau = budget spent past its own bound "
                       "on real content")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, path)
        return True


# ═════════════════════════════════════════════════════════════════════════════
#  Pooled per-class IoU — the one aggregate that comes from Population, not here
# ═════════════════════════════════════════════════════════════════════════════

def pooled_per_class_iou(clean_metric, adv_metric, out_path,
                         target_class: Optional[int] = None,
                         title: Optional[str] = None, log=print) -> Dict:
    r"""
    Per-class IoU from the POOLED confusion matrices, clean vs patched.

    The population run already accumulates these four matrices for its pooled
    mIoU; this is the per-class breakdown of the same object, and it costs one
    division. It is the aggregate answer to the question a single-image
    per_class_iou.png raises and cannot settle: aggregate mIoU hides WHICH
    classes the attack destroys, and one image cannot tell a class the attack
    reliably destroys from a class that happened to be fragile in that scene.

    `classes='gt'` throughout — a class with no ground truth anywhere in the
    pool is nan in both passes and is dropped from the figure rather than
    scored 0, which is the artefact miou.py's header exists to prevent.
    """
    from .report import _iou_figure  # local: report imports diagnostics too

    ciou, aiou = clean_metric.per_class("gt"), adv_metric.per_class("gt")
    present = [c for c in range(ciou.numel())
               if not bool(torch.isnan(ciou[c])) and not bool(torch.isnan(aiou[c]))]
    if not present:
        return {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _iou_figure(ciou, aiou, present, out_path, target_class, title=title)

    out = {class_name(c): {"clean": float(ciou[c]), "adv": float(aiou[c]),
                           "drop": float(ciou[c] - aiou[c])} for c in present}
    log(f"\n  POOLED PER-CLASS IoU (remote pixels, one confusion matrix over "
        f"every image)")
    for c in present:
        log(f"    {c:2d} {class_name(c):12s}: {ciou[c]:6.2f} -> {aiou[c]:6.2f} "
            f" ({aiou[c] - ciou[c]:+6.2f})")
    worst = max(out, key=lambda k: out[k]["drop"])
    log(f"    -> largest pooled loss: {worst} loses {out[worst]['drop']:.2f} "
        f"IoU points. ONE class losing tens of")
    log(f"       points while the rest move by a few is structured collapse, "
        f"not broad degradation.")
    return out
