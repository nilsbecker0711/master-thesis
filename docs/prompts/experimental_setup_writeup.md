# Prompt — write the Experimental Setup section

Paste everything below the rule into a fresh session, from the repository root.
It is self-contained: a cold session needs no other context.

Keep this file in sync when the setup changes. A prompt that describes a setup
the code no longer has produces a thesis section that is wrong in a way nobody
catches until the defence.

---

You are helping write the **Experimental Setup** section of a master's thesis in
computer vision. Work from the repository you are in; do not invent
configuration, numbers, or citations.

## Your task

Produce a LaTeX section (`\section{Experimental Setup}`) that a reader could use
to reproduce every result in the thesis. It describes **what was run and how it
was measured** — not what came out. Results, discussion and interpretation
belong to later sections and must not appear here beyond what is needed to
justify a methodological choice.

Target length 4–6 pages. Write it into `docs/experimental_setup.tex` as a
standalone `\section` that compiles inside the existing thesis preamble
(`booktabs`, `amsmath`, `natbib` are available).

## Read these first — they are ground truth, this prompt is a summary

| file | what it settles |
|---|---|
| `docs/csf_patch_procedure.tex` | the formal procedure, every equation, and the provenance table. **The written section must not contradict it.** |
| `patchreach/patch/csf.py` | the constraint as implemented: budget, reparameterisation, visibility index, range fit, calibrated path |
| `patchreach/patch/optimise.py` | the attack loop and the exact result record |
| `scripts/_common.py` | every CLI default quoted below |
| `patchreach/models/registry.py` | the architecture lineup and what each entry actually is |
| `patchreach/metrics/miou.py` | `drop_remote`, `any_flip_rate`, the shared-denominator rule |
| `scripts/sweep_operating_point.py` | the staged sweep and its decision rules |

If this prompt and the code disagree, **the code wins** — and say so in your
reply so the prompt gets fixed.

---

## The setup

### Threat model

A localised adversarial patch, optimised **per image**, against a **frozen**
semantic-segmentation network. White box: gradients flow through the network,
`∇_φ f` is never formed. Untargeted — the objective is degradation of the
prediction outside the patch, not a chosen class.

The patch replaces image content and the **base is the content it replaces**, so
`τ → 0` recovers the unperturbed image exactly and the no-attack baseline is a
no-op by construction rather than a weak attack.

### Data

- **Cityscapes**, 19 train IDs, label `255` is void and is excluded everywhere.
- Input resolution **512 × 1024** (`--img_h 512 --img_w 1024`). Note this is the
  dataloader size, not the mmseg config's training crop; one config per
  checkpoint covers every resolution.
- Single-image experiments: `val` image 420 unless stated.
- Population experiments: `val`, `--images random --n_images 500 --sample_seed 68`.
- Calibration measurements: the full `train` split (n = 2975).

### Architectures

Registry entries with Cityscapes checkpoints: `segformer_b0` (MiT-B0 encoder,
all-MLP decoder), `segformer_b5`, `deeplabv3plus_r101` (dilated ResNet-101,
ASPP), `unet_s5d16` (the only entry with an entry stride of 1),
`setr_pup` (ViT-L, 16×16 patch embed, entry stride 16), `internimage_t`
(DCNv3, UPerNet head, emits a 150-channel ADE20K-dimensioned head of which
0..18 are active).

The lineup is being extended into an **encoder × decoder factorial**, because
every current entry varies encoder and decoder together and so cannot identify
which of the two causes the observed CNN robustness. Describe the factorial as
the design; mark cells without Cityscapes checkpoints as label-free-probe only.

### The patch and the constraint

Patch scale `s = 0.25` of image height → rendered side `p = 128 px`; parameter
resolution `S = 128`. Placement `center` by default; `gradcam` and `semantic`
are the alternatives, and placement is treated as an experimental axis rather
than a fixed choice.

The perceptual constraint is a **contrast-sensitivity budget on the DFT of the
residual**:

- **Observer model**: Barten CSF, parameters from CSFlow App. A. Alternative
  `--csf_model sso` (Standard Spatial Observer, with the oblique effect) exists
  as a control with no luminance parameter.
- **Viewing geometry**: pixel size `0.0114 cm`, viewing distance `50 cm`,
  display peak `100 cd/m²`. Every visibility statement is conditional on these,
  and they are swept (not fixed) — see T18.
- **Pooling**: Minkowski with `β = 3.0` (`--csf_beta`), the probability-summation
  exponent. The `β → ∞` max criterion is more permissive and must not be used.
- **Low-frequency cutoff**: `m_c = 2` cycles across the patch (`--csf_min_cycles`).
  Without it the near-DC bins receive the largest allowance of any frequency.
- **Contrast conversion**: `κ = 4`, encoding Michelson contrast `A/μ` at an
  assumed `μ = 0.5`. `--csf_lref 0` (the default) keeps this legacy convention.

Two parameterisations, and **they are not interchangeable**:

- `--csf_param pgd` (default) — the parameter *is* the residual, every rfft bin
  is clamped to its budget after each step. Enforces `v ≤ τ`.
- `--csf_param squash` — the smooth reparameterisation `B(f)·Z/√(1+|Z|²)` plus a
  rescale to exactly τ. Enforces `v = τ`.
- Their learning rates differ by roughly the ratio of their parameter RMS
  (≈ 0.99 vs ≈ 0.027 measured at τ 0.25, size 128), so
  `INCUMBENT_LR = {"squash": 0.2, "pgd": 0.01}` in
  `scripts/sweep_operating_point.py`. **Never carry an lr across**: 0.2 under
  pgd is a 7.4× step that overshoots the constraint set every iteration, and
  the run does not fail — it converges somewhere else.
- **Every csf number on record was measured under `squash`** (τ 0.25, lr 0.2,
  1000 steps, drop 50.53 ± 1.54), while `pgd` is the current default. Any
  quoted number must name the parameterisation it was measured under.

Two enforcement modes:

- `--csf_enforce nominal` (default) — τ bounds the residual we *intend* to add.
- `--csf_enforce realised` — τ bounds the residual that *survives compositing*.
  **Any number quoted with a τ attached must come from a `realised` run**;
  clipping against real content was measured inflating nominal τ to ≈ 2.8× at
  1000 steps.

### Objective and optimisation

**CosPGD**, with two declared deviations from the source: the cosine weight is
**detached** (no gradient through the weight), and the attack uses **Adam on the
raw gradient** rather than sign-SGD with ε-projection, because a patch is not an
ε-ball perturbation. Say this explicitly — the thesis measures *our* CosPGD.

The scored support Ω excludes void labels **and the patch footprint**, the latter
to prevent the degenerate solution in which the patch adopts the attacked
appearance instead of influencing its surroundings.

- Initialisation: `z ~ N(0, I)`. Gaussian rather than zero because the visibility
  rescale is undefined at `δ₀ = 0`; the construction is scale-invariant in `z`,
  so only the *shape* of the initialisation matters.
- Schedule: `--lr_schedule cosine`, annealing to zero over the run. Because the
  scheduler is built with `T_max = steps`, **a run length cannot be read off a
  longer run's history** — the ladder needs separate runs.
- Steps: 1000 for `csf` mode.
- **The seed does not buy reproducibility.** The bilinear upsample in the loss
  path has a non-deterministic CUDA backward, so two seed-42 runs of one config
  differ from step 1. `--seeds N` samples both sources of spread together.
  Always quote mean ± sample sd (n−1), never one run.

### What is measured

| quantity | definition |
|---|---|
| `drop_remote` | clean − adversarial mIoU on Ω. The headline. |
| | Mean over classes **present in the ground truth restricted to Ω**, so both terms share a denominator. Counting a class because it is *predicted* makes the two means run over different class sets, shifting Δ by ≈ mIoU/(n−1) with no pixel-level change. |
| `any_flip_rate` | % of remote pixels whose argmax changed at all |
| `final_visibility` | realised Minkowski visibility, μ = 0.5 convention |
| `final_visibility_local` | the same against the locally measured mean — roughly 2.4× the nominal one, and the honest number |
| `final_resid_rms`, `final_resid_absmax` | residual magnitude in pixel units |
| `final_frac_at_bound`, `final_spend_mean` | spectral occupancy; → 1 means the allocation is frozen and the optimiser can only rotate phase |
| `wall_clock_s` | per-image attack cost |
| RAPSD, LPIPS | evaluation only, never in the objective |

Every run writes `results.json` (or `summary.json` + `seed*/`), `config.json` and
a full `run.log` into its own directory; `analysis/build_index.py` flattens a
directory of runs into one CSV.

---

## The measurement plan — 23 tables in four blocks

Describe this as the *design*, not as results. Tag convention: every run carries
`--tag T##_<slug>`; `sweep_operating_point.py` uses `--name`; the three probes
(`measure_erf.py`, `measure_frequency_sensitivity.py`,
`measure_footprint_luminance.py`) have no `--tag` and take
`--out_dir results/tables/T##_<slug>`.

**Block A — premise.** T01 spectral efficiency gap (η = Δflip/ΔV per band);
T02 τ nominal vs calibrated; T03 ERF reach × architecture; T04 band profile ×
architecture. T03/T04 are label- and dataset-independent, so they run on ADE20K
weights and cover the full encoder × decoder grid without training.

**Block B — operating point, one image, one model.** T05 run length; T06
learning rate; T07 the τ curve and its usable range; T08 cost of enforcement;
T09 τ × lr interaction; T10 `pgd` vs `squash`; T11 seed spread; T12 controls.
T05–T09 come from a single staged sweep.

**Block C — generalisation.** T13 cross-image distribution at n = 500 with
bootstrap CI; T14 scene predictors of the drop; T15 placement, paired; T16
cross-model at each architecture's own learning rate; T17 transfer matrix
(diagonal = image transfer, off-diagonal = model transfer).

**Block D — scope and cost.** T18 viewing geometry; T19 patch scale; T20 LPIPS
and residual RAPSD; T21 per-class IoU; T22 sink classes; T23 compute cost.

### Why the sweep is staged, and why the order is what it is

τ and lr are circular under `nominal` enforcement: the lr is selected at equal
perceptual cost, so it depends on τ, while realised visibility at a given τ
depends on lr through clip harmonics. `realised` breaks the circle by holding
visibility at exactly τ at every step. State this — it is a methodological
choice, not an implementation detail.

The sweep is a star around an incumbent (one factor at a time) plus a 3 × 3
τ × lr block that checks the star was allowed. Selection rules that are **not**
argmax, and which the section must state as such:

- **Run length**: the *shortest* run within 1.0 mIoU of the longest tested.
  Argmax returns the end of the ladder every time and measures the ladder.
- **Learning rate**: the best mean drop **among runs at or below a visibility
  ceiling**. Plain argmax rewards whichever lr breaks the perceptual constraint
  hardest and calls it better optimisation.
- **τ**: nothing is selected. τ is the axis the family is reported against; the
  stage reports the *usable range*, i.e. the τ above which the dynamic-range fit
  binds and τ ceases to control anything. That knee is reference-dependent and
  is measured per sweep rather than assumed.

---

## Measurements already on record

Give these as preliminary/motivating where the section needs them to justify a
choice. Do **not** present them as results of the thesis.

- **Run length.** 400 steps: `drop_remote 35.60 ± 4.78`, looked bimodal — it was
  truncation, not two basins. 1000 steps: `50.53 ± 1.54`, single mode. With
  `--csf_enforce realised`: `45.95 ± 3.57`, i.e. enforcement costs ≈ 4.6 mIoU and
  buys a τ that is actually held. Quote the enforced figure.
- **τ calibration** (full train split, 128 px centre footprint). Legacy/calibrated
  budget ratio: **2.98× low, 1.97× mid, 1.88× near-Nyquist**. Decomposes as
  `μ = 0.5` alone 3.35× too permissive against `E = 500 Td` pulling back 0.56×.
  Median footprint sRGB code 0.303, linear Y 0.0971. Correct sRGB-space budget:
  `B(f) = τ (c_ref + 0.055) / (4.8 · CSF(f; Y_ref))`.
- **Reference luminance.** Making the budget's reference luminance per-image buys
  only ≈ 13 % of budget at Nyquist (p95/p5 ratio 1.13×); the `sso` control is flat
  at 1.930× across all frequencies. Gamma encoding already performs most of the
  luminance adaptation. **Scope: this measures the L_ref term only** — it says
  nothing about per-image adaptivity of the base, residual, range fit or
  placement, all of which remain justified.
- **Spectral gap.** At fixed rms 0.02 every radial band moves the network about
  equally (flip 0.36–0.53 %) while perceptual cost differs enormously (70.7 JND
  at DC vs 1.5 JND near Nyquist) — an efficiency ratio of ≈ 47×, not a higher
  flip rate. SegFormer is roughly spectrally flat in what it reads.

## What the section must state

1. **τ is nominal, not a psychophysical JND.** Three approximations stand between
   them: the mean luminance in κ is assumed rather than measured; the Minkowski
   sum runs over DFT bins rather than perceptual channels; and the viewing
   geometry is an assumption that CSFlow itself calls a conceptual limitation.
   Report τ as a relative axis.
2. **No imperceptibility claim.** A claim of imperceptibility requires a study
   with human observers, which this does not constitute.
3. **Both τ columns.** τ nominal and τ calibrated everywhere, so past runs are
   restatable by a constant rather than invalidated.
4. **Never a drop without its perceptual cost.** Any table reporting a drop
   reports realised visibility beside it.
5. **The two CosPGD deviations**, explicitly.
6. **Bounded by CSF ≠ statistically inconspicuous.** Natural image spectra follow
   an approximate power law, so energy placed near Nyquist is atypical of the
   data precisely where the eye is least sensitive. Invisibility to an observer
   and indistinguishability from natural image statistics are distinct
   properties; only the former is constrained.
7. **Novelty framing.** The provenance table in `csf_patch_procedure.tex` claims
   five elements have no external source. Two are prior art: inverting a CSF into
   a per-band amplitude budget (Deng & Karam, ECCVW 2020; and Watson's DCTune,
   Proc. SPIE 1913, 1993, which has budget + Minkowski pooling + "1 = visually
   lossless" units in one paper), and the uniform-rescale range fit (Rauber &
   Bethge, arXiv 2007.07677). The low-frequency cutoff is derivable from grating
   area-summation psychophysics. The smooth reparameterisation survives, but its
   *argument* is Carlini & Wagner's change-of-variables argument and must be
   attributed. Frame the contribution as a **transfer** of a routine
   perceptual-coding construction into a localised, image-conditioned patch
   threat model on **semantic segmentation**, where no psychophysically-bounded
   attack exists at any date — a stronger positioning claim than any of the five.

## What the section must not claim

- Do not present the four-architecture numbers measured at a single lr 0.2
  (+54.54 / +27.45 / +11.91 / +14.54) as an architecture ranking. Two of the four
  peaked early and decayed; ranking on final vs best swaps them. That is a
  measurement of the optimiser.
- Do not report a band ranking from flip rates that sit inside their own spread.
- Do not claim scene-dependence of the readable band — the data does not support
  it yet.
- Do not quote a τ above the knee as achieved.
- Do not compare single-image overfit numbers against published *universal*-patch
  numbers. A per-image patch is a much easier problem.

## Open items — describe as planned, never as done

- The encoder/decoder registry restructure (T03, T04, T16 depend on it).
- LPIPS and residual RAPSD are not yet wired into the single-image diagnostics,
  so T20 has no data yet.
- The T14 join (per-image contestability and footprint luminance) is unwritten.

## How to write it

Suggested subsections: threat model and scope; dataset and preprocessing;
architectures; the perceptual constraint; attack objective and optimisation;
evaluation metrics; experimental protocol and hyperparameter selection;
reproducibility and reporting rules.

Prose, not bullet lists, wherever a paragraph will carry the argument. Every
non-obvious choice gets its one-sentence justification — a reader should never
have to ask "why that value". State units. Cite `csf_patch_procedure.tex`'s
equations by number rather than restating them in full. Where a value is a
declared assumption rather than a measurement, say which it is.

When you are done, list separately: any place the code and this prompt
disagreed, and any claim you could not source from the repository.
