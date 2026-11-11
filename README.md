# csf-patch (Master Thesis)

Adversarial patches for semantic segmentation under a human contrast
sensitivity (CSF) constraint, on Cityscapes.

The patch is a residual whose spectrum is bounded by the CSF, so the amplitude
budget at each spatial frequency is set by what a human observer can see at
that frequency rather than by a uniform L-infinity ball. The budget is scaled
by a single visibility threshold, **tau**, in JND units: `tau = 1.0` is
nominally the detection threshold.

---

## 1. Install

The code ran on **three** conda environments. They are separate because the
dependency sets are mutually exclusive, not for convenience: `mmcv-full 1.5.0`
is a compiled extension built against `torch 1.11.0+cu113`'s ABI, so anything
needing a current torch cannot live beside the segmentors.

| env | files | runs | python / torch |
|---|---|---|---|
| `patchreach` (on the cluster: `thesis_backup3`) | `envs/main.yml` + `envs/requirements-main.txt` | everything that builds a segmentor — all four mechanics below, `defences/`, `analysis/` | 3.9 / 1.11.0+cu113 |
| `lpips_eval` | `envs/perceptual.yml` + `envs/requirements-perceptual.txt` | all perceptual / visibility analysis: `scripts/lpips_eval.py`, `scripts/score_reference.py` | 3.10 / 2.4.1+cu118 |
| `sam3` | `envs/sam3.yml` + `envs/requirements-sam3.txt` | the cross-paradigm transfer arm: `scripts/sam3_transfer.py` | 3.12 / 2.10.0+cu128 |

Each yml supplies the interpreter (and, for the main env, the gcc 11.4 the CUDA
extensions were compiled with); the pip stack comes from the matching
requirements file, which carries its own index URLs. The reason for every pin is
in the header of each file.

```bash
# main — the two-stage pip is NOT optional: pycocotools and mmcv-full compile
# at install time and need numpy, setuptools and ninja importable first
conda env create -f envs/main.yml && conda activate patchreach
pip install numpy==1.26.4 setuptools==60.2.0 ninja==1.13.0
pip install -r envs/requirements-main.txt
pip install -e . --no-deps
```

```bash
conda env create -f envs/perceptual.yml && conda activate lpips_eval
pip install -r envs/requirements-perceptual.txt
pip install -e . --no-deps
```

```bash
conda env create -f envs/sam3.yml && conda activate sam3
pip install -r envs/requirements-sam3.txt
git clone https://github.com/facebookresearch/sam3.git && pip install -e sam3
pip install -e . --no-deps
```

`pip install -e . --no-deps` everywhere: `pyproject.toml` lists dependencies
loosely for tooling, and the requirements files are the single source of truth
for what actually ran. `requirements.txt` at the root is a pointer to
`envs/requirements-main.txt`.

`envs/iap.yml` is not a fourth environment — it is the main env plus
`torchmetrics`, and cloning beats solving it from scratch.

Two things are deliberately not installable from these files: **dcnv3**
(InternImage's CUDA op, built from its source tree; only `--arch internimage_*`
needs it) and **sam3** (Meta's source release, cloned above). Every other
architecture runs without dcnv3.

## 2. Point it at your data and checkpoints

**No environment variable is read by the Python code.** The dataset is a CLI
flag and the checkpoints are hardcoded:

```bash
python scripts/overfit.py --cityscapes_root /path/to/cityscapes ...
```

`patchreach/models/registry.py` maps an architecture name to a config and a
checkpoint through two module constants at the top of the file:

```python
_W = "/pfs/work9/workspace/scratch/ma_nilbecke-thesis"   # edit these two
_M = f"{_W}/checkpoints"
```

Point `_M` at your own checkpoint directory, or override per call with
`--cfg_path <config>.py --weights <ckpt>.pth`. The config must match the
checkpoint variant — a B0 config will not build B5. An arch with no registered
paths exits with the override instructions rather than failing later.

```
segformer_b0  segformer_b5  setr_pup  deeplabv3plus_r18/r50/r101
unet_s5d16  swin_t  convnext_t  pidnet_s/m/l  internimage_t
internimage_l  internimage_xl        <- these two have NO registered paths
```

Short aliases resolve to a default variant: `segformer`→`segformer_b0`,
`deeplab_18/50/101`, `pidnet`→`pidnet_l`, `unet`, `internimage`.

> Only SegFormer, DeepLab and SETR were used for the reported experiments. The
> others build and run only in theory, but were never tested.

> **`--arch` defaults to `internimage_t`**, the one family that needs the DCNv3
> extension. Pass `--arch` explicitly!

## 3. Run

The portable interface is the Python entry point. The 23 `*.sh` files in the
repository root are the **cluster launchers as they were used**: each hardcodes
the workspace path, the conda env and the full invocation, often with earlier
variants commented out above it. They are not parameterised by environment
variables, so `ARCH=... sbatch overfit.sh` has no effect — edit the file, or
copy the `python` line out of it.

| launcher | entry point |
|---|---|
| `overfit.sh`, `csf_single.sh` | `scripts/overfit.py` — one patch, one image |
| `universal_csf.sh`, `universal_raw.sh`, `run.sh`, `benchmarkb0.sh` | `scripts/train.py` — one patch, whole split |
| `validation.sh` | `scripts/overfit_population.py` — per-image, whole val set |
| `eval.sh` | `scripts/evaluate.py` — score a saved patch |
| `matrix.sh` | `scripts/transfer_matrix.py` |
| `iap.sh` | `scripts/iap_attack.py` |
| `lpips_eval.sh` | `scripts/lpips_eval.py` *(perceptual env)* |
| `spectral.sh` / `footprint_luminance.sh` | `measure_frequency_sensitivity.py` / `measure_footprint_luminance.py` |
| `sweep.sh`, `lr_sweep.sh`, `optimiser_ablation.sh`, `tradeoff*.sh`, `tsallis_q.sh`, `baselines.sh`, `1overnight.sh` | sweeps and ablations, several entry points each |

---

## Mechanic 1 — overfitting a single image

One patch optimised against one image. This is the strongest the attack gets:
nothing has to generalise, so it measures what the constraint permits at all.

**What happens per step** (`patchreach/patch/optimise.py`):

1. **Clean forward.** `clean_pred = model(img).argmax(1)` — the reference the
   loss and the metrics are scored against.
2. **Resolve placement.** Where the patch goes. `semantic` and `gradcam` read
   the clean prediction, which is why step 1 comes first; reversing the two
   silently falls back to centre.
3. **Render the patch.** The parameter becomes pixels. Under `csf` the
   parameter is a residual, projected so every rfft bin sits inside its CSF
   budget at `tau`.
4. **Paste** it into the frame at the resolved position.
5. **Forward and loss.** The objective is scored *outside* the patch footprint,
   so the number measures reach into the scene, not the patch covering its own
   pixels.
6. **Backward, step, re-project.** Adam updates the parameter; the projection
   puts it back inside the budget.
7. **Log** every `--log_every` steps, and checkpoint `best.pt` if this is the
   best score so far. `final.pt` is written at the end.

```bash
python scripts/overfit.py \
    --arch segformer_b0 --cityscapes_root /path/to/cityscapes \
    --img_h 1024 --img_w 2048 \
    --patch_mode csf --from_image \
    --patch_size 256 --patch_scale 0.25 --placement center \
    --csf_threshold 0.25 --csf_enforce realised \
    --loss_fn cospgd --image 420 \
    --steps 1000 --lr 0.2 --lr_schedule cosine --log_every 25
```

| flag | default | what it does |
|---|---|---|
| `--image` | `2` | which validation image to attack, by index |
| `--steps` | `300` | attack iterations. 400 was measured truncating the run; 1000 converges |
| `--lr` | `0.01` | step size. |
| `--lr_schedule` | `cosine` | `none` or `cosine`. A flat lr was measured swinging 9.6 mIoU across identical runs |
| `--optimiser` | `adam` | `adam`, or `sign` for PGD's update at equal lr |
| `--seeds` | `1` | repeat the whole attack N times and report mean ± sd |
| `--log_every` | `20` | logging interval — and the granularity at which `best.pt` is written |
| `--from_image` | off | initialise the patch base with the image region it replaces. Use with `csf` |
| `--loss_fn` | `cospgd` | `ce`, `cospgd`, `ce_target`, `ipatch_cospgd`, `tsallis`, `margin`, `cos_margin` |
| `--margin_kappa` | `5.0` | the margin the `margin` / `cos_margin` objectives drive the true class below |
| `--target_class` | `8` | target trainId for the targeted objectives |
| `--no_diagnostics` | off | skip the diagnostic suite |
| `--out_root` / `--tag` | `results/overfit` / `""` | where the run directory goes, and a suffix for its name |

> `--exclude_footprint` exists but is `default=True` with no negative form, so
> the loss is **always** scored outside the footprint. Passing it changes
> nothing.

## Mechanic 2 — the universal patch

**One** patch shared across the whole training split, pasted onto whatever
content it lands on. Nothing is fitted per image, so this is the honest threat
model and a much harder problem than Mechanic 1.

**What happens** (`scripts/train.py`):

1. One patch parameter is created and kept for the entire run.
2. For each batch: paste the *same* patch into every image, forward, accumulate
   the loss, backward, step, re-project onto the CSF budget.
3. Every `--val_every` epochs, evaluate on `--val_images` validation images and
   checkpoint if it improved.
4. `--resume` restores the patch, Adam's moments, the scheduler position and
   the RNG from `train_state.pt`, so a long run can be fed to a queue in slices.

```bash
python scripts/train.py \
    --arch segformer_b0 --cityscapes_root /path/to/cityscapes \
    --img_h 1024 --img_w 2048 \
    --patch_mode universal_csf \
    --patch_size 256 --patch_scale 0.25 --placement center \
    --csf_threshold 0.25 \
    --loss_fn cospgd --lr 0.01 --lr_schedule cosine \
    --batch_size 4 --num_workers 16 \
    --epochs 20 --val_every 5 --val_images 500
```

| flag | default | what it does |
|---|---|---|
| `--epochs` | `100` | passes over the training split |
| `--lr` | `0.005` | step size.  |
| `--lr_schedule` | `plateau` | `plateau`, `cosine`, `none`. Use `cosine` for `universal_csf` |
| `--batch_size` | `4` | images per step |
| `--num_workers` | `4` | DataLoader workers. Raise to 8–16 — Cityscapes decodes a 2048x1024 PNG per sample and can starve the GPU |
| `--val_every` | `5` | validate every N epochs |
| `--val_images` | `20` | validation images per evaluation. 500 is the full val set |
| `--n_train` | `0` | optimise on a random subset of this many training images. 0 keeps the full 2975 |
| `--train_seed` | `None` | seed for that subset |
| `--resume` | off | continue from `train_state.pt`. Requires `--out_dir` |
| `--checkpoint_every` | `1` | epochs between `train_state.pt` writes |
| `--out_root` / `--out_dir` | `results/runs` / auto | explicit output directory. `--out_dir` is required with `--resume` |
| `--config` | `None` | YAML supplying defaults, overridden by CLI flags (see `configs/experiments/`) |
| `--reach_mode` | `off` | `radial` or `empirical` adds the reach-constrained objective |
| `--loss_fn` | `cospgd` | `ce`, `cospgd`, `ce_target`, `ipatch_cospgd`, `tsallis` — **no `margin` here** |
| `--diag_image` / `--panel_images` | `2` / `0 2 5` | which images the diagnostics and panels use |

> `universal_csf` is `train.py` only; on a single image it *is* `csf`. It
> requires `--shape square`, because the spectral bound is defined on the full
> footprint. `--placement gradcam` is rejected with an explicit error, since one
> shared patch cannot follow a different image's sensitivity map each batch.

## Mechanic 3 — the whole validation set

Mechanic 1, run once per validation image, reported as a distribution. One
image is an anecdote; this is the measurement.

```bash
python scripts/overfit_population.py \
    --arch segformer_b0 --cityscapes_root /path/to/cityscapes \
    --img_h 1024 --img_w 2048 \
    --patch_mode csf --from_image \
    --patch_size 256 --patch_scale 0.25 --placement center \
    --csf_threshold 0.25 --csf_enforce realised \
    --loss_fn cospgd --steps 1000 --lr 0.2 --lr_schedule cosine \
    --images all \
    --out_dir results/validation/b0_csf
```

| flag | default | what it does |
|---|---|---|
| `--images` | `random` | `all` (full val set), `random`, `fixed10`, or a list like `'2 5 45'` |
| `--n_images` | `100` | population size when `--images random` |
| `--sample_seed` | `0` | seed for the random sample. Record it — it is what makes the subset reproducible, and raising `--n_images` with the same seed extends the sample rather than replacing it |
| `--log_every` | `50` | logging interval per image |
| `--n_panels` | `3` | how many per-image figure panels to write |
| `--panel_select` | `spread` | which images get panels; `best` picks the strongest, and the distribution figure shows where they sit |
| `--classes` | `gt` | score over the classes present in the ground truth, or their `union` with the prediction |
| `--select_key` | `drop_remote` | the metric `best.pt` and `--panel_select` rank on |
| `--no_aggregate` | off | skip the pooled diagnostic suite |
| `--resume` | off | skip images already done. Requires `--out_dir` |
| `--out_root` / `--out_dir` | `results/population` / auto | where the run goes |

Output: `summary.json` with the per-image drop and its bootstrap CI,
`distribution.png`, pooled diagnostics in `aggregate/`, and every image's patch
under `patches/`.

## Mechanic 4 — evaluation

Scores a saved patch. No training.

```bash
python scripts/evaluate.py \
    --arch segformer_b0 --cityscapes_root /path/to/cityscapes \
    --img_h 1024 --img_w 2048 \
    --checkpoint results/overfit/<run>/best.pt \
    --images all --inference auto
```

| flag | default | what it does |
|---|---|---|
| `--checkpoint` | `None` | a checkpoint **file** written by `Patch.save`, e.g. `best.pt`, not a run directory |
| `--patch_png` | `None` | evaluate the pixels of this PNG instead. With a checkpoint it replaces only the parameter, the checkpoint still supplying the geometry; on its own it is a square `raw` patch at the frame centre |
| `--patch_scale` | from ckpt | patch side as a fraction of image height. Only for `--patch_png` *without* a checkpoint; ignored when a checkpoint supplies the geometry |
| `--images` | `fixed10` | `all`, `fixed10`, or an explicit list |
| `--diagnostics_on` | `0` | how many images get the full diagnostic suite |
| `--no_panels` | off | skip the figures |
| `--loss_fn` / `--target_class` | `None` | only needed to re-score with an objective |

---

## Other entry points

Same flag vocabulary, same `results/` convention.

| script | env | what it does |
|---|---|---|
| `scripts/transfer_matrix.py` | main | does a patch survive a new image, a new model, or both |
| `scripts/clean_baseline.py` | main | clean dataset mIoU per architecture and resolution — the denominator for every drop |
| `scripts/sweep_operating_point.py` | main | fixes run length, learning rate and schedule for the single-image CSF attack |
| `scripts/measure_erf.py` | main | effective receptive field across architectures |
| `scripts/measure_frequency_sensitivity.py` | main | which frequency bands the network actually reads |
| `scripts/measure_footprint_luminance.py` | main | how dark the ground under the patch is — step 0 for the universal patch |
| `scripts/iap_attack.py` | main (or `envs/iap.yml`) | IAP (Dutta & Zhang, ICCV 2025) as a comparison arm |
| `scripts/best_checkpoint_stats.py` | main | the numbers the run log only ever printed for `final.pt`, recomputed for `best.pt` |
| `scripts/train_conditional_generator.py`, `scripts/export_conditional_patches.py` | main | the image-conditioned generator and its test-time use |
| `defences/evaluate_defended.py` | main | the same evaluation with JEDI or SAC in front of the model |
| `scripts/lpips_eval.py` | **perceptual** | LPIPS, DISTS, ST-LPIPS, LPIPS+, AHIQ, TopIQ-FR, PieAPP, SSIM/MS-SSIM/FSIM, ColorVideoVDP, NIQE, BRISQUE, CLIP-IQA, MUSIQ |
| `scripts/score_reference.py` | **perceptual** | score candidate reference images before committing a run |
| `scripts/sam3_transfer.py` | **sam3** | do the patches transfer to SAM 3 |

---

## The patch itself

These apply to every task above.

| flag | default | what it does |
|---|---|---|
| `--patch_mode` | `raw` | `raw` (free pixels), `csf` (per-image CSF residual), `universal_csf` (shared CSF residual), `lap` (Tan et al. regularisers), `gan` (BigGAN latent), `raw_ganinit` |
| `--patch_size` | `128` | the parameter's own resolution |
| `--patch_scale` | `0.25` | pasted side as a fraction of image height |
| `--patch_scale_ref` | `height` | `height`: side = `scale*H`. `area`: side = `scale*sqrt(H*W)`, constant frame coverage at any input shape |
| `--placement` | `center` | `center`, `fixed`, `semantic` (onto a class), `gradcam` (the sensitivity hotspot) |
| `--placement_class` | `0` | trainId for `semantic`. 0 = road, omnipresent in Cityscapes |
| `--placement_xy` | `0.5 0.5` | fractional position for `fixed` |
| `--placement_margin` | `0` | keep the footprint this many pixels inside the frame |
| `--pixel_param` | `sigmoid` | `raw` only. `sigmoid` never has a gradient killed by a bound; `direct` is PGD's projected pixel |
| `--raw_init` | `grey` | `random` gives the random-patch baseline — combine with `--lr 0` |
| `--logit_clip` | `6.0` | pre-sigmoid clamp, so `sigmoid` cannot saturate |
| `--shape` | `square` | `alpha`, `chroma` or `auto` cut a silhouette out of `--reference` |
| `--reference` | `None` | the image behind `lap`, the silhouette and `--from_image`-style initialisation |
| `--reference_fit` / `--shape_bg` / `--shape_thresh` | `auto` / `white` / `0.15` | how the reference is fitted and how its background is keyed out |
| `--init_from` | `None` | warm-start the parameter from an earlier checkpoint |

**The CSF constraint**

| flag | default | what it does |
|---|---|---|
| `--csf_threshold` | `0.25` | **tau**, the visibility budget in JND. 1.0 is nominally the detection threshold |
| `--csf_enforce` | `nominal` | `nominal` bounds the residual we *intend* to add; `realised` bounds what survives compositing, i.e. what an observer sees. Clipping against real content was measured inflating visibility to 2.8x tau. **Use `realised` for any number quoted with a tau attached** |
| `--csf_param` | `pgd` | `csf` only. `pgd` clamps each rfft bin to its budget and requires `--shape square`; `squash` is the original reparameterisation, kept so earlier runs reproduce and needed for shaped patches. The taus are not interchangeable — compare at matched *realised* visibility |
| `--csf_composite` | `clip` | `universal_csf` only. `clip` is the honest default and reports `frac_clipped`; `fit` rescales instead, at the cost of a per-image scale |
| `--csf_lref` | `0.0` | reference linear luminance. 0 keeps the legacy mu=0.5 convention; `0.097` is the measured Cityscapes train median at centre and is ~1.9x tighter at Nyquist |
| `--csf_model` | `barten` | `barten` or `sso` |
| `--csf_beta` | `3.0` | Minkowski exponent pooling per-bin visibility into one number |
| `--csf_min_cycles` | `2.0` | low cut, in cycles across the patch. 1/CSF diverges at DC, so the lowest bins are removed |
| `--csf_max_cpp` | `0.0` | high cut in cycles/pixel, 0 = off. The 1/CSF envelope puts most of its power above 0.5 c/px |
| `--csf_pixel_size_cm` | `0.0114` | **the observer assumption.** Every visibility claim is conditional on it |
| `--csf_viewing_distance_cm` | `50.0` | with the pixel size, this fixes cycles/pixel → cycles/degree |
| `--csf_display_peak_cd_m2` | `100.0` | sRGB reference white; read only by the luminance-aware path (`--csf_lref > 0`) |

**Model and resolution**

| flag | default | what it does |
|---|---|---|
| `--img_h` / `--img_w` | `512` / `1024` | input resolution. **Cityscapes is natively 1024x2048, which is what every reported experiment used** |
| `--scale` | `resize` | `resize` squeezes the whole frame; `crop` cuts a window from the native frame, preserving pixel density |
| `--inference` | `whole` | `whole` is one forward and what every reach measurement uses. `auto` honours the checkpoint's `test_cfg`. `slide` makes each window an independent forward, which caps reach — do not use it for cross-architecture comparisons |
| `--slide_checkpoint` | off | recompute each window in the backward pass instead of holding every window's graph. Use when the attack OOMs at native resolution |
| `--low_pr` | off | 16-bit forward. Roughly halves activation memory; compare only against a baseline run with the same flag |
| `--num_classes` | `19` | Cityscapes trainIds |
| `--seed` | `42` | RNG seed |

---

## Three things that will cost you a run

**Geometry must be pinned.** The patch is pasted without resampling only when
`--patch_size == int(img_h * patch_scale)` — at 1024 with scale 0.25 that is
256. `universal_csf` refuses to run otherwise and prints the `--patch_size` that
would fix it, because resampling a residual resamples the spectrum its budget
was projected onto. **Every other mode rescales silently.**

**`--csf_threshold` is read only in `csf` and `universal_csf`.** In `raw`,
`lap` and `gan` it is accepted and has no effect, so a "tau 0.25 raw" run is
simply unconstrained.

**`fixed10` is not arbitrary.** `--images fixed10` is the ten val indices
`[430, 46, 441, 331, 45, 162, 195, 424, 91, 353]`, defined once in
`scripts/_common.py` and shared by every script, so numbers from different
entry points are comparable. It is the default for `evaluate.py`.

## Layout

```
patchreach/              the library — no argparse, no scripts
  patch/spec.py            PatchConfig + Patch, the central abstraction
  patch/csf.py             contrast sensitivity, budgets, projection
  patch/optimise.py        the single-image attack loop
  patch/iap.py             Invisible Adversarial Patch (Dutta & Zhang, 2025)
  patch/lap.py             LAP regularisers (Tan et al., 2021)
  patch/placement.py       center / fixed / semantic / gradcam
  patch/shape.py           silhouette masking
  patch/segmentation_cam.py  Grad-CAM for a segmentor, which has no logit
  patch/conditional_generator.py   image-conditioned generator
  losses/                  adversarial.py (ce, ce_target, cospgd,
                           ipatch_cospgd, margin, cos_margin), tsallis.py, reach.py
  models/registry.py       one entry per checkpoint
  models/wrapper.py        gradient-preserving mmseg wrappers
  models/pidnet.py         vendored PIDNet backbone for mmseg 0.x
  data/cityscapes.py       dataset, palette, remap, normalisation
  metrics/                 miou.py, curves.py (reach), population.py
  diagnostics/             geometric, semantic, spectral, universal, untargeted,
                           aggregate, report
  optim.py                 SignSGD
  utils.py                 seeding, device, run directories

scripts/                 entry points; all logic lives in the library
analysis/                result aggregation and figures
defences/                JEDI, SAC, localisation metrics, defended evaluation
configs/mmseg/           PIDNet configs (the rest come from the checkpoint dir)
configs/experiments/     YAMLs for train.py --config
envs/                    the three environments, one pair of files each
tests/                   544 tests, model-free — they need neither CUDA nor data
docs/                    the procedure note and the experiment log
refs/ figures/           reference images for shaped patches; pipeline figures
GANLatentDiscovery/      vendored, for --patch_mode gan (see its licenses/)
```

Results are written flat, one directory per run holding `config.json`,
`results.json` and the patch (`best.pt`, `final.pt`).
`analysis/build_index.py` walks them into a single CSV, so a new experimental
axis becomes a new column rather than a new directory level. It defaults to
`results/runs`, so point it at the root you want:

```bash
python analysis/build_index.py --runs results/overfit --out results/index.csv
```

```bash
pytest -q
```