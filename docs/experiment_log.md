# Experiment log

## Protocol citations: evaluation resolution and inference mode

Backing for the setup claim: *all attack experiments are conducted at a common
512x1024 whole-frame operating point; clean baselines under each model's published
inference procedure reproduce the zoo values; attack effects are reported relative to
each model's own clean score at the common operating point.*

`[v]` = section read directly at the source. `[s]` = reported by another paper or a
search summary, not verified at the source. Page numbers are the printed page of the
linked PDF.

### Load-bearing: transformers AND our resolution

**Maag & Fischer, "Detecting Adversarial Attacks in Semantic Segmentation via
Uncertainty Estimation: A Deep Analysis."** arXiv:2408.10021.
<https://arxiv.org/abs/2408.10021>

- `[v]` Sec. 5.1 "Segmentation Models" (p. 8): CNNs PIDNet, DDRNet, DeepLabv3+ **and
  transformer architectures SETR and SegFormer**. Table 1 (p. 8) gives Cityscapes val
  mIoU: PIDNet 80.89, DDRNet 79.99, DeepLabv3+ 80.21, SETR 77.00, SegFormer 82.25.
- `[v]` Sec. 5.1 "Adversarial Attacks" (p. 9, 2nd para): they use the **mmsegmentation
  model zoo** (footnote 3 links the repo) and resize Cityscapes to **512x1024**,
  because the native 1024x2048 needs a large amount of memory to run **a full backward
  pass** for generating adversarial samples.

Matches our setup on three axes at once: mmseg zoo checkpoints, 512x1024, and the
backward-pass memory justification (the right reason for an attack paper, as against
Arnab's forward-pass reason). **The only paper found that covers transformer
segmenters at a stated reduced Cityscapes resolution.**

CAVEAT, do not copy their reporting: Table 1's values are the published zoo numbers
(82.25 is mmseg SegFormer-B5; 77.00 is SETR-**MLA**, not PUP), yet the attacks run at
512x1024. They place published clean mIoU beside attack results measured at a
different operating point. Our `--inference auto` baselines exist so we need not do
this. They also never state whole vs. slide.

### Protocol principle (CNN-only, but the methodological stance)

**Arnab, Miksik & Torr, "On the Robustness of Semantic Segmentation Models to
Adversarial Attacks."** CVPR 2018. arXiv:1711.09856.
<https://arxiv.org/abs/1711.09856>

- `[v]` Sec. 4 "Datasets": Cityscapes resized to **1024x512** for evaluation, because
  the native resolution needs too much memory for some models.
- `[v]` Sec. 4 "Models": each model is evaluated **the same way it was trained**; no
  CRF post-processing and no multiscale ensembling unless the network carried those as
  layers during training. The principle covering a declined test-time enhancement.
- `[v]` Sec. 4 "Evaluation metric": because model accuracies differ, they report the
  **IoU Ratio**, adversarial IoU over clean IoU per model. This is our `rel_drop`,
  adopted for our exact reason. They note the ranking under the ratio usually matches
  the ranking under absolute IoU.
- `[v]` Sec. 6.2: perturbations generated at one scale transfer poorly to another,
  because CNNs are not scale invariant. Supports attacking and evaluating at the *same*
  scale, and warns that our patches are scale-specific.
- `[v]` Fig. 4 plots IoU Ratio against clean IoU: the "does clean performance predict
  the robustness ranking" check we should run across our six.

**Halmosi, Mohos & Jelasity, "Evaluating the Adversarial Robustness of Semantic
Segmentation: Trying Harder Pays Off."** ECCV 2024. arXiv:2407.09150.
<https://arxiv.org/abs/2407.09150>

- `[v]` Sec. 5 "The evaluation procedure" (p. 9): Cityscapes scaled down to
  **1024x512**; after computing the prediction mask, **no augmentation is applied** to
  improve it.
- `[v]` Same paragraph, running p. 9 to p. 10 (the sentence hyphenates at "tiled
  evalu-" across the page break): they **deviate from related work**. Croce et al.
  resize to 512x512 then crop to 473x473; Gu et al. and Xu et al. use a **tiled
  evaluation with overlapping tiles** plus mirrored-input augmentation on clean inputs.
- `[v]` p. 10, the two sentences to cite as a pair: because of those differences their
  measurements are **not identical** to the originally published ones; but when
  applying those techniques they **are able to reproduce the published values**.
  Exactly the two-track structure of our baseline runs.
- `[v]` Sec. 5 opening (p. 9): their methodology differs from the original evaluations
  of these models, and some of those details were undocumented and had to be read off
  the implementation.
- `[v]` Sec. 5 "How to aggregate IoU?" (p. 10): Eq. (6) **CmIoU** (pooled over the
  dataset; our `SegMetric.compute()`) vs Eq. (7) **NmIoU** (averaged per image; our
  per-image mIoU). The sentence introducing Eq. (7) motivates per-image aggregation as
  emphasising errors on individual images, which they call the main goal of adversarial
  attacks.
  - When citing: their motivation for NmIoU is a VOC-shaped argument (few objects per
    image, mixed sizes) and they name PASCAL VOC, **not** Cityscapes. Cite for the
    distinction and for "the choice matters under attack", not for a demonstrated
    per-image preference on Cityscapes.
- `[v]` Sec. 4 "Investigated Models" (pp. 7-8) and the Table 1 legend (p. 9): **no
  transformers.** PSPNet and DeepLabv3 (ResNet-50) on Cityscapes and VOC; UPerNet +
  ConvNeXt (tiny/small) on extended VOC 2011 only. ConvNeXt is convolutional. Their
  scope is adversarial *training* methods (DDC-AT, SegPGD-AT, SEA-AT, PGD-AT).
  - Consequence: the tiling they decline is an **optional evaluation enhancement other
    authors applied to CNNs**, not a config-mandated inference mode for transformers.
    Cite for resolution and for the methodological stance; do NOT cite as covering
    models whose published inference is sliding-window.

**Maag & Fischer, "Uncertainty-weighted Loss Functions for Improved Adversarial
Attacks on Semantic Segmentation."** arXiv:2310.17436. CNN-only companion to
2408.10021. <https://arxiv.org/abs/2310.17436>

- `[v]` Sec. 4.1 (p. 5): Cityscapes re-scaled to **512x1024** for the computation of
  the adversarial examples, to **reduce the amount of memory to run a full backward
  pass**. Models are DeepLabv3+, BiSeNet, PSPNet, DDRNet; no transformers.

### Prior art that DOES tile

**Xu et al., DDC-AT.** arXiv:2003.06555. <https://arxiv.org/abs/2003.06555>
**Gu et al., SegPGD.** ECCV 2022. arXiv:2207.12391. <https://arxiv.org/abs/2207.12391>

- `[v]` Neither paper states its inference protocol. DDC-AT Sec. 6.2 "Implementation
  Details" (p. 5) follows the hyper-parameters suggested in its ref. [49]; SegPGD
  Sec. 4.1 "Models" (p. 9) says the standard configuration of the model architectures
  is used as in its ref. [56] (the PSPNet paper). Both **inherit** rather than declare.
- `[s]` What they inherit is `hszhao/semseg` <https://github.com/hszhao/semseg>, whose
  Cityscapes evaluation applies PSPNet in a **sliding-window fashion at 713x713 over
  base_size 2048**, citing high resolution and GPU memory limits.
- `[s]` Halmosi et al. Sec. 5 confirm from reading the implementations: overlapping
  tiles plus mirrored-input augmentation.

Tiled evaluation is therefore real in this literature, but as an undocumented codebase
inheritance. Stating our protocol explicitly is a small contribution in itself;
Halmosi et al. make that point directly.

### Native-resolution patch attacks (lightweight models only)

**Nesti et al., "Evaluating the Robustness of Semantic Segmentation for Autonomous
Driving against Real-World Adversarial Patch Attacks."** WACV 2022. arXiv:2108.06179.
<https://arxiv.org/abs/2108.06179>

- `[v]` Sec. 4.1 (pp. 5-6): Cityscapes at the **full 1024x2048**. 250 images sampled
  from the training set for patch optimisation; the **entire validation set** for
  evaluating the patches. Models: DDRNet, BiSeNet, ICNet (real-time), plus PSPNet for
  the EOT-based attack. Table 1 clean mIoU: 0.78 / 0.69 / 0.78 / 0.79.
- `[v]` Sec. 4.2: patch sizes 150x300, 200x400 and 300x600 px at native. Non-EOT
  patches placed at the image centre each iteration; the EOT version uses random
  scaling 80-120% and limited random translation.

THE ARGUMENT THIS GIVES US: nobody attacks heavy transformers at native resolution.
Nesti et al. get native because their models are real-time and cheap; Maag & Fischer
get SETR and SegFormer because they downsample. Native resolution OR the heavy
global-attention architectures the ERF hypothesis requires, not both, and that is a
GPU fact rather than a preference. **Our architecture set forces our resolution.**

### Hypothesis source (bracket taxonomy)

**"Towards Robust Semantic Segmentation against Patch-based Attack via Attention
Refinement."** IJCV 2024. arXiv:2401.01750. <https://arxiv.org/abs/2401.01750>

- `[v]` Sec. 5.1 "Models" (p. 11): Nonlocal/R50 (CNN, global attention);
  SegFormer/MiT, Segmenter/ViT, UPerNet/DeiT (**global** attention); SeMask/Swin,
  UPerNet/Swin (**local** attention). This is the global-vs-local taxonomy in
  `patchreach/models/registry.py`.
- `[v]` Sec. 5.1 "Attack methods" (p. 12): unless stated otherwise, patch attacks add
  a **150x150 patch to the lower right corner** of the clean image.
- `[v]` Sec. 5.1 "Dataset" (p. 11): **ADE20K is primary**; VOC2012 and Cityscapes are
  secondary. They defer to "the settings in the mmsegmentation framework" and never
  state a resolution or an inference mode.
  - So: cite for the hypothesis, NOT as a protocol citation.

TODO: `registry.py:19` cites this as "Yuan et al. (ACM MM 2024)". The paper found is
IJCV 2024, and its model list does not include UPerNet/ConvNeXt, which the registry
note names for the 5.86 figure. Either it is a different paper or the note drifted.
Verify before the bibliography is frozen.

### Gaps: no precedent found

- **Whole-image inference for transformer segmenters.** Nothing read states whole vs.
  slide for SETR/SegFormer. Maag & Fischer give the resolution and the architectures
  but not the inference mode. This choice has to be owned, not cited.
- **Tiling truncates patch reach.** Our own argument, no precedent. Under slide each
  window is an independent forward, so a patch cannot influence pixels outside the
  windows containing it. At 1024x2048 with a centred 128px patch that caps reach at
  448/704 px L/R for setr_pup (crop 768, stride 512), while segformer and deeplab
  reach 960: an architecture-dependent ceiling on the quantity `aggregate.py` bins out
  to 1200 px, tightest on the model the vulnerable bracket rests on. This is the real
  justification in our case and belongs in the methods section.

### Novelty check (unread, flagged)

**Aggarwal et al., "OmniPatch: A Universal Adversarial Patch for ViT-CNN
Cross-Architecture Transfer in Semantic Segmentation."** arXiv:2603.20777, Mar 2026,
accepted ICLR 2026. <https://arxiv.org/abs/2603.20777>

`[s]` Abstract only. A universal patch generalising across images and across ViT and
CNN architectures without target-parameter access; adjacent to the transfer matrix.
Read before finalising the contribution statement.

## 2026-09-21: slide-inference memory at native resolution (torch 1.11 checkpoint bug)

**Symptom.** Patch attacks under `--inference slide` (SegFormer, native 1024x2048,
crop 1024 / stride 768 = 3 windows) ran out of memory. Adding `--slide_checkpoint`
did not help, and neither did `--low_pr` on top of it: `segformer_b0`, csf, image
420, fp16 died on an 80 GB A100 (77.2 GB allocated) in the backward of step 2,
inside the checkpoint's recompute (`torch/utils/checkpoint.py(368): unpack`). Step 1
completed. A b0 window should need a few GB, so 77 GB could not be activations alone.

**Cause, two layers.**

1. *By design, without checkpointing:* the loss is computed on the STITCHED logits,
   so the graphs of all windows are live at backward, roughly n_windows x one
   window's activations. That was the original OOM.
2. *The bug, with checkpointing:* `SlidingWindowSegModel` called
   `checkpoint(..., use_reentrant=False)`. In torch 1.11 (the cluster pin) that
   implementation recomputes a window on first unpack and appends every saved
   activation to a closure list `storage` that is **never popped or cleared**
   (read at the v1.11.0 source). All three recomputed windows therefore stay live
   during backward, so checkpointing saved nothing. Later torch versions release
   entries as they are consumed, which is why the code looked right.
   - Inferred, not reproduced: the stored tensors reference the recomputed graph,
     whose saved-tensor hooks reference the closure holding `storage`. That is a
     reference cycle through C++ autograd nodes that Python's GC cannot collect,
     so step 1's activations most likely outlived step 1. This matches the log
     (step 1 fine, OOM in the next backward's recompute).

**Fix (PR #40).** `use_reentrant=True`: each window's forward runs without a graph
and saves only its input. Backward recomputes that one window, backprops it and
frees it before the next, so peak is about one 1024x1024 window's forward+backward.
Guarded on `win.requires_grad` (reentrant silently returns no gradient otherwise).
Gradients are bit-identical to the un-checkpointed path in fp32/bf16/fp16 (CPU
stand-in model). Cost: every window's forward runs twice per step. Not compatible
with `--placement gradcam` (it calls `autograd.grad` through the window).

**Why b5 fits now.** The config name `segformer_mit-b5_8x1_1024x1024_160k` means 1
image per GPU at a 1024x1024 crop in *training*, which also holds weight gradients
and optimiser state. One frozen-model attack window is strictly cheaper than one of
those training steps. b5 slide runs were reported to complete after the fix; no b5
numbers are recorded here yet.

**First run after the fix** (fp32, `--scale crop` 1024x2048, `--inference auto`,
`--slide_checkpoint`; `segformer_b0`, raw, cospgd, image 420, 600 steps, cosine lr
0.1, tag `sliding_window_to_fail`):

- clean all / remote: 63.18 / 61.46. final 0.02 / 0.02. **drop remote +61.44**,
  any_flip_rate 100%. About 93% of flipped pixels go to `fence`. Reach stays about
  100% out to about 800 px (84.7-95.2% at 906-1097 px).
- The 63.18 is per-image mIoU over the 14 GT classes of image 420, not a pipeline
  fault. Wall and fence are present in GT but score 0.00 clean, and each weighs as
  much as road. The other 12 average 73.7. Not comparable to the pooled ~76% dataset
  mIoU (mmseg: 76.54); see CmIoU vs NmIoU under Halmosi et al. above. `clean_baseline.py
  --images 420` (PR #41) reproduces the single-image number.

**Precision (`--low_pr`, PRs #38/#39).** Autocast in the segmentor forward only;
logits, losses, metrics and the CSF budget stay fp32. On torch 1.11 the CUDA
bilinear upsample has no bf16 kernel, so the cluster falls back to **fp16** with a
fixed 2^10 gradient scale; without scaling, 15% of input gradients underflowed to
zero in a CPU check. Unset, the path is bit-identical to before. With the
checkpoint fix, `--low_pr` should be unnecessary for SegFormer; **prefer fp32** so
precision is not a confound, and never mix precisions in one table.

**Consequences for the text.**

- TODO: the "Native-resolution patch attacks" section above argues that native
  resolution with heavy transformers is not possible and calls this "a GPU fact
  rather than a preference". That no longer holds for **slide** inference: b0 and
  b5 attacks now run at native 1024x2048 on one 80 GB card. The argument still
  stands for *whole*-frame native (untested) and for the reach truncation of slide
  (see "Tiling truncates patch reach"). Restate it as "whole-frame at native is
  memory-bound; slide at native is feasible but truncates reach", or measure
  whole-frame native memory before claiming it.
- Any slide result from before PR #40 that finished did so in spite of the bug.
  The bug affected memory only, never gradients, so those numbers stand.

## 2026-10-05: reporting best.pt instead of final.pt for the overfit block

**Problem.** `overfit.py` saves `best.pt` whenever the remote drop sets a new record,
but every aggregate it prints and writes -- whole-image mIoU, remote mIoU,
`any_flip_rate`, realised visibility -- is computed from the patch the run ENDED on.
The only `best.pt` quantity in `results.json` is `best_drop_remote`. A run that
degraded after its peak (the `degraded_after_peak` flag) therefore has a drop from
one patch and a flip rate and a visibility from another, and those cannot share a
row of a table.

**Fix, evaluation-only.** `scripts/best_checkpoint_stats.py` reloads each run's
`best.pt` and recomputes the whole row on the image the patch was trained on, under
the architecture, resolution, scale and inference mode the run recorded in
`config.json`. Nothing is retrained.

- Reported per run: clean/best whole-image mIoU and its drop, clean/best remote mIoU
  and its drop, `any_flip_rate`, and realised visibility (nominal and local) for the
  csf arms.
- The `final.pt` columns are READ from `results.json`, never recomputed, so the table
  cannot drift from the run log it sits beside.
- Placement is taken from the checkpoint rather than re-resolved: the image is the
  one the patch was trained on, so the stored coordinate is exact, and `--placement
  gradcam` runs stay readable without rebuilding the CAM.
- `mode='csf'` always rebuilds the base from the evaluated region
  (`set_reference_from_image` after the placement is known). It is not a flag here;
  without it the base is grey and both the drop and the visibility describe a visible
  square rather than the residual.
- **Cross-check.** The recomputed `best_drop_remote` is compared against the logged
  one -- the only number that exists on both sides. A gap over 0.5 mIoU is printed as
  a warning and flagged in the CSV (`checkpoint_matches_log`); such a row means the
  reconstruction differs from the run and must not be quoted until explained.

Outputs: `<run>/best_stats.json` beside each checkpoint, plus a collected
`results/best_eval/best_vs_final_<tag>/best_vs_final.csv`. Launcher `best_eval.sh`.

**Consequence for the text.** Quote `best.pt` for the overfit block and say so once
in the caption: it is the patch the attack found, `final.pt` is where the optimiser
happened to stop, and the `delta` column in the printed table is the difference. The
white-box diagonal in `transfer_matrix_new.py` already reads `best_drop_remote`, so
this makes the overfit table and the transfer matrix describe the same patch.

## 2026-10-16: IAP (ICCV 2025) as a comparison arm — the procedure, ported to segmentation

**Why.** IAP (Dutta & Zhang, *Invisible Adversarial Patch Attack through
Perceptibility-Aware Localization and Perturbation Optimization*, ICCV 2025;
[arXiv:2507.06856](https://arxiv.org/abs/2507.06856), code
[subratkishoredutta/IAP](https://github.com/subratkishoredutta/IAP)) is the closest
published neighbour to `--patch_mode csf`. Both bound how visible the patch is with a
per-pixel visibility model rather than an epsilon-ball, and both start the patch FROM
the content it replaces. They differ in the model: csf uses a contrast sensitivity
function over spatial frequency, IAP a local-texture sensitivity map with no frequency
axis at all. Measuring both under one dataset, one set of architectures and one
geometry turns "our patch is less visible than IAP's" from a cross-paper claim into a
measurement.

**What was ported.** All four parts of the procedure, from the released code rather
than from the paper's prose where the two disagree:

1. the sensitivity map `sens = 1/(sqrt(min(std_x, std_y)) + 1e-3)` over 1x3 and 3x1
   neighbourhoods, channel-averaged;
2. localisation by the p x p window maximising a Grad-CAM response;
3. the perceptibility-regularised loss `w1*(toward target) + w2*(away from truth)
   + w3*sum(sens*|delta - base|)`, weights 1.5 / 1.0 / 3;
4. the colour-constant update `delta <- delta - lr*mean_c(grad)*delta/sens`,
   plus the `(0.05, 0.09, 0.2)` restart ladder at 1000 steps per rung and the
   early stop on attack success.

**What changed, and nothing else did.** The two adversarial terms. IAP's are
classification cross-entropies on one logit vector; here they are this repository's
own segmentation objectives from `losses/adversarial.py`, scored over every remote
pixel — `ipatch_cospgd` for the targeted term, `ce` for the term away from the truth.
The signs line up without a negation, because every loss in that module is already
defined so that minimising it strengthens the attack, which makes IAP's explicit
`- CE(pred, truth)` literally `ce_loss`. Parts 1, 2 and 4 are pixel-space operations
and are task-blind; they are ported unchanged.

IAP's parameterisation is this repository's `--patch_mode raw --pixel_param direct`
(the parameter IS the pixel, projected back into [0,1] after the step), so no new
`PatchConfig` mode was added — what IAP contributes is a step rule and a regulariser,
not a new patch object. `pixel_param sigmoid` is **refused**: the update multiplies by
`delta` itself, which under a sigmoid parameterisation is a logit, negative for any
pixel below 0.5, so the step would reverse sign for half the patch and the run would
still complete.

**Eight declared deviations**, each named in `patchreach/patch/iap.py` as D1-D8 and
each belonging in the write-up. The two that change numbers rather than mechanics:

- **D1 — the paper and the code localise differently.** The paper scores a window by
  `J_y / Sens`; `utils.py` computes exactly that as `perturbPrior` and then calls
  `find_location` with the **raw CAM**, so the released localisation is Grad-CAM
  alone. Both are available (`--iap_locate cam | cam_over_sens`); the default is the
  released behaviour, because that is the procedure the published numbers came from.
  A test pins that the two place the patch differently on real maps, so the flag
  cannot quietly stop doing anything.
- **D3 — `w3 = 3` is not transferable to a 256px patch.** The regulariser is a SUM
  over patch elements and the patch grew from 84x84 to 256x256, so at equal mean
  `|delta - base|` the term is **9.3x larger** against adversarial losses that are
  per-pixel means and have not changed scale. `--iap_reg_reduce sum84` rescales by
  `(84/p)**2` to restore the published operating point; `iap.sh` sweeps `--iap_w3`
  across 0.3-30 rather than inheriting 3. This was visible in the first smoke run at
  p=16: the regulariser reached 77 while both adversarial terms stayed near 2.7, i.e.
  the loss was optimising the patch back to the clean image.

The rest: D2 Grad-CAM (segmentation CAM in place of a classification logit; the
reference's `eigen_smooth`/`aug_smooth` are not reproduced), D4 the ladder's failure
branch (the released code reports stale values from the previous rung when every rung
fails; here every rung is recorded and the best by remote drop is the headline), D5 the
early stop (`argmax == target and softmax >= 0.9` has no single-label analogue, so the
criteria are built on `metrics/miou.attack_rates`, and all of them are strictly harder
to satisfy than a one-label flip), D6 clipping (the paper says it clips, the code never
does; clipping is on by default via `Patch.project()` and `--iap_no_clamp` reproduces
the code), D7 the sensitivity map's two-pixel border band at `sens = 1000`
(reproduced, not fixed), D8 footprint exclusion from both adversarial terms (mandatory
for segmentation or the cheapest win is to paint the patch itself in the target class).

**Where the numbers come from.** `scripts/iap_attack.py`, one patch per image over
`--images`, the per-image threat model. The right neighbour for these rows is
`overfit_population.py`, **not** `train.py` — a universal patch is a different and
much harder threat model, and the two sets of rows must not be merged. The record
schema matches `optimise.attack_image`'s for every shared key, so
`analysis/build_index.py` indexes an IAP run beside a csf run with no special case.

**Stealthiness.** SSIM and PSNR, footprint and whole frame, are computed in-run
because the reference computes them. The rest of the paper's suite (LPIPS, CLIPScore,
UIQ, SRE) comes from `scripts/lpips_eval.py`, which already quotes IAP as its reason
for reporting LPIPS over the patch region: the run writes its `a_clean.png` /
`c_patched.png` panels and per-image `best.pt` / `final.pt` in the layout that
script's `png` and `checkpoint` sources already discover, so no new plumbing.

**Cost.** Three rungs x 1000 steps, restarted per rung, and the early stop almost never
fires in segmentation (D5) — so budget ~3000 full-resolution forward+backward passes
per image at 1024x2048, tens of minutes each. `--iap_steps` and `--iap_lrs` cut it and
then the run is no longer IAP's own setting; say which was used.

**Environment.** `envs/iap.yml`, a separate env as the lpips_eval arm is. It needs
exactly one package the attack env does not — `torchmetrics`, for the reference's SSIM
and PSNR — and deliberately none of `pytorch_grad_cam`, `opencv` or `timm`: the
reference's Grad-CAM is built for a Swin classifier with a `reshape_transform` and
there is no classification logit in a segmentor, so localisation goes through
`patch/segmentation_cam.py`. The reliable route is a clone of the attack env plus that
one package; building the torch 1.11 / mmcv 1.x / mmseg 0.x triple from scratch is a
solve-time gamble and the two mmcv/mmseg pins in the file must be confirmed against
the working env. `--arch internimage_*` is unavailable there (it needs `mmseg_custom`
and a compiled `ops_dcnv3`).

**Verification.** `tests/test_iap.py`, 37 tests. The reference's double Python loops
are transcribed verbatim in structure and the vectorised port is held equal to them —
the sensitivity map (including on near-constant content, where the sum-of-squares
shortcut for variance is fp32 noise and the stacked form is exact), the window search
(the reference maximises a sum, `find_max_response_placement` a mean; equal argmax only
because the window size is constant), the regulariser, and the update rule (the
reference's channel axis is 1 on `[1,3,p,p]`, this repository's is -3 on `[3,p,p]`, and
a mean over the wrong axis still runs). Colour constancy is pinned as an invariant:
`step/value` must be identical across the three channels. The two documented corners of
the multiplicative update are pinned too — a pixel at exactly 0 is frozen for ever, and
a flat neighbourhood moves by ~1e-3 of what a textured one does under the same
gradient. **No IAP run has been launched yet**; the numbers are not in `results/`.

## 2026-10-17: IAP's released regulariser has no gradient (deviation D9)

**Found by a step-by-step verification of the port**, not by the unit tests — the
tests pin each of IAP's four parts in isolation, and this is a property of how the
reference's *loop* is wired.

**What the reference does.** `src/IAP.py` builds the patch and its reference as

```python
delta = torch.nn.Parameter(delta)
temp  = delta.clone()
```

`Tensor.clone()` on a leaf that requires grad is **differentiable**: `temp` carries
`grad_fn=CloneBackward0` back to `delta`. So for the perceptibility term
`R = sum(sens * |delta - temp|)` autograd walks two paths into one leaf,

```
dR/d(delta) through delta = + sens * sign(delta - temp)
dR/d(delta) through temp  = - sens * sign(delta - temp)
```

and they cancel **exactly, elementwise, at every step**. Verified numerically on the
reference's own expression, transcribed verbatim: with `|delta - temp|` held at 0.1
the term is worth 2542.7 and what reaches `delta.grad` is `0.000e+00`.

**What that means.** The perceptibility regulariser changes the *reported* loss — it
is the paper's "human distance" column — and changes the *optimisation* not at all.
Whatever invisibility the released code achieves comes from the `1/sens` factor in
the colour-constant update and from initialising at the image region, i.e. from
parts 1 and 4 of the procedure, not from part 3. A direct consequence: **under the
released code `w3` cannot affect the result**, so a `w3` ablation run against it is a
null experiment by construction. `iap_attack.py` refuses `--iap_reg_grad released`
together with a non-default `--iap_w3` for that reason, rather than producing
identical runs under different directory names.

**The choice made.** The default is the **paper**: `base` is detached, so the
gradient is live and equals `w3 * sens * sign(delta - base)`. This is the one place
the port deliberately prefers the paper over the released code, and the reason is
that the alternative is to reproduce a method whose central regularisation term
provably does nothing — which is not what "IAP" denotes in the literature and not
how a reviewer would read the name. Note the contrast with **D1**, which goes the
other way, to the released code: there both branches are coherent procedures, while
here one of them is a silently dropped term.

`--iap_reg_grad released` reproduces the cancellation exactly, by adding the term's
value to the loss **detached** — identical value, identical (zero) gradient. So "the
regulariser contributes nothing" becomes a measured row rather than an argument, and
the two arms stay comparable on the human-distance column. `iap.sh` carries the arm.

**Pinned by three tests**, one of them on the reference's own expression so the
finding survives independently of anything in `patchreach/patch/iap.py`:
`test_the_released_regulariser_has_no_gradient`,
`test_the_ported_regulariser_does_have_a_gradient`,
`test_reg_grad_released_reports_the_value_but_not_the_gradient`.

**The rest of the verification passed**, 16 checks, and three are worth recording
because nothing else in the suite covers them:

- the `_rung` trajectory is **bit-identical** to a literal reference-structure loop
  over five steps (max `|param_mine - param_ref|` = 0.000e+00 at every step), holding
  the objective fixed so only the mechanics are under test — paste, channel-mean
  gradient, multiplicative step, `1/sens` scaling, and the order of grad-zero,
  stop-check, backward and update;
- `Patch.apply`'s normalised composite equals `norm()` of the reference's in-place
  slice paste to 0.0 — the `F.pad` + mask construction and the in-place assignment
  agree exactly, so the gradient path through `1/std` is the same on both sides;
- the record's `final_remote`, `ssim_patch` and `psnr_image` are all measured on the
  patch the object is left holding, which is the parameter written to `best.pt` —
  the 2026-10-05 best/final invariant, now also true one level up across the rung
  ladder. A stop that fires overrides best-by-drop for that rung, so
  `iap_stopped_at` and the saved patch cannot disagree.

Two apparent failures in the first verification run were artefacts of the verifier
itself and are noted so they are not rediscovered: `prepare()` recovers `[0,1]` by
un-normalising, which round-trips through float32 and lands ~6e-8 from the original
frame, so the sensitivity map needs `allclose` rather than `equal`; and the
invisibility metrics must be scored against that same round-tripped frame, or the
6e-8 is attributed to the patch — which is invisible at any real drop but moves PSNR
by 0.03 dB at 138 dB, where the stub patch has barely moved and the metric is
measuring float noise.

## 2026-10-17 (correction): Grad-CAM under slide inference failed SILENTLY, not loudly

**Correcting the same day's claim.** The `df0bbc4` commit message and the first version
of the guard in `patchreach/patch/segmentation_cam.py` both stated that a CAM on a
tiled model dies with `RuntimeError: The size of tensor a (128) must match the size of
tensor b (80)`. **That was wrong**, and it understated the problem. The RuntimeError
came from a stub whose `encode_decode` did not resize logits back to the input size;
every real mmseg `EncoderDecoder` does, so the padding arithmetic in
`SlidingWindowSegModel.forward` is consistent and nothing raises.

**What actually happens.** `SlidingWindowSegModel` calls the segmentor once per window.
`SegmentationCAM._hook` does `self._feats = out` — an **assignment, not an append** —
so after a tiled forward only the **last** window's feature map survives, which is the
bottom-right tile because `slide_windows` yields row-major. The rest of `__call__` then
proceeds as if that map described the whole frame:

- `autograd.grad(score, A)` succeeds — the last window *is* in the graph, so this is
  the gradient of a whole-frame scalar with respect to one tile's activations;
- `as_spatial(A, H, W)` passes a 4-D tensor through untouched, so the map stays
  window-sized;
- `F.interpolate(cam, size=(H, W))` stretches the tile's CAM across the full frame;
- per-sample min-max normalisation then makes it look like any other map.

Measured on a hand-rolled tiled model at 64x256 with a 64-crop / 48-stride (5 windows):
the 4-D path returns a map that is **0.055 mean absolute distance** from the whole-frame
map on a `[0,1]` normalised scale, with no error. The token path is worse: `_token_grid`
factorises the **window's** 256 tokens against the **frame's** 64x256, finds stride 8 ->
`(8, 32)` instead of the true `(16, 16)`, and reshapes the tokens into a transposed,
meaningless map — again with no error. This is exactly the failure `_token_grid`'s own
docstring warns about, one level up.

So the pre-guard behaviour was a **plausible map of the wrong region**, which is the
failure class `segmentation_cam.py`'s header exists to prevent, and the guard added in
`df0bbc4` is worth more than its commit message claimed: it converts a silent wrong
result into a refusal.

**AUDIT ITEM.** `--placement gradcam` with `--inference slide` or `auto`, and **without**
`--slide_checkpoint`, produced a silently wrong patch placement: the window was chosen
from a stretched bottom-right tile rather than from the frame. Any such run on record
has a `placement` that does not mean what `placement_policy: gradcam` says it does, and
its `placement_dist_from_centre` is not interpretable. `overfit.py`,
`overfit_population.py` and `train_conditional_generator.py` all accept the combination
and nothing refused it. With `--slide_checkpoint` the run failed instead, because the
reentrant checkpoint does not support `autograd.grad` through a window — so only the
un-checkpointed slide runs are affected. Check `config.json` for
`placement: "gradcam"` together with `inference` in `("slide", "auto")` and
`slide_checkpoint: false` before quoting any such run.

Nothing else changes: `--inference whole` was always correct, and for SegFormer/Swin
slide and whole are the same computation at 1024x1024 and below, so only full-width
native-resolution gradcam runs could be affected.

## 2026-10-17: D10 — Grad-CAM localisation under slide inference

**Why it was blocked.** A tiled model has no frame-shaped feature map to weight.
`SlidingWindowSegModel` calls the segmentor once per window, so a forward hook on the
backbone fires `n_windows` times and `SegmentationCAM._hook` assigns rather than
appends — only the last window's activations survive, and the rest of the CAM treats
one tile as the frame. It does not raise; see the correction entry above for the
measured behaviour. `SegmentationCAM` now refuses a `SlidingWindowSegModel` outright.

**The way out.** The CAM is a **pre-optimisation artefact**: `iap.prepare()` calls it
exactly once, before the loop, purely to choose the `p x p` window. Nothing requires it
to share the attack's inference mode. So the map is now computed from one whole-frame
forward on the **un-tiled** model (the `WrappedSegModel` the slide wrapper holds at
`.model`), while every attack step runs tiled, with `--slide_checkpoint`, exactly as the
rest of the full-resolution corpus does. `scripts/iap_attack.py` resolves this after
`setup_model` — `--inference auto` only becomes slide once the checkpoint's `test_cfg`
has been read — and asserts the un-tiled model rather than reaching blindly through a
wrapper.

This means `--iap_locate cam` now works at native 1024x2048 against the published slide
protocol, which is what makes the IAP arm comparable to the existing csf corpus on
**resolution, patch size, inference mode and localisation** at once. The
`--slide_checkpoint` refusal is gone entirely: it has been wrong twice (first blanket,
then conditional on `--iap_locate`) and is now unnecessary, because nothing calls
`autograd.grad` through a checkpointed window any more. The combination that genuinely
cannot work is no longer expressible.

**State it both ways, because it cuts both ways.** In favour: the whole-frame map is the
better localisation signal — it is not truncated at tile boundaries, and a tiled map
would be an artefact of the tiling grid rather than of the model's sensitivity. Against:
the map and the attack no longer see the same model behaviour, so "IAP localised this
patch" and "the attack achieved this drop" are statements about two different forward
procedures. The record carries `iap_cam_untiled` and `iap_cam_inference`
(`whole` / `same_as_attack`), and `config.json` carries the former, so no row or
directory can hide which pair produced it. **Quote the pair, not just the drop.**

**The trap inside it, and the test that holds it.** The CAM's own logits must be
**discarded**. Everywhere else `prepare()` reuses them to save a forward, which is free
when the CAM and the attack share a model. Here they do not: reusing them would measure
`clean_all` / `clean_remote` under whole-frame inference and the attacked frame under
slide, making every drop in the run the difference of two inference modes. A slide
clean mIoU and a whole clean mIoU differ by a point or two, so the error would be
invisible rather than obvious. `prepare()` therefore runs a separate clean forward on
the attack's model, and
`test_cam_untiled_takes_the_clean_baseline_from_the_attack_model` pins it by giving the
two models deliberately different constant outputs and asserting which one answered.

**Cost, and what is NOT yet measured.** One un-tiled forward *and* backward at the full
input size, per image — precisely what the 2026-09-21 entry calls memory-bound at
native resolution. Slide plus checkpointing makes the attack *loop* cheap, not this
pass. `prepare()` resets the CUDA peak counter around the CAM call, logs the figure as
`[iap  ] cam pass  : whole-frame ... peak N.NN GiB (D10)`, and converts an OOM into a
message naming `--iap_locate center` as the fallback. **No GPU number exists yet** — all
of this was developed and tested on CPU with stub segmentors, so the first native run is
also the first measurement of whether the pass fits. Read that log line before trusting
a sweep, and if it does not fit for the larger backbones, record that here and keep
`--iap_locate center` for them rather than assuming b0 generalises.

Tests: 45 in `tests/test_iap.py`, 498 in the suite
(`test_core.py::test_tv_zero_on_constant` was already failing and is unrelated), and
both standalone verifiers still pass.

## 2026-10-17: IAP's first real runs — lr*w3 is a limit cycle, and a best-patch off-by-one

Three findings from the first two IAP runs on segformer_b0, image 464, 1024x2048.
The runs themselves are not quotable (see the bug below), but the mechanisms are.

### 1. `lr * w3` is the stability control parameter, and 0.3 oscillates

The regulariser's gradient is proportional to `sens` and the colour-constant update
divides by `sens`, so **the two cancel**: the regulariser contributes a
sens-independent relative change of `lr * w3 * sign(delta - base)` per step. The `* delta`
factor then shrinks toward **zero**, not toward `base`, so the step overshoots `base`,
the sign reverses, and the dynamic becomes a limit cycle rather than a descent.

Measured on the regulariser alone — no model, no adversarial term — starting 0.05 above
base, 200 steps, reporting mean `|delta - base|`:

| lr | w3 | lr*w3 | step 1 / 2 / 3 / 10 / 200 | swing over the last 50 |
|---|---|---|---|---|
| 0.1 | 3.0 | **0.300** | 0.117 / 0.034 / 0.062 / 0.088 / 0.055 | 0.0226 **limit cycle** |
| 0.1 | 1.0 | 0.100 | 0.024 / 0.025 / 0.027 / 0.024 / 0.028 | 0.0029 limit cycle |
| 0.1 | 0.3 | 0.030 | 0.033 / 0.019 / 0.018 / 0.018 / 0.020 | 0.0003 settled |
| 0.01 | 3.0 | 0.030 | 0.033 / 0.019 / 0.018 / 0.018 / 0.020 | 0.0003 settled |
| 0.003 | 3.0 | 0.009 | 0.045 / 0.040 / 0.035 / 0.016 / 0.018 | 0.0001 settled |

Rows 3 and 4 are **identical trajectories** at different `lr` and different `w3` with the
same product, which is the proof that the product alone controls the dynamic. The
observed run at `lr 0.1, w3 3.0` swung `|d|` between 0.0041 and 0.0513 across its log
rows — the predicted amplitude.

**Operating rule: keep `lr * w3` at or below ~0.03.** And note the convergence with D3:
`--iap_reg_reduce sum84` scales the regulariser by `(84/256)**2 = 0.1076`, i.e. an
effective `w3 = 0.32`, giving `lr * w3 = 0.032` at `lr 0.1`. The patch-size rescaling and
the stability boundary land on the same number independently, which is a strong argument
for making `sum84` the arm that gets swept.

### 2. An edge-placed patch under slide can only reach half the frame

The Grad-CAM put the patch at `(623, 1792)` — flush against the right edge, 927px from
centre. Under slide at 1024x2048 the windows are `[0,1024]`, `[768,1792]`, `[1024,2048]`,
and that patch lies inside **exactly one** of them:

```
patch at (623, 1792): reachable 1,048,576 / 2,097,152 px  (50.0% of the frame)
centred patch:        reachable 2,097,152 / 2,097,152 px  (100.0%)
```

So `target_hit_rate` could not exceed ~50% before the attack started, and the run's
`--iap_stop_thresh 0.99` was **unreachable by construction**. The whole-inference run at
the same placement reached 91.1% flip and +49.01; the slide run reached 1.5% and +5.13.
That gap is the tiling ceiling interacting with an edge placement, NOT the threshold —
and the two runs differed in both at once, so neither can be attributed from them.

This is the concrete instance of the reach truncation that `wrapper.py` warns about, and
it is worse than the architecture-dependent ceiling recorded there, because IAP's
localisation actively seeks edges: the sensitivity map's hottest ridge is the near-field
road boundary, and `--placement_margin` (which exists for exactly this) defaults to 0.
**Set `--placement_margin` when running IAP localisation under slide**, or expect the
window to be chosen in the one place where tiling costs the most.

### 3. BUG, now fixed: `best_param` was one update past the drop it was credited with

`_rung` computes `cur` from `logits`, i.e. from the parameter as it stood at the forward
pass, and then cloned `best_param` **after** the update had been applied. Every logged
step therefore credited its measured drop to a different patch.

`optimise.attack_image` has carried the same pattern harmlessly, because Adam under a
cosine schedule moves the patch by ~lr per coordinate and consecutive patches are
near-identical. Under IAP's multiplicative limit cycle they are not: the slide run
reported `best_drop_remote = +12.17` while the parameter it saved measures **+5.13**.

Fixed by snapshotting the candidate before the update on logged steps (one clone per
logged step). `drift_mean_abs` now also describes the measured patch rather than the
post-step one. Pinned by `test_best_drop_is_reproducible_from_the_saved_patch`, which
checks that step 1's history row reports **exactly zero** drift — the forward saw the
un-updated patch, which is the image region — and which was verified to FAIL with the
old code rather than being vacuous. An earlier version of that test did pass with the
bug present, because on a stub the drop is flat and the post-loop best/final comparison
overwrites the candidate either way.

**Consequence: neither of the two runs above is quotable.** Re-run them.

### 4. PSNR is None, not inf, on identical frames

Exposed by the fix: a run whose regulariser dominates never beats its own step-1 drop, so
its best patch *is* the clean content, MSE is 0 and PSNR is `+inf`. `json.dump` writes a
bare `Infinity`, which Python reads back but which is not valid JSON, and `results.json`
is read outside this repository. `ssim_psnr` now returns `None` there; SSIM already
reports 1.0, which carries the same fact without leaving the grammar, and
`_invisibility_stats` skips None. That failure mode is not contrived — it is exactly what
a too-large `w3` produces.

## 2026-10-23: IAP's training dynamics are a 40-step limit cycle, and the early stop hides it

Measured on the `log_every=10` run (segformer_b0, whole inference, image 464, 1024x2048,
`lr 0.1`, `w3 3`, `--iap_stop target 0.9`, stopped at step 592 reporting **+48.99**).
This supersedes the impression that the run is merely "noisy": it is a **deterministic,
periodic collapse-and-rebuild**, and the number a run reports depends on where in the
cycle it stops.

### The signature

| | reg | mean &#124;delta-base&#124; |
|---|---|---|
| logged steps with drop > 35 (n=13) | 2.7e5 | 0.0407 |
| collapsed steps (n=10) | 7.1e5 | 0.0173 |

`corr(reg, |d|) = -0.758`, `corr(reg, drop) = -0.493`, `corr(reg, target) = -0.522`.
So a collapse is **2.6x the regulariser at 0.42x the drift** — the perturbation has not
merely shrunk, it has moved into the high-`sens` (flat) pixels where it buys nothing and
costs everything. The patch's own targeted loss returns to **~19**, which is its value at
step 1: the attack is being undone completely and rebuilt from scratch.

### The period, and why it is exactly what the mechanism predicts

Collapsed logged steps, once the attack has exceeded +20: 260, 300, 310, 350, 390, 430,
470, 510, 520, 560 — spacings `[40, 10, 40, 40, 40, 40, 40, 10, 40]`, i.e. a **40-step
period** with the two 10s being the two-row-wide floors of the same trough. The
`|d|` value at successive collapses walks 0.0051, 0.0099, 0.0149, 0.0198, 0.0245, 0.0292
and then wraps to 0.0054 — a slow beat on top of the 40-step cycle, so it is
quasi-periodic rather than a clean period-40 orbit.

The width of the collapse matches the mechanism arithmetically. From the 2026-10-17
entry, the regulariser's contribution to the multiplicative update is sens-independent
and equal to `lr * w3 * sign(delta - base)` per step, so with consistent sign the
perturbation is multiplied by `(1 - 0.3) = 0.7` each step:

```
0.7 ** 10 = 0.028      ten steps remove 97% of the accumulated perturbation
```

Ten steps is exactly the observed collapse width, and the ~30 remaining steps of each
cycle are the attack rebuilding against a 30%-per-step headwind. Duty cycle predicted
~25% collapsed; measured 10 of 30 logged steps after step 300 are below +10, i.e. 33%.

### Why this makes the reported number a biased estimator

Over the logged steps from 300 onward:

```
min +4.27   max +46.51   range 42.24 mIoU points   sd 16.84   median +33.62
```

A run with `--iap_stop none` would report **wherever the sawtooth happened to be at step
1000** — a 42-point coin flip, and below +10 a third of the time. That is exactly what
the slide run did: final +5.13 against a best of +12.17.

With `--iap_stop target 0.9` the run instead stops **the first time the cycle crosses the
threshold**, so it reports a peak *by construction* and can never report a trough. The
early stop is therefore doing load-bearing work, not saving compute, and `+48.99` is not
"the attack's strength" — it is "the first peak that crossed 0.9". `best_drop_remote`
carries the same upward bias, being a max over logged samples of a cycle.

**So for an oscillating run there is no unbiased summary**, and the three candidates fail
differently: the final is cycle phase, the best is a max over samples, and the
stop-based number is a defined-but-biased operating point. The fix is not to pick one but
to leave the regime: **keep `lr * w3` at or below ~0.03**, where the reg-only dynamic
settles (swing 3e-4 against 2.3e-2 at 0.3). `--iap_reg_reduce sum84` does it at an
effective `w3 = 0.32`, which is also what D3's patch-size rescaling independently asks
for.

### What to re-run, and what to report

- Re-run the arm at `lr * w3 <= 0.03` and check the log is monotone-ish rather than
  sawtoothed before quoting anything.
- Where an oscillating run must be quoted (e.g. to document this failure), report the
  stop step, the threshold, AND the spread of the logged drops over the last few hundred
  steps. A single number from a limit cycle is not a measurement.
- The instability is a property of **IAP's own update rule at its own published
  weights**, not of this port: `w3 = 3` is the paper's value, and `lr 0.1` is within its
  published ladder (0.05, 0.09, 0.2 all give `lr * w3` of 0.15-0.6, i.e. every rung of
  the released ladder is in the limit-cycle regime at this patch size). That is a result
  about IAP worth stating, and it is inseparable from D3: the regulariser's scale grows
  with patch area, so the published `lr * w3` lands in a different regime at 256px than
  at 84px.

### Correction, same day: the instability is NOT patch-size dependent, and D3 does not cause it

The paragraph above claimed the published `lr * w3` "lands in a different regime at 256px
than at 84px", on the grounds that the regulariser is a sum whose scale grows with patch
area. **That reasoning is wrong.** `R = SUM_i sens_i |delta_i - base_i|`, so
`dR/d(delta_j) = sens_j * sign(delta_j - base_j)` — the gradient with respect to one
element does not depend on how many elements the sum has. The per-pixel kick is
`lr * w3 * sign(...)` at every patch size.

Measured, reg-only, same `lr 0.1` and `w3 3` at four sizes:

| P | elements | &#124;d&#124; @ step 1 / 2 / 3 / 10 | swing (last 50) |
|---|---|---|---|
| 32 | 3,072 | 0.1146 / 0.0199 / 0.0648 / 0.0915 | 0.0307 |
| 64 | 12,288 | 0.1155 / 0.0202 / 0.0661 / 0.0932 | 0.0293 |
| 128 | 49,152 | 0.1152 / 0.0201 / 0.0648 / 0.0937 | 0.0294 |
| 256 | 196,608 | 0.1153 / 0.0201 / 0.0650 / 0.0931 | 0.0291 |

Identical to the resolution of the different random draws. And `sum84` at P=84 is a
no-op by construction (`(84/84)**2 = 1`, effective `w3` still 3.0, still a limit cycle);
it settles at P=256 only because it scales the term down by 0.1076, i.e. it is a `w3`
change wearing a patch-size costume.

**What this does to the finding.** It makes it stronger and simpler. The instability is
not an artefact of running IAP at a patch size it was not tuned for — it is a property of
the paper's loss at the paper's own weights and the paper's own learning-rate ladder, at
**any** patch size including 84x84. Every rung of that ladder gives `lr * w3` between 0.15
and 0.60 against a stability boundary near 0.03.

**And it connects to D9 rather than to D3.** The released code's `temp = delta.clone()`
zeroes the regulariser's gradient entirely, so the released implementation *cannot*
oscillate — and cannot regularise either. The two findings are consistent and explain
each other: the instability is exactly what a live version of that term produces at those
weights, and the released code has it inert. The paper's stated method and its released
method are different methods, and only one of them runs.

**D3 stands, but narrowed.** The regulariser's VALUE does grow with patch area, 9.3x from
84x84 to 256x256, so the logged `reg` number is not comparable across patch sizes and
dwarfs the mean-reduced adversarial terms in the printed loss. That is a reporting and
loss-balance observation. It is not the dynamics, and the phrase "not the same operating
point" in D3 should be read as being about the term's weight in the total, not about
stability — for which the operating point is `lr * w3`, full stop.
