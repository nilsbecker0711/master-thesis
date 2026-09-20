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
