# patchreach

Geometric and semantic bounds on adversarial patch reach in semantic segmentation.

## Install

```bash
pip install -e .          # torch/mmcv/mmseg come from the cluster conda env
pytest -q                 # ~7 invariant tests, seconds
```

Fill the `cfg` / `weights` fields in `patchreach/models/registry.py`, or pass
`--cfg_path` / `--weights` on every call.

## Layout

```
patchreach/            importable library — no scripts, no argparse
  models/registry.py     ARCH_REGISTRY; backbone import driven by the CONFIG
  models/wrapper.py      encode_decode wrapper (aug_test destroys gradients)
  data/cityscapes.py     dataset, palette, remap, normalisation
  patch/spec.py          PatchConfig + Patch  <- the central abstraction
  patch/lap.py           L_rat, L_tv, L_nps, ASI/AGI/ADE  (Tan et al. 2021)
  patch/shape.py         silhouette masking (their Bg term)
  patch/placement.py     center / fixed / semantic
  losses/adversarial.py  ce, cospgd, ipatch_cospgd
  losses/reach.py        radial + empirical reach masks
  metrics/miou.py        SegMetric, remote mIoU, flip/hit rates
  metrics/curves.py      targeted + untargeted reach curves

scripts/               thin entry points; all logic lives in the library
analysis/build_index.py  results/runs/*/ -> results/index.csv
configs/experiments/   one YAML per thesis block
results/runs/<id>/     FLAT. config.json + results.json + patches
```

## Two decisions worth knowing

**Parameterisation is unified on `sigmoid`.** The old single-image script used
raw pixels with an in-loop clamp; the trainer used logits through a sigmoid.
That split caused two initialisation bugs and forced every init path to be
written twice. Sigmoid everywhere: no clamp, no dead gradients at the `[0,1]`
boundary, one code path. `gan` mode still clips its latent, which is correct —
BigGAN was trained on `z~N(0,1)`.

**Results are flat, the index is generated.** Nested result paths broke as soon
as a new axis appeared (untargeted runs filed under `cls-1`). Four axes were
added in three weeks. Flat store + `build_index.py` means a new axis is a new
column.

## Ordering constraint

Semantic placement reads the *clean prediction*, so:

```python
clean_pred = model(img).argmax(1)[0]      # 1. clean forward
patch.resolve_placement(H, W, clean_pred) # 2. resolve
patched, fp = patch.apply(imgs)           # 3. apply
```

Getting this backwards silently falls back to centre. `scripts/train.py` shows
the correct order.

## Experiment blocks

| block | runs | what it establishes |
|---|---|---|
| 0 | 0 | clean dataset mIoU per arch x resolution — pick the fair comparison point |
| A | 0 | ERF across InternImage / Swin / SegFormer (label-free probe) |
| B | 2 | architecture headline |
| C | 4 | ce vs cospgd vs ipatch |
| D | 6 | resolution transfer matrix |
| E | 8 | realism ladder (InternImage only) |
| F | 2 | reach-restricted optimisation |
| G | 4 | targeted class sweep across the contestability range |
| H | 0 | multi-image validation of everything above |

Blocks 0, A and H need no training.
