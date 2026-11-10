# Environments

The code ran on three conda environments. They are separate because the
dependency sets are mutually exclusive, not for convenience: `mmcv-full 1.5.0`
is a compiled extension built against `torch 1.11.0+cu113`'s ABI, so anything
that needs a current torch cannot live in the same env as the segmentors.

| env | files | runs | python / torch |
|---|---|---|---|
| `patchreach` (on the cluster: `thesis_backup3`) | [`main.yml`](main.yml) + [`requirements-main.txt`](requirements-main.txt) | everything that builds a segmentor: `scripts/train.py`, `overfit.py`, `overfit_population.py`, `evaluate.py`, `transfer_matrix.py`, `clean_baseline.py`, `measure_erf.py`, `measure_footprint_luminance.py`, `measure_frequency_sensitivity.py`, `sweep_operating_point.py`, `iap_attack.py`, `defences/`, `analysis/` | 3.9 / 1.11.0+cu113 |
| `lpips_eval` | [`perceptual.yml`](perceptual.yml) + [`requirements-perceptual.txt`](requirements-perceptual.txt) | all perceptual / visibility analysis: `scripts/lpips_eval.py` (LPIPS, DISTS, ST-LPIPS, LPIPS+, AHIQ, TopIQ-FR, PieAPP, SSIM/MS-SSIM/FSIM, ColorVideoVDP, NIQE, BRISQUE, CLIP-IQA, MUSIQ), `scripts/score_reference.py` | 3.10 / 2.4.1+cu118 |
| `sam3` | [`sam3.yml`](sam3.yml) + [`requirements-sam3.txt`](requirements-sam3.txt) | the cross-paradigm transfer arm: `scripts/sam3_transfer.py` | 3.12 / 2.10.0+cu128 |

[`iap.yml`](iap.yml) is not a fourth environment: it is the main env plus
`torchmetrics`, and the reliable route is `conda create --name iap --clone
<main env>` rather than solving it from scratch. Its header explains why.

## Creating them

Each yml provides the interpreter (and, for the main env, the gcc 11.4 the
CUDA extensions were compiled with); the pip stack comes from the matching
requirements file. The full commands, and the reason for every pin, are in the
header of each file. In short:

```bash
# main - note the two-stage pip, which is NOT optional
conda env create -f envs/main.yml && conda activate patchreach
pip install numpy==1.26.4 setuptools==60.2.0 ninja==1.13.0
pip install -r envs/requirements-main.txt
pip install -e . --no-deps

# perceptual
conda env create -f envs/perceptual.yml && conda activate lpips_eval
pip install -r envs/requirements-perceptual.txt
pip install -e . --no-deps

# sam3
conda env create -f envs/sam3.yml && conda activate sam3
pip install -r envs/requirements-sam3.txt
git clone https://github.com/facebookresearch/sam3.git && pip install -e sam3
pip install -e . --no-deps
```

`pip install -e . --no-deps` everywhere: `pyproject.toml` declares the
interpreted dependencies loosely for tooling, and the requirements files are
the single source of truth for what actually ran.

## What is deliberately not installable from these files

| package | env | why |
|---|---|---|
| `dcnv3` | main | InternImage's CUDA op, built from its source tree against the pinned torch. Only `--arch internimage_*` needs it. |
| `sam3` | sam3 | Meta's source release, cloned and installed with `pip install -e sam3`. |
| checkpoints | main | `mim download mmsegmentation --config <name>`; SAM 3's weights come from its own `download_ckpt_from_hf` or `--sam3_ckpt`. |

`local/` holds a fourth, Windows-only variant of the main env (`torch 2.1.0 +
mmcv-full 1.7.2`, the only combination with a prebuilt cp310 win_amd64 wheel)
for runs on an own GPU. No thesis number comes from it.
