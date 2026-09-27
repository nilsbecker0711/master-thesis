#!/bin/bash
#SBATCH -p dev_gpu_a100_il
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000
#
# ONE SLICE of the SegFormer-B0 Nesti benchmark.
#
#     bash benchmarkb0.sh --list        the config table (login node, no GPU)
#     sbatch benchmarkb0.sh raw         one 30-minute slice
#     sbatch driverb0.sh                let the driver feed slices
#
# THE CONFIG NAME IS REQUIRED. Every other launch script in this repo takes
# no argument; this one does.
#
# Each slice passes --resume and the FULL --epochs. A slice stops because
# SLURM kills it at the walltime, never because it was told to do less:
# --epochs sets the cosine T_max, so a shorter slice would run a different
# optimisation. train.py checkpoints every epoch (param, Adam moments,
# scheduler position, gstep, RNG); tests/test_train_resume.py pins that a
# sliced run reproduces an uninterrupted one bit for bit.
#
# ── PROTOCOL: Nesti et al. WACV 2022 / Rossolini et al. TNNLS 2024 ───────────
# Adopted: 1024x2048 native, 250 sampled TRAIN images, the entire 500-image
# val split, Adam, 200 epochs, mIoU+mAcc outside the patch area, and a
# random-patch baseline column (under an outside-the-patch metric the clean
# number is a different pixel set, so it is not the comparison).
#
# Say "we adopt their DATA PROTOCOL", never "their training procedure":
#   1. NO EOT. Theirs adds 5% noise, 80-120% scaling and bounded translation.
#      Patch.apply() composites one (top,left) for the whole batch, so this
#      is their non-EOT arm. For universal_csf the scaling half also fights
#      the budget: resampling a residual resamples its spectrum.
#   2. ONE SIZE, scale 0.25 (256px), not their 150x300/200x400/300x600
#      ladder, and square not 1:2. 3.125% of frame -- between their first
#      two rungs, matching neither. Never quote a row against a rung.
#   3. NOT their lr 0.5 (tuned on an unconstrained pixel patch).
#   4. NOT their models. DDRNet/BiSeNet/ICNet are real-time, which is why
#      they get native resolution.
# "200 epochs" is not transferable without their batch size; what is matched
# is IMAGE PRESENTATIONS, 250 x 200 = 50,000.
# ─────────────────────────────────────────────────────────────────────────────

set -u

# ── the repo root ────────────────────────────────────────────────────────────
# NOT dirname "$BASH_SOURCE": sbatch copies this script to
# /var/spool/slurmd/job<id>/slurm_script and runs the copy, so that resolves
# to the spool directory. SLURM already starts the job in the directory
# sbatch was invoked from and names it SLURM_SUBMIT_DIR. The BASH_SOURCE form
# is only the fallback for running this by hand on a login node.
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"

ARCH=segformer_b0
CS="${CS:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes}"
OUTROOT="results/runs/bench_nesti/$ARCH"

# Geometry. 0.25 is the operating point the rest of the thesis uses, and
# because scale_ref='height' fixes coverage by aspect ratio alone it is
# 3.125% of the frame at 512x1024 and at 1024x2048 alike -- so this row stays
# comparable to our own work. 256 = int(1024*0.25); universal_csf refuses a
# mismatch between the two.
GEOM="--patch_size 256 --patch_scale 0.25"

# --inference auto = SLIDE for segformer (whole for deeplab/unet). Attacking
# under a mode nobody deploys flatters the attack. The usual objection --
# that slide caps a patch's reach, since a patch cannot influence pixels
# outside the windows containing it -- does NOT apply here: at 1024x2048 the
# 1024 crop / 768 stride grid is [0,1024] [768,1792] [1024,2048] and a
# centred 256px patch is in all three, so the cap is zero. Recompute with
# patchreach.models.wrapper.slide_windows before changing arch, resolution
# or placement (setr_pup's 768 crop DOES cap it). The ERF/reach tables stay
# on whole for that reason.
# --slide_checkpoint recomputes each window in backward instead of holding
# all three graphs; without it the backward is ~3x the activations.
INFER="--inference auto --slide_checkpoint"

# --train_seed 42 is what this run was launched with. (68 is the seed
# T13/T15 share; this table is not paired with them, so it does not matter.)
PROTO="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048 \
       --placement center --n_train 250 --train_seed 42 $INFER \
       --val_images 500 --val_every 50 --epochs 200 \
       --optimiser adam --lr_schedule cosine --batch_size 1 \
       --num_workers 16 --loss_fn cospgd --no_diagnostics --resume"

# ── the config table: name <TAB> out_dir <TAB> args ──────────────────────────
# The driver reads this back through --list, so out_dir lives in ONE place.
#
#   raw       the unconstrained patch. 200 epochs, many slices.
#   raw_null  random PIXELS at lr 0 -- the baseline column. A raw patch
#             starts at uniform grey, so --lr 0 alone would measure a GREY
#             SQUARE; --raw_init random is the random patch the literature
#             reports. 1 epoch: nothing moves, the 500-image evaluation
#             after the loop is the whole point.
#   raw_grey  that grey square, kept: it separates "a patch of this size
#             here perturbs the scene" from "arbitrary high-frequency
#             content does".
#
# No clean row -- that number already exists per arch. To reuse it, it must
# match THIS measurement: 1024x2048, full val split, --inference auto.
# For a csf arm add, with --csf_threshold 0.25 --lr 0.01 (NOT 0.1):
#   emit csf "$OUTROOT/csf" "$PROTO $GEOM --patch_mode universal_csf ..."
configs () {
  emit () { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }
  emit raw      "$OUTROOT/raw" \
       "$PROTO $GEOM --patch_mode raw --raw_init random --lr 0.1 --tag s0.25"
  emit raw_null "$OUTROOT/raw_null" \
       "$PROTO $GEOM --patch_mode raw --raw_init random --lr 0 --epochs 1 --tag s0.25_null"
  emit raw_grey "$OUTROOT/raw_grey" \
       "$PROTO $GEOM --patch_mode raw --lr 0 --epochs 1 --tag s0.25_grey"
}

# ── --list ───────────────────────────────────────────────────────────────────
# NOTHING above this line may write to stdout. Anything that does becomes a
# phantom row in the driver's table, and the driver spends two dev-GPU
# slices submitting a config named after it.
if [ "${1:-}" = "--list" ]; then
    configs | cut -f1,2
    exit 0
fi

WANT="${1:-}"
if [ -z "$WANT" ]; then
    { echo "usage: $0 <config-name> | --list"
      echo
      echo "  A config name is REQUIRED (the other launch scripts here take"
      echo "  none, which is why this is easy to forget). One of:"
      configs | cut -f1 | sed 's/^/      sbatch benchmarkb0.sh /'
      echo
      echo "  Or:  sbatch driverb0.sh"
    } >&2
    exit 2
fi

row="$(configs | awk -F'\t' -v n="$WANT" '$1 == n')"
if [ -z "$row" ]; then
    { echo "FATAL: no config named '$WANT'. Known:"; configs | cut -f1; } >&2
    exit 2
fi
out="$(printf '%s' "$row" | cut -f2)"
args="$(printf '%s' "$row" | cut -f3)"

cd "$ROOT" || exit 1
if [ ! -f scripts/train.py ]; then
    echo "FATAL: '$ROOT' is not the repo root -- no scripts/train.py." >&2
    echo "  sbatch from the repo root, or set REPO_ROOT=/path/to/repo." >&2
    exit 1
fi

echo "=== $ARCH / $WANT ==="
echo "host   : $(hostname)   $(date)"
echo "root   : $ROOT"
echo "out    : $out"

# ── environment, best effort ─────────────────────────────────────────────────
# Redirected to stderr as a group so nothing here can ever reach the --list
# table if this block is later moved above it.
#
# On uc3 all three of these FAIL, in every launch script in this repo, and
# always have: there is no cuda/11.8 module, no ~/miniconda3, and the conda
# launcher in the workspace carries a stale /hkfs shebang from HoreKa. What
# actually puts python on PATH is sbatch --export=ALL inheriting the
# submitting shell's environment. That is fine, and fragile, hence the
# preflight below. CUDA_MODULE is empty by default: `module spider cuda` on
# the login node, then export the right name.
{
  [ -n "${CUDA_MODULE:-}" ] && { module --ignore_cache load "$CUDA_MODULE" \
      || echo "[warn] module '$CUDA_MODULE' did not load"; }
  CONDA_SH="${CONDA_SH:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/etc/profile.d/conda.sh}"
  CONDA_ENV="${CONDA_ENV:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3}"
  if [ -r "$CONDA_SH" ]; then
      # shellcheck disable=SC1090
      source "$CONDA_SH" && conda activate "$CONDA_ENV" \
          || echo "[warn] conda activate failed; using the inherited PATH"
  fi
  export PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib"
} >&2

# Preflight. Everything above is best effort; this is not. A slice that
# cannot import torch, or that got a GPU it cannot see, writes no epoch and
# is indistinguishable to the driver from one that was merely slow. Exit 3
# is the driver's signal to stop the whole run rather than retry.
if ! python - <<'PY'
import sys, torch
print(f"python : {sys.executable}")
print(f"torch  : {torch.__version__}  cuda={torch.cuda.is_available()}")
sys.path.insert(0, ".")
import patchreach  # noqa: F401
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible to torch")
PY
then
    { echo
      echo "FATAL: unusable python environment; nothing was started."
      echo "  On the login node:"
      echo "    which python && python -c 'import torch;print(torch.__version__)'"
      echo "    module spider cuda      # then export CUDA_MODULE=<name>"
      echo "  Simplest fix: activate the env before sbatch -- --export=ALL"
      echo "  carries it into the job."
    } >&2
    exit 3
fi

echo "cmd    : python scripts/train.py $args --out_dir $out"
echo
# shellcheck disable=SC2086
python scripts/train.py $args --out_dir "$out"
rc=$?
echo "slice exited $rc at $(date)"
exit $rc
