#!/bin/bash
#SBATCH -p dev_gpu_a100_il
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000
#
# T24 — SAC (Segment and Complete) against a trained patch.
#
#     bash   defences/sac/run_sac.sh --fetch   download the segmenter (login node)
#     bash   defences/sac/run_sac.sh --list    the run table, no GPU
#     sbatch defences/sac/run_sac.sh           the three runs below
#
# NOTHING OUTSIDE defences/ IS TOUCHED. Run 1 calls the UNMODIFIED
# scripts/evaluate.py; runs 2 and 3 call defences/evaluate_defended.py, which
# imports the shared plumbing from scripts/_common.py without changing it.
#
# THREE RUNS, AND ALL THREE ARE REQUIRED. A defended attack number alone says
# nothing:
#   1. UNDEFENDED eval — the canonical reference, from the same script every
#      other number in the thesis comes from.
#   2. DEFENDED eval   — the 2x2 plus localisation against the real footprint,
#      which we have for free and the SAC and Jedi papers did not.
#   3. CLEAN COST      — the whole val split, no patch. Every pixel SAC erases
#      is then a false positive by construction.
#
# RUN 2 RE-DERIVES THE UNDEFENDED COLUMN, so its `clean_remote` and
# `drop_remote` must MATCH run 1's. They are computed by different files that
# repeat the same per-image sequence (resolve_placement -> reference -> apply),
# so a disagreement means that sequence has drifted apart and every defended
# number is measured against a different attack. CHECK IT BEFORE READING
# ANYTHING ELSE — it is the one invariant this layout cannot enforce by
# construction.
#
# ── PROTOCOL: Liu et al., CVPR 2022, arXiv:2112.04532 ────────────────────────
# Adopted: their released self-adversarially-trained COCO patch segmenter
# (ckpts/coco_at.pth, base_filter 16), their 0.5 mask threshold, their L1 shape
# completion (init_l1_thresh 0.9, decay 0.7, 15 iterations), and their removal
# rule — ZERO the located region. SAC does not inpaint, so a "recovered" mIoU is
# recovered despite a hard black occlusion, not by restoring the scene.
#
# FOUR DEVIATIONS, all deliberate, all to be stated in the write-up:
#   1. DOMAIN. Their segmenter was trained on COCO patches for object
#      DETECTION. We run it on Cityscapes semantic segmentation, on
#      contrast-budgeted residuals it has never seen. This is the
#      OFF-THE-SHELF arm. Re-running their self-adversarial training on our own
#      patches is a separate, stronger arm — only worth it if the off-the-shelf
#      segmenter already catches something here.
#   2. SQUARE SIZES = [100 75 50 25], which is what their released code
#      actually executes: PatchDetector stores self.square_sizes and then calls
#      ShapeCompletionL1 without passing it (patch_detector.py:194), so the
#      [125 100 75 50 25] its own eval script builds never arrives. Verified
#      against the repo 2026-10-01. Our patch side is 128px, so every candidate
#      square is SMALLER than the patch.
#   3. n_patch = 1. Patch.apply() composites exactly one footprint, so their
#      k-means multi-patch path is refused rather than silently ignored.
#   4. NO ADAPTIVE ATTACK. The patch was optimised against the UNDEFENDED
#      model, so this is the oblivious threat model. SAC ships a
#      straight-through estimator for the adaptive case and we do not use it.
#      An adaptive row is a DIFFERENT THREAT MODEL, not a better version of
#      this one.
#
# NEVER QUOTE THEIR ~90% DETECTION FIGURES BESIDE THESE NUMBERS. Different
# task, different dataset, different patches, unadapted segmenter.
# ─────────────────────────────────────────────────────────────────────────────

set -u

# NOT dirname "$BASH_SOURCE": sbatch runs a copy out of the spool directory.
# This script lives two levels down, so the by-hand fallback climbs twice.
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

MA="${MA:-/pfs/work9/workspace/scratch/ma_nilbecke-thesis}"
CS="${CS:-$MA/data/cityscapes}"
SAC_CKPT="${SAC_CKPT:-$MA/checkpoints/sac_coco_at.pth}"
SAC_URL="https://raw.githubusercontent.com/joellliu/SegmentAndComplete/main/ckpts/coco_at.pth"

ARCH="${ARCH:-segformer_b0}"
RES="--img_h 512 --img_w 1024"
TAG=T24_sac
# POINT THIS AT THE ATTACK RUN YOU ARE DEFENDING AGAINST.
CKPT="${CKPT:-$MA/master-thesis/results/overfit/REPLACE_ME/best.pt}"

# ── --fetch: one-time weight download (4.4 MB, login node) ───────────────────
if [ "${1:-}" = "--fetch" ]; then
    mkdir -p "$(dirname "$SAC_CKPT")"
    echo "fetching the SAC patch segmenter (4.4 MB)"
    echo "  from $SAC_URL"
    echo "  to   $SAC_CKPT"
    curl -fL -o "$SAC_CKPT" "$SAC_URL" || { echo "download FAILED"; exit 1; }
    ls -l "$SAC_CKPT"
    echo "expected size: 4385199 bytes"
    exit 0
fi

if [ "${1:-}" = "--list" ]; then
    echo "root     : $ROOT"
    echo "arch     : $ARCH"
    echo "res      : $RES"
    echo "patch    : $CKPT"
    echo "segmenter: $SAC_CKPT"
    echo "tag      : $TAG"
    echo
    echo "1. undefended eval  (scripts/evaluate.py, UNMODIFIED)"
    echo "                    -> results/eval/*_${TAG}_undef/"
    echo "2. defended   eval  (defences/evaluate_defended.py)"
    echo "                    -> results/defence/*_sac_${TAG}/"
    echo "3. clean cost       (same script, no --checkpoint, --images all)"
    echo "                    -> results/defence/clean_${ARCH}_*_sac_${TAG}/"
    echo
    echo "then: run 2's undefended drop_remote MUST equal run 1's."
    exit 0
fi

mkdir -p "$ROOT/slurm/defence"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    exec 1> "$ROOT/slurm/defence/${SLURM_JOB_ID}_sac_${ARCH}.out"
    exec 2> "$ROOT/slurm/defence/${SLURM_JOB_ID}_sac_${ARCH}.err"
fi

echo "Running on $(hostname) at $(date)"
module --ignore_cache load "cuda/11.8"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$MA/miniconda3/envs/thesis_backup3"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MA/miniconda3/lib"
cd "$ROOT" || exit 1
python -c "import sys; print('python', sys.executable)"

if [ ! -f "$SAC_CKPT" ]; then
    echo "the SAC segmenter is not at $SAC_CKPT"
    echo "run: bash defences/sac/run_sac.sh --fetch"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "set CKPT to the attack run you are defending against (got $CKPT)"
    exit 1
fi

BASE="--arch $ARCH --cityscapes_root $CS $RES"
SAC="--defence sac --sac_ckpt $SAC_CKPT"

# ── 1. undefended reference, from the untouched script ───────────────────────
echo; echo "=== 1/3  undefended eval (scripts/evaluate.py) ==="
python scripts/evaluate.py --checkpoint "$CKPT" $BASE --from_image \
    --images fixed10 --diagnostics_on -1 --no_panels --tag "${TAG}_undef"

# ── 2. defended: the 2x2 + localisation ──────────────────────────────────────
echo; echo "=== 2/3  defended eval (+ localisation vs the true footprint) ==="
python defences/evaluate_defended.py --checkpoint "$CKPT" $BASE --from_image \
    $SAC --images fixed10 --figures --tag "$TAG"

# ── 3. the clean-image price, over the whole val split ───────────────────────
echo; echo "=== 3/3  clean cost (no patch: every erased pixel is a FP) ==="
python defences/evaluate_defended.py $BASE $SAC --images all --tag "$TAG"

echo; echo "done $(date)"
echo "FIRST: check run 2's undefended drop_remote against run 1's. They are"
echo "computed by two files repeating one sequence; if they disagree, stop."
echo "THEN: run 3's clean mIoU vs your undefended clean_baseline.py for $ARCH."
echo "That difference is what every robustness number in run 2 is paid for with."
