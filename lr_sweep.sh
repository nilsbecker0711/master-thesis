#!/bin/bash
#SBATCH -p accelerated
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
# The four-architecture overfit comparison was run at ONE lr (0.2) for every
# model, and two of the four peaked early and then decayed:
#
#     arch                  best      at step     final
#     segformer_b0        +54.54         1000    +54.54
#     segformer_b5        +27.45         1000    +27.45
#     deeplabv3plus_r101  +11.91          120    +10.04
#     setr_pup            +14.54           40     +4.33
#
# Ranking on final puts deeplab above setr; ranking on best swaps them. A
# ranking that depends on which checkpoint you read is not a measurement of
# architecture, it is a measurement of the optimiser. This sweep separates them.
#
# ── THE RANGE SPANS 0.001 TO 1.0, AND WHY IT IS NOT CENTRED ON 0.2 ──────────
# The 0.2 that produced the +54.54 on segformer_b0 was measured under the
# SQUASH, where the parameter is a DIRECTION: the render renormalises to
# exactly tau, so the parameter's magnitude is meaningless and lr sets how fast
# that direction rotates. Measured here at tau 0.25, size 128:
#
#     squash : param rms 0.9939  ->  an Adam step of 0.2 is 0.2x the
#                                    per-coordinate magnitude. A 20% step.
#     pgd    : param rms 0.0272  ->  an Adam step of 0.2 is 7.4x it.
#
# So the SCALE-EQUIVALENT of squash's 0.2 under pgd is 0.2 * 0.0272/0.9939
# ~= 0.0055, and 0.01 is what universal_csf.sh already uses for this same
# parameterisation. That is why the range is centred where it is.
#
# BUT THE RANGE STILL RUNS TO 1.0, DELIBERATELY. "lr so large that every step
# overshoots the constraint set and is clipped back" is not obviously wrong for
# an adversarial attack -- it is bang-bang descent against the boundary, which
# is what classical PGD attacks do with sign(grad) steps at the epsilon ball,
# and they work. The scale argument says the equivalent lr is ~0.005; it does
# not say the loss landscape in residual space has its optimum in the same
# place as in direction space. That is an experiment, not a deduction, so the
# sweep is built so it cannot foreclose the answer.
#
# IF THE WINNER LANDS AT EITHER EDGE (0.001 or 1.0) THE SWEEP IS TRUNCATED and
# the number is an artifact of where the grid stopped -- extend it and re-run.
# The same failure as reading a 400-step run that had not converged.
#
# ── SMOKE FIRST ─────────────────────────────────────────────────────────────
#     MODE=smoke sbatch lr_sweep.sh     # 8 short runs, checks plumbing only
#     MODE=full  sbatch lr_sweep.sh     # the real sweep: 4 archs x 7 lrs = 28
#                                       # runs. --no_diagnostics drops the
#                                       # expensive part, so ~2-3 min each.

echo "Running on $(hostname)"
echo "Date: $(date)"

module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

echo "Python path:"; which python
echo "Python version:"; python --version

MODE="${MODE:-smoke}"
IMAGE="${IMAGE:-42}"

if [ "$MODE" = "full" ]; then
    STEPS=1000; LRS="0.001 0.003 0.01 0.03 0.1 0.3 1.0"
    ARCHS="segformer_b0 segformer_b5 deeplabv3plus_r101 setr_pup"
else
    STEPS=40;   LRS="0.01 0.1"
    ARCHS="segformer_b0 setr_pup"
fi
echo "[sweep] mode=$MODE image=$IMAGE steps=$STEPS"
echo "[sweep] lrs: $LRS"
echo "[sweep] archs: $ARCHS"

RES="--img_h 512 --img_w 1024"

for ARCH in $ARCHS; do
  for LR in $LRS; do
    # The tag carries the lr, so no two cells share an output directory. The
    # earlier runs all wrote to <arch>_csf_cospgd_img<N>/ and a re-run would
    # have silently overwritten the one before it.
    TAG="lrsweep_$(echo "$LR" | tr '.' '_')"
    echo ""
    echo "=============================================================="
    echo "  arch=$ARCH  lr=$LR  ->  $TAG"
    echo "=============================================================="
    python scripts/overfit.py \
        --arch "$ARCH" --cityscapes_root "$CS" $RES \
        --patch_mode csf --csf_param pgd --from_image \
        --loss_fn cospgd \
        --image "$IMAGE" --steps "$STEPS" --lr "$LR" \
        --lr_schedule cosine \
        --csf_threshold 0.25 --csf_enforce realised \
        --seeds 1 --log_every 50 \
        --no_diagnostics \
        --out_root results/lr_sweep --tag "$TAG"
  done
done

echo ""
echo "=============================================================="
echo "  DONE. Pick the lr PER ARCH with the existing selector, which"
echo "  chooses at EQUAL PERCEPTUAL COST rather than argmax drop --"
echo "  otherwise you select whichever lr breaks tau hardest:"
echo ""
for ARCH in $ARCHS; do
  echo "    python analysis/pick_lr.py results/lr_sweep/${ARCH}_*  \\"
  echo "        --metric best_drop_remote --ceiling 0.25"
done
echo ""
echo "  If a chosen lr is 0.001 or 1.0 -- an EDGE of the grid -- the sweep is"
echo "  truncated and that number is where the grid stopped, not an optimum."
echo "  Extend LRS in this script and re-run that arch."
echo "=============================================================="
