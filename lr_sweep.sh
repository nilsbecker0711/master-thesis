#!/bin/bash
#SBATCH -p accelerated
#SBATCH -n 1
#SBATCH -t 02:00:00
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
# ── THE LR RANGE IS CENTRED ON 0.01, NOT 0.2 ────────────────────────────────
# --csf_param pgd (the new default) changes what the parameter IS. Under the
# old squash the parameter was a DIRECTION -- the render renormalised to
# exactly tau, so scaling the parameter 1000x moved the patch by 7.5% -- and
# lr set how fast that direction rotated. Under pgd the parameter IS the
# residual, in pixel units, with rms ~0.027 at tau 0.25. An Adam step of 0.2
# per coordinate is then ~7x the entire parameter every step: it overshoots the
# constraint ball and the projection throws the overshoot away.
#
# 0.01 is what universal_csf.sh already uses for this exact parameterisation.
# The sweep brackets it by a decade either side.
#
# ── SMOKE FIRST ─────────────────────────────────────────────────────────────
#     MODE=smoke sbatch lr_sweep.sh     # 8 short runs, checks plumbing only
#     MODE=full  sbatch lr_sweep.sh     # the real sweep, ~100 min

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
    STEPS=1000; LRS="0.001 0.003 0.01 0.03 0.1"
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
echo "=============================================================="
