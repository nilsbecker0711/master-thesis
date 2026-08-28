#!/bin/bash
#SBATCH -p gpu_a100_il
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=400000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# ═══════════════════════════════════════════════════════════════════════════
#  GRID SWEEP:  patch_mode  x  tau  x  learning rate
# ═══════════════════════════════════════════════════════════════════════════
#
# 50 population runs (40 csf + 10 raw), ~6 h at the defaults below.
#
# Nothing aborts the sweep: a failed run is logged and skipped, so one bad
# combination does not cost the night. Each run writes its own directory under
# $ROOT, and the summary at the end reads them all back.
#
# WHY tau IS SKIPPED FOR raw: --csf_threshold does nothing in raw mode, so
# iterating it there would run the same configuration four times. raw is run
# once, at the nominal tau, purely to give the lr axis an unconstrained
# reference point.
#
# READING THE RESULT — do not just take the largest drop_remote. For csf, a
# bigger lr drives the residual harder against the [0,1] boundary, the final
# clamp injects harmonics, and REALISED visibility climbs above the tau you
# asked for. Selecting on drop_remote alone rewards whichever lr breaks the
# perceptual constraint hardest. analysis/pick_lr.py applies the right rule:
# best drop AMONG runs whose realised visibility stays under the ceiling.

set -uo pipefail          # NOT -e: one failed run must not kill the sweep

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/overnight
exec 1> "slurm/overnight/${SLURM_JOB_ID:-local}.out"
exec 2> "slurm/overnight/${SLURM_JOB_ID:-local}.err"
module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib

# ── knobs ────────────────────────────────────────────────────────────────────
: "${CS:=/hkfs/work/workspace/scratch/ma_nilbecke-thesis/data/cityscapes}"
: "${N:=50}"                 # images per run
: "${STEPS:=600}"            # per-image optimisation steps
: "${SEED:=0}"
: "${VIS_CEILING:=1.0}"      # JND — 1.0 is the detection threshold
: "${ROOT:=results/sweep}"

MODES=(raw csf)
TAUS=(0.05 0.1 0.25 0.5)
LRS=(0.01 0.05 0.1 0.2 0.5)
RAW_TAU=0.25                 # the single tau raw is run at

BASE="--arch segformer --cityscapes_root $CS --img_h 512 --img_w 1024"
GRADCAM="--placement gradcam --placement_margin 64 --cam_target pred"
mkdir -p "$ROOT/logs"

python --version
echo "N=$N STEPS=$STEPS SEED=$SEED ROOT=$ROOT"

n_ok=0; n_fail=0; started=$SECONDS

# ═══════════════════════════════════════════════════════════════════════════
#  THE SWEEP
# ═══════════════════════════════════════════════════════════════════════════
for MODE in "${MODES[@]}"; do
  for TAU in "${TAUS[@]}"; do

    # tau is meaningless for raw — run it once instead of four times
    if [ "$MODE" = "raw" ] && [ "$TAU" != "$RAW_TAU" ]; then
      continue
    fi

    # csf needs the base to be the region the patch covers, or it is a visible
    # grey square with an invisible texture on it
    EXTRA=""
    if [ "$MODE" = "csf" ]; then EXTRA="--from_image"; fi

    for LR in "${LRS[@]}"; do
      TAG="${MODE}_tau${TAU}_lr${LR}"
      LOG="$ROOT/logs/${TAG}.log"
      t0=$SECONDS
      echo ""
      echo "──── $TAG   ($(date +%H:%M:%S))"

      if python scripts/overfit_population.py $BASE $GRADCAM $EXTRA \
            --patch_mode "$MODE" --loss_fn cospgd --csf_threshold "$TAU" \
            --images random --n_images "$N" --sample_seed "$SEED" \
            --steps "$STEPS" --log_every 50 --lr "$LR" \
            --n_panels 0 \
            --out_root "$ROOT" --tag "$TAG" > "$LOG" 2>&1; then
        printf '     ok    %5ds   %s\n' $(( SECONDS - t0 )) \
          "$(grep -m1 'drop   REMOTE\|mean  ' "$LOG" | tr -s ' ' || true)"
        n_ok=$(( n_ok + 1 ))
      else
        printf '     FAIL  %5ds   -> %s\n' $(( SECONDS - t0 )) "$LOG"
        tail -n 12 "$LOG" | sed 's/^/       | /'
        n_fail=$(( n_fail + 1 ))
      fi
    done
  done
done

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════"
printf ' %d ok, %d failed, %dh %dm wall clock\n' "$n_ok" "$n_fail" \
  $(( (SECONDS-started)/3600 )) $(( ((SECONDS-started)%3600)/60 ))
echo "════════════════════════════════════════════════════"

# Per-tau lr table. pick_lr prints the grid to stderr and the chosen lr to
# stdout, and refuses to choose when every lr broke the visibility ceiling —
# in that case the fix is a lower tau, not a different lr.
for TAU in "${TAUS[@]}"; do
  echo ""
  echo "──── csf, tau=$TAU"
  python analysis/pick_lr.py "$ROOT"/csf_tau${TAU}_lr* \
      --ceiling "$VIS_CEILING" > /dev/null
done
echo ""
echo "──── raw (unconstrained: no visibility ceiling can bind)"
python analysis/pick_lr.py "$ROOT"/raw_tau${RAW_TAU}_lr* > /dev/null

echo ""
echo "results under $ROOT/    logs under $ROOT/logs/"
