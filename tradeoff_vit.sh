#!/bin/bash
#SBATCH -p accelerated
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem=400000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2
#SBATCH --output=slurm/tradeoff/vit_%A_%a.out
#SBATCH --error=slurm/tradeoff/vit_%A_%a.err

# ═══════════════════════════════════════════════════════════════════════════
#  THE TRADE-OFF BLOCK for the attention bracket: b0, b5, setr_pup.
#  ONE JOB ARRAY, ONE FIGURE.
# ═══════════════════════════════════════════════════════════════════════════
#
#     sbatch tradeoff_smoke.sh          # five minutes, first
#     sbatch tradeoff_vit.sh            # the runs, three tasks in parallel
#     bash   tradeoff_vit.sh plot       # the panel, on the login node
#
# ONE PLOT, NOT ONE PER ARCHITECTURE. analysis/tradeoff_panel.py takes the
# architectures as COLUMNS of a single figure — that is what it is for, and a
# per-architecture figure would defeat the comparison the block exists to
# make. Three jobs is a SCHEDULING decision, not a plotting one: the panel
# reads finished runs off disk, so the runs can be split across as many jobs,
# nodes or days as the queue likes and the figure is assembled afterwards from
# whatever is there. Splitting also means setr_pup's walltime does not have to
# cover b0's, and one architecture that OOMs does not cost the other two.
#
# WHY AN ARRAY RATHER THAN THREE FILES: the three tasks differ in exactly one
# token, the architecture. Three near-identical scripts is three places for
# the tau grid, the enforcement mode or the placement to drift apart, and a
# curve whose columns were measured under quietly different settings is worse
# than no curve — it looks comparable.
#
# WALLTIME. Set for the slowest member at the largest variant; b0 will finish
# in a fraction of it. The block's size is a CELL COUNT, not an hour count:
#
#     STAGES=tau + INCUMBENT_LR    9 csf + 3 raw = 12 runs of $STEPS per arch
#     STAGES=lr,tau (default)     16 csf + 3 raw = 19 runs of $STEPS per arch
#
# Multiply by the per-run time the architecture actually costs, which spans
# roughly an order of magnitude across this lineup. DRY=1 prints the plan.

set -uo pipefail

module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

mkdir -p slurm/tradeoff figures

# registry.py's "global" bracket — efficient or full self-attention, large ERF.
# The other three entries (deeplab, unet, internimage) are the no-attention
# bracket and belong to a different comparison; add them here only if the
# figure is meant to be about the bracket rather than about the ViTs.
ARCHS=(segformer_b0 segformer_b5 setr_pup)

: "${ROOT:=results/tradeoff}"
: "${IMAGE:=420}"
: "${STEPS:=1000}"
: "${SEEDS:=1}"
: "${STAGES:=lr,tau}"
: "${TAUS:=0.05 0.1 0.25 0.5 1 2 4 8 16}"
: "${RAW_LRS:=0.01 0.05 0.2}"
: "${DRY:=}"
# Per-architecture learning rate, if lr_sweep.sh already chose one. Indexed
# like ARCHS. Empty entries fall through to the lr stage. Skipping that stage
# is 8 of the 19 runs, so fill these in when you can:
#
#     python analysis/pick_lr.py results/lr_sweep/setr_pup_* --ceiling 0.25
#
# and then run with STAGES=tau.
LRS=("" "" "")

# ── the panel. No GPU, seconds. `bash tradeoff_vit.sh plot` on the login node
#    once the array has drained, or submit it with
#    --dependency=afterok:<arrayjobid>.
if [ "${1:-}" = "plot" ]; then
  SWEEPS=""
  for A in "${ARCHS[@]}"; do SWEEPS="$SWEEPS --sweep $ROOT/tradeoff_${A}"; done
  JOIN=$(IFS=,; echo "${ARCHS[*]}")

  python analysis/tradeoff_panel.py $SWEEPS --archs "$JOIN" \
      --metrics any_flip_rate,drop_remote,final_visibility \
      --out figures/tradeoff_vit.png \
      --title "CSF trade-off across the attention bracket (image $IMAGE)"

  # The same runs against REALISED visibility. Above the knee tau is a
  # request nobody met, so the tau-axis figure has a tail that means nothing;
  # here the x-coordinate is what an observer would actually see and that
  # tail collapses onto a point.
  python analysis/tradeoff_panel.py $SWEEPS --archs "$JOIN" \
      --metrics any_flip_rate,drop_remote --x realised \
      --out figures/tradeoff_vit_realised.png \
      --title "Attack success vs realised visibility, attention bracket"

  # And as a fraction of each architecture's OWN unconstrained ceiling: the
  # one-number-per-arch version of the whole block.
  python analysis/tradeoff_panel.py $SWEEPS --archs "$JOIN" \
      --metrics any_flip_rate,drop_remote --normalise \
      --out figures/tradeoff_vit_normalised.png \
      --title "Fraction of the unconstrained attack surviving the CSF budget"
  exit 0
fi

# ── one architecture per array task ─────────────────────────────────────────
IDX=${SLURM_ARRAY_TASK_ID:-0}
ARCH=${ARCHS[$IDX]}
LR=${LRS[$IDX]}

echo "Running on $(hostname)"
echo "Date: $(date)"
echo "arch=$ARCH  stages=$STAGES  steps=$STEPS  lr=${LR:-<from lr stage>}"
which python
python --version

# THE TAU LADDER. --enforce realised is what makes the x-axis a visibility
# axis at all: it holds realised visibility AT tau until the dynamic-range fit
# binds. Under 'nominal' realised sits below the request by a factor nobody
# chose and the panel's knee rule has to fall back to a saturation test.
python scripts/sweep_operating_point.py \
    --cityscapes_root "$CS" --arch "$ARCH" --image "$IMAGE" \
    --img_h 512 --img_w 1024 \
    --losses cospgd --stages "$STAGES" --steps_grid "$STEPS" \
    --tau_grid "$TAUS" --incumbent_tau 0.25 \
    --csf_param pgd --enforce realised --lr_schedule cosine \
    --seeds "$SEEDS" --ceiling 1.0 ${LR:+--incumbent_lr "$LR"} \
    --name "tradeoff_${ARCH}" --out_root "$ROOT" ${DRY:+--dry_run}

[ -n "$DRY" ] && exit 0

# THE UNCONSTRAINED CEILING, into the sweep's own cells/ so one --sweep
# argument carries both arms. Without it the panel can show that the attack is
# weak on b5 and setr_pup but not whether the CONSTRAINT is what weakened it —
# and that is the entire question this block was written for.
#
# Its own small lr grid: raw is the sigmoid parameterisation and its parameter
# is not the same object as the csf residual, so carrying one lr across the two
# measures the step size. The panel takes the strongest as the ceiling, which
# is what an upper bound is, and prints the count in the legend.
for RLR in $RAW_LRS; do
  python scripts/overfit.py \
      --arch "$ARCH" --cityscapes_root "$CS" --img_h 512 --img_w 1024 \
      --patch_mode raw --loss_fn cospgd \
      --image "$IMAGE" --steps "$STEPS" --lr "$RLR" --lr_schedule cosine \
      --placement center --patch_scale 0.25 \
      --seeds "$SEEDS" --no_diagnostics \
      --out_root "$ROOT/tradeoff_${ARCH}/cells" \
      --tag "ceiling_raw_lr$(echo "$RLR" | tr '.' '_')"
done

echo "Done: $(date)"
echo "When all three tasks have finished:  bash tradeoff_vit.sh plot"
