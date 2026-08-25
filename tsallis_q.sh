#!/bin/bash
#SBATCH -p dev_gpu_a100_il     # DEV partition, 10 min cap -> MODE=smoke ONLY.
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 00:10:00            # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=40000            # Memory request (adjust as needed)
#SBATCH --gres=gpu:1           # Request 1 GPU (adjust if you need more)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)
##SBATCH --output=slurm/attack_%J_%j_%a.out
##SBATCH --error=slurm/attack_%J_%j_%a.err

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — choose q for the Tsallis attack objective
# ═══════════════════════════════════════════════════════════════════════════
#
# q is a NEW hyperparameter against THIS model and THIS data. The paper's
# validation-selected linear -2 -> 1 is a prior, not a result that transfers:
# it was chosen on their model by their sweep, and inheriting it unchecked
# makes every later Tsallis number a claim about their tuning rather than ours.
#
# SIX ARMS ON ONE SHARED SAMPLE
#
#   q = 1        the CE limit. |1-q| < 1e-6 takes the -log p_y fallback, so
#                this must land on --loss_fn ce within noise. If it does not,
#                the integration is wrong and nothing below this line counts.
#   q = 0        gradient weight p^(1-q) peaks at p* = (1-q)/(2-q) = 0.50
#   q = -1                                                        p* = 0.67
#   q = -2                                                        p* = 0.75
#   q = -3                                                        p* = 0.80
#                i.e. MORE NEGATIVE q aims the attack at pixels the model is
#                still CONFIDENT about, rather than at ones already half-fooled.
#   linear -2 -> 1   the paper's schedule, swept across the run.
#
# WHY internimage AND NOT segformer. configs/experiments/E_realism.yaml records
# SegFormer saturating at ~99.9% flip with a plain raw square, which is why the
# realism ladder runs on InternImage: with no headroom every rung reads "no
# effect". A loss-function sweep has exactly the same problem -- if every arm
# already flips everything, no q can distinguish itself and the sweep returns a
# flat line that means "saturated", not "q does not matter". InternImage at
# ~21% has room in both directions. Override with ARCH=segformer to run the
# saturation control deliberately, and label it as such.
#
# RUN LENGTH IS PART OF THE SCHEDULE, NOT A BUDGET. A linear sweep over 300
# steps and one over 1000 spend different fractions of the run at each q, so
# they are different attacks. STEPS is ONE variable here and every arm shares
# it by construction. Do not vary it between arms and then compare them.
#
# EVERY ARM SHARES --sample_seed. analysis/compare_populations.py pairs arms BY
# IMAGE INDEX, and that pairing is the only thing that resolves a small effect
# under the ~10-point scene-to-scene spread. A different seed per arm silently
# degrades the test to whatever images happen to overlap.
#
# --n_panels 0 EVERYWHERE. This phase picks a NUMBER; panels and the per-image
# diagnostic suite are the expensive part of the script and nobody looks at a
# hyperparameter sweep's pictures. The pooled aggregate stays ON -- it costs a
# few tensor reductions per image against a full optimisation, and it is what
# the distribution figure is built from.
#
# RESUMABLE BY CONSTRUCTION. Each arm gets an explicit --out_dir and is
# launched with --resume, which is a no-op on a fresh directory and skips
# already-completed images on a rerun. On a 10-minute dev slot this job WILL be
# killed part way through: resubmit the identical command and it continues.
# Completed arms still pay their model load on the resubmit (~1 min each), they
# just do no optimisation.

echo "Running on $(hostname)"
echo "Date: $(date)"

module --ignore_cache load "cuda/11.8"

# initialize YOUR conda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
# NOTE: overfit.sh has this as `=$/pfs/...` -- `$/` is not a valid expansion,
# so bash leaves a literal `$` AND the assignment clobbers whatever the module
# load just put on the path. This is the form validation.sh uses.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

echo "Python path:"
which python
python -c "import sys; print(sys.executable)"

echo "Python version:"
python --version

# ── MODE ────────────────────────────────────────────────────────────────────
# smoke : fits the 10-minute dev slot in the header above. Proves all six arms
#         launch, parse, optimise and write. The NUMBERS ARE MEANINGLESS at
#         n=2 / 20 steps -- this checks plumbing, not attacks.
# full  : the real sweep. SWAP THE HEADER FIRST, it will not fit in 10 minutes:
#             #SBATCH -p accelerated
#             #SBATCH -t 02:00:00
#         then:   MODE=full sbatch tsallis_q.sh
MODE="${MODE:-smoke}"
ARCH="${ARCH:-internimage}"

if [ "$MODE" = "full" ]; then
    N=20
    STEPS=300
else
    N=2
    STEPS=20
fi
echo "[mode] $MODE — arch=$ARCH n=$N steps=$STEPS"

export RES="--img_h 512 --img_w 1024"
export BASE="--arch $ARCH --cityscapes_root $CS $RES"
export SAMPLE="--images random --n_images $N --sample_seed 0"
# --patch_mode raw: the unconstrained ceiling. Pick q here, THEN carry the
# winner into the constrained (csf) arms -- not the other way round, or the
# choice of q is confounded with the perceptual budget.
export RUN="--patch_mode raw --steps $STEPS --lr_schedule cosine --n_panels 0"
export ROOT="results/population/tsallis_q_${MODE}_${ARCH}"

mkdir -p "$ROOT"

# ── the CE reference arm ────────────────────────────────────────────────────
# Run FIRST so a broken integration costs one arm rather than six. --loss_fn ce
# is the pre-existing objective, untouched by the Tsallis change; the q=1 arm
# below has to reproduce it.
python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --loss_fn ce \
    --out_dir "$ROOT/ce-reference" --resume

# ── the const-q sweep ───────────────────────────────────────────────────────
# Tag names avoid a bare leading '-' by spelling the sign: qc-p1, qc-0, qc-m1.
for Q in 1 0 -1 -2 -3; do
    case "$Q" in
        1)  NAME="qc-p1" ;;
        0)  NAME="qc-0"  ;;
        *)  NAME="qc-m${Q#-}" ;;
    esac
    echo ""
    echo "=================================================================="
    echo "  ARM $NAME   --tsallis_q $Q"
    echo "=================================================================="
    python scripts/overfit_population.py $BASE $SAMPLE $RUN \
        --loss_fn tsallis --tsallis_q "$Q" \
        --out_dir "$ROOT/$NAME" --resume
done

# ── the scheduled arm ───────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  ARM q-linear   --tsallis_schedule linear -2 -> 1 over $STEPS steps"
echo "=================================================================="
python scripts/overfit_population.py $BASE $SAMPLE $RUN \
    --loss_fn tsallis --tsallis_schedule linear \
    --tsallis_q_start -2 --tsallis_q_end 1 \
    --out_dir "$ROOT/q-linear" --resume

# ── paired comparison ───────────────────────────────────────────────────────
# --baseline 0 makes ce-reference the A arm of every pair, so the q=1 row is
# the integration check and the rest are the result.
#
# describe_arm() prints q for every Tsallis arm and warn_if_multiple_axes_differ()
# tracks the tsallis_* keys, so a pair that differs in q AND in something else
# raises the usual one-axis warning rather than passing silently.
echo ""
python analysis/compare_populations.py \
    "$ROOT/ce-reference" \
    "$ROOT/qc-p1" "$ROOT/qc-0" "$ROOT/qc-m1" "$ROOT/qc-m2" "$ROOT/qc-m3" \
    "$ROOT/q-linear" \
    --baseline 0 --key drop_remote --out_dir "$ROOT/compare"

echo ""
echo "  -> $ROOT/"
echo "  Read compare/comparison.json. The q=1 row must be ~0 against"
echo "  ce-reference; if it is not, stop and fix the integration."
