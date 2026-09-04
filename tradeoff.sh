#!/bin/bash
#SBATCH -p accelerated
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mem=100000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# ═══════════════════════════════════════════════════════════════════════════
#  THE ATTACK-SUCCESS / VISIBILITY TRADE-OFF, per architecture.
# ═══════════════════════════════════════════════════════════════════════════
#
#     MODE=pilot  sbatch tradeoff.sh     one image per arch, ~1-2 h
#     MODE=figure sbatch tradeoff.sh     a population per arch, the deliverable
#     MODE=pilot  DRY=1 bash tradeoff.sh cost first, run nothing
#
# THE QUESTION. segformer_b0 is the only transformer the CSF-bounded attack
# currently moves; b5 and setr_pup fall over under the frequency constraint
# while both are attackable UNCONSTRAINED. Two explanations fit that equally
# well and they need different fixes:
#
#   the budget binds     tau is simply too small for those two, and the curve
#                        rises to the unconstrained ceiling once tau is raised.
#   something else binds the curve flattens BELOW the ceiling and stays there
#                        however large tau gets, in which case the perceptual
#                        constraint is not what is stopping the attack and the
#                        frequency story is not the explanation.
#
# ONE FIGURE SEPARATES THEM, and only if it carries the ceiling. A tau ladder
# on its own cannot: "weak attack" and "weak because constrained" are the same
# small number. So every architecture gets a --patch_mode raw arm on the SAME
# images, and analysis/tradeoff_panel.py draws it as a horizontal line the
# curve is read against.
#
# THE LADDER RUNS PAST THE POINT OF INVISIBILITY, DELIBERATELY. tau 4, 8, 16
# are plainly visible perturbations and are not operating points -- they are
# there to reach the asymptote, because "the constrained attack approaches the
# unconstrained one" is a claim about a limit and needs the limit measured.
# tradeoff_panel.py shades everything above the knee where the dynamic-range
# fit binds and tau stops controlling anything; nothing in the shading is
# quotable as an achieved operating point.
#
# --csf_enforce realised ON EVERY CSF RUN, and this is not a detail. Under
# 'nominal' the realised visibility sits below the request by a factor nobody
# chose, so the x-axis stops being the perceptual cost and the whole figure
# becomes uninterpretable. Under 'realised' the run holds realised visibility
# AT tau until the range fit binds, which is exactly what makes the axis a
# visibility axis and the knee detectable.
#
# THE LEARNING RATE IS CHOSEN PER ARCHITECTURE, and this is the part that
# cannot be skipped for THIS question. lr_sweep.sh exists because a four-arch
# comparison was once run at one lr and ranked the architectures by whose step
# size happened to suit them. Sweeping tau at b0's lr and concluding "b5 is
# robust to the constrained attack" would repeat that error with tau in the
# caption. The pilot picks lr per arch at equal perceptual cost
# (analysis/pick_lr.py); the figure runs at what the pilot chose.

set -uo pipefail          # NOT -e: one failed cell must not kill the block

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/tradeoff
if [ -n "${SLURM_JOB_ID:-}" ]; then
  exec 1> "slurm/tradeoff/${SLURM_JOB_ID}.out"
  exec 2> "slurm/tradeoff/${SLURM_JOB_ID}.err"
  module --ignore_cache load "cuda/11.8"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
  export PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
fi

: "${CS:=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes}"
: "${MODE:=pilot}"
: "${ARCHS:=segformer_b0 segformer_b5 setr_pup}"
: "${IMAGE:=420}"          # pilot only
: "${N:=50}"               # figure only — images per arm
: "${SAMPLE_SEED:=0}"
: "${DRY:=}"
: "${ROOT:=results/tradeoff}"

# RUN LENGTH IS NOT SWEPT BY DEFAULT. 1000 steps is where the single-image csf
# attack was measured to converge and 400 was measured to truncate it, so the
# steps stage would spend twelve runs per architecture re-deriving a known
# answer. That answer was established ON b0 -- if b5 or setr_pup look
# suspiciously flat, put the stage back before blaming tau:
#
#     STAGES=steps,lr,tau STEPS_GRID="400 1000 2000" MODE=pilot bash tradeoff.sh
#
: "${STEPS:=1000}"
: "${STEPS_GRID:=$STEPS}"
: "${STAGES:=lr,tau}"
# --seeds 1 in the pilot: it is a pilot. Its job is to locate the lr and the
# knee, and the figure below measures the spread over IMAGES, which is the
# spread the claim is about. Raise it if a pilot number is going to be quoted.
: "${SEEDS:=1}"

# The ladder. 0.25 is the default every existing csf number is quoted at; the
# rungs above 1.0 are the asymptote, not operating points.
TAUS="0.05 0.1 0.25 0.5 1 2 4 8 16"
# The unconstrained arm gets its own small lr grid rather than csf's chosen
# one: raw is the sigmoid parameterisation and its parameter is not the same
# object, so carrying one lr across the two measures the step size. The panel
# takes the strongest as the ceiling — an upper bound is what a ceiling is —
# and prints the count in the legend so it cannot be quoted as a single run.
RAW_LRS="0.01 0.05 0.2"

python --version
echo "MODE=$MODE ARCHS='$ARCHS' CS=$CS ROOT=$ROOT"

# ═══════════════════════════════════════════════════════════════════════════
#  PILOT — one image per architecture. Decides the operating point, and gives
#  a first trade-off panel in a couple of hours.
# ═══════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "pilot" ]; then
  for ARCH in $ARCHS; do
    echo ""
    echo "═══════════════ pilot: $ARCH ═══════════════"
    python scripts/sweep_operating_point.py \
        --cityscapes_root "$CS" --arch "$ARCH" --image "$IMAGE" \
        --img_h 512 --img_w 1024 \
        --losses cospgd --stages "$STAGES" --steps_grid "$STEPS_GRID" \
        --csf_param pgd --enforce realised --lr_schedule cosine \
        --tau_grid "$TAUS" --incumbent_tau 0.25 \
        --seeds "$SEEDS" --ceiling 1.0 \
        --name "tradeoff_${ARCH}" --out_root "$ROOT" ${DRY:+--dry_run}

    # THE CEILING, INTO THE SWEEP'S OWN cells/ DIRECTORY, so one --sweep
    # argument carries both arms. tradeoff_panel.py classifies by patch_mode,
    # not by where the directory sits.
    [ -n "$DRY" ] && continue
    for LR in $RAW_LRS; do
      python scripts/overfit.py \
          --arch "$ARCH" --cityscapes_root "$CS" --img_h 512 --img_w 1024 \
          --patch_mode raw --loss_fn cospgd \
          --image "$IMAGE" --steps "$STEPS" --lr "$LR" \
          --lr_schedule cosine --placement center --patch_scale 0.25 \
          --seeds "$SEEDS" --no_diagnostics \
          --out_root "$ROOT/tradeoff_${ARCH}/cells" \
          --tag "ceiling_raw_lr$(echo "$LR" | tr '.' '_')"
    done
  done

  [ -n "$DRY" ] && exit 0
  SWEEPS=""
  for ARCH in $ARCHS; do SWEEPS="$SWEEPS --sweep $ROOT/tradeoff_${ARCH}"; done
  python analysis/tradeoff_panel.py $SWEEPS \
      --archs "$(echo "$ARCHS" | tr ' ' ',')" \
      --metrics any_flip_rate,drop_remote,final_visibility \
      --out figures/tradeoff_pilot.png \
      --title "CSF trade-off, single-image pilot (image $IMAGE)"
  exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE — a population per rung. The deliverable.
# ═══════════════════════════════════════════════════════════════════════════
#
# EVERY ARM OF ONE ARCHITECTURE USES THE SAME SAMPLE. The ceiling is only a
# ceiling for THESE images; a raw arm on a different draw would be compared
# against a different set of scenes, and scene-to-scene spread here is ~10
# mIoU points against effects of 2.
POP="$ROOT/population"
mkdir -p "$POP"

for ARCH in $ARCHS; do
  # The lr the pilot chose for this architecture, at equal perceptual cost.
  # Falls back to 0.01 (universal_csf.sh's value for this parameterisation)
  # with a loud line, rather than silently sweeping tau at a default.
  MAN="$ROOT/tradeoff_${ARCH}/sweep.json"
  LR=$(python - "$MAN" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))["decisions"]["cospgd"]
    print(f"{float(d['lr']['lr']):g}")
except Exception:
    print("")
PY
)
  ST=$(python - "$MAN" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))["decisions"]["cospgd"]
    print(int(d["steps"]["steps"]))
except Exception:
    print("")
PY
)
  if [ -z "$LR" ]; then
    echo "  !! no pilot decision at $MAN — falling back to lr 0.01."
    echo "     The curve is then measured at an lr nothing chose for $ARCH."
    LR=0.01
  fi
  # Empty whenever the steps stage was skipped, which is the DEFAULT above.
  # Not an error: $STEPS is then the declared run length rather than a chosen
  # one, and it is the same one the pilot ran at.
  [ -z "$ST" ] && ST=$STEPS
  echo ""
  echo "═══════════════ figure: $ARCH   lr=$LR steps=$ST ═══════════════"

  for TAU in $TAUS; do
    python scripts/overfit_population.py \
        --arch "$ARCH" --cityscapes_root "$CS" --img_h 512 --img_w 1024 \
        --patch_mode csf --from_image --csf_param pgd \
        --csf_enforce realised --csf_threshold "$TAU" \
        --loss_fn cospgd --placement center --patch_scale 0.25 \
        --lr "$LR" --lr_schedule cosine --steps "$ST" \
        --images random --n_images "$N" --sample_seed "$SAMPLE_SEED" \
        --n_panels 0 --log_every 200 \
        --out_root "$POP" --tag "tau${TAU}"
  done

  for RLR in $RAW_LRS; do
    python scripts/overfit_population.py \
        --arch "$ARCH" --cityscapes_root "$CS" --img_h 512 --img_w 1024 \
        --patch_mode raw --loss_fn cospgd \
        --placement center --patch_scale 0.25 \
        --lr "$RLR" --lr_schedule cosine --steps "$ST" \
        --images random --n_images "$N" --sample_seed "$SAMPLE_SEED" \
        --n_panels 0 --log_every 200 \
        --out_root "$POP" --tag "ceiling-raw-lr${RLR}"
  done
done

python analysis/tradeoff_panel.py "$POP"/*/ \
    --archs "$(echo "$ARCHS" | tr ' ' ',')" \
    --metrics any_flip_rate,drop_remote,final_visibility \
    --out figures/tradeoff_panel.png \
    --title "Attack success against realised visibility, n=$N per point"

# The same data on the honest axis. Above the knee tau is a request nobody
# met, so the tau-axis figure has a flat tail that means nothing; against
# REALISED visibility the x-coordinate is what the observer would actually
# see and the tail collapses to a point.
python analysis/tradeoff_panel.py "$POP"/*/ \
    --archs "$(echo "$ARCHS" | tr ' ' ',')" \
    --metrics any_flip_rate,drop_remote \
    --x realised --out figures/tradeoff_realised.png \
    --title "Attack success vs REALISED visibility (n=$N per point)"

# And normalised, which is the one-number-per-arch version: what fraction of
# each architecture's own unconstrained attack survives the constraint.
python analysis/tradeoff_panel.py "$POP"/*/ \
    --archs "$(echo "$ARCHS" | tr ' ' ',')" \
    --metrics any_flip_rate,drop_remote --normalise \
    --out figures/tradeoff_normalised.png \
    --title "Fraction of the unconstrained attack that survives the CSF budget"

echo ""
echo "Done: $(date)"
