#!/bin/bash
#SBATCH -p dev_gpu_a100_il
#SBATCH -n 1
#SBATCH -t 00:35:00
#SBATCH --mem=60000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# ═══════════════════════════════════════════════════════════════════════════
#  THE OPTIMISER ABLATION — is declining to call this attack PGD justified?
#  ONE JOB, ~25 MINUTES, no human in the middle.
# ═══════════════════════════════════════════════════════════════════════════
#
# WHY THIS RUN EXISTS
# -------------------
# Two docstrings carry the reason this work is not described as a PGD attack,
# and neither had ever been measured:
#
#   losses/adversarial.py  "a patch is not an epsilon-ball perturbation, so we
#                           use Adam"                        — the STEP RULE
#   patch/spec.py          "a hard clamp zeroes the gradient for any pixel
#                           sitting on the bound, which is exactly where an
#                           adversarial patch wants to live"
#                                                            — the PROJECTION
#
# Adam has been the optimiser since commit 320c5cb; nothing here had ever run
# sign-SGD, and the saturation measurement in residual() tested forward-clamp
# against post-step projection on the SPECTRUM, under Adam — a different axis.
# The claim rests on two arguments and zero results. This turns them into
# results, or retracts them.
#
# THE DESIGN IS A 2x2, because the two claims are independent.
#
#            |  pixel_param=sigmoid        pixel_param=direct
#   ---------+----------------------------------------------------
#   adam     |  A  the current attack      B  projection only
#   sign     |  C  step rule only          D  PGD
#
#   A is every raw-mode number in this thesis. D is as close to PGD as a patch
#   threat model reaches — a sign step plus a projection onto the feasible
#   pixel box — the only missing ingredient being an epsilon-ball, which does
#   not exist here. B and C are what make this a 2x2 rather than A vs D: if D
#   differs from A, B and C say WHICH claim did the work.
#
# LOSS: ce, not cospgd. The loss axis is a separate question and mixing them
# would need a 2x2x2. ce is also the arm whose NAME is under dispute.
#
# ═══════════════════════════════════════════════════════════════════════════
#  WHERE THE 12 HOURS WENT, and what was cut to get to 25 minutes
# ═══════════════════════════════════════════════════════════════════════════
#
# The first version was 216,000 attack steps at 512x1024. Four cuts, in
# descending order of how much they buy and ascending order of what they cost:
#
#  1. PANELS AND POOLED DIAGNOSTICS OFF (--n_panels 0 --no_aggregate).
#     population.py calls the panel suite "the expensive part" — a Grad-CAM, an
#     ERF probe and a dozen figures per panel image, 20 runs x 3 panels in the
#     old design. This ablation reads ONE number per image, drop_remote, which
#     lives in pop.records either way. Scientific cost: zero.
#
#  2. 512x1024 -> 256x512, with --patch_size 64 so the parameter resolution
#     still matches the footprint instead of over-resolving it 4x. Buys ~4x.
#     COST, AND IT MUST BE DECLARED: the absolute drops are not comparable to
#     the 512x1024 tables. This is a WITHIN-ablation contrast on a shared
#     sample, so it is internally valid and answers "does the step rule
#     matter", but it cannot be quoted beside the headline numbers.
#
#  3. n=100 -> n=40 in phase 2. compare_populations.py pairs BY IMAGE INDEX,
#     so power comes from the paired sd, not from the per-image sigma~10 that
#     set n=100 for the unpaired arms in population.sh.
#
#  4. The ladder shrunk to 3 rungs x 4 images x 120 steps, and its lr pick is
#     now automatic (analysis/pick_ladder_lr.py) instead of a human reading
#     summary.json between two sbatch calls. That round-trip, not the compute,
#     is what made the old version an overnight job.
#
# WHAT WAS NOT CUT, and why you should not cut it either
# ------------------------------------------------------
#  * STEPS, still 300 in phase 2. Truncation systematically favours the
#    faster-converging optimiser, which is Adam, which is the incumbent. A
#    short run would hand the ablation the answer it is supposed to test.
#    Cosine annealing over T_max=steps mitigates this (both arms fully anneal
#    into whatever budget they get) but does not remove it.
#  * THE SHARED SAMPLE. Every arm takes the same --images/--n_images/
#    --sample_seed. Break that and compare_populations.py silently degrades
#    from a paired test to an unpaired one, costing roughly 4x the effective n.
#
# BUDGET ARITHMETIC, so you can resize from the pilot's measured rate:
#     total_steps ~= 12*LADDER_N*LADDER_STEPS + 4*N*STEPS
#                 =  12*4*120 + 4*40*300  =  5,760 + 48,000  =  53,760
# Phase 0 measures steps/s on this node and prints the projection. If it says
# you will blow the wall clock, lower N first — never STEPS.
#
# If the job does hit its limit, every phase-2 arm is resumable:
#     python scripts/overfit_population.py ... --resume
#
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

mkdir -p slurm/optimiser
exec 1> "slurm/optimiser/${SLURM_JOB_ID:-local}.out"
exec 2> "slurm/optimiser/${SLURM_JOB_ID:-local}.err"

echo "Running on $(hostname)"
echo "Date: $(date)"

module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

python --version

export ROOT=results/optimiser

# ── the knobs, all in one place ──────────────────────────────────────────────
RES="--img_h 256 --img_w 512 --patch_size 64"
N=40                 # phase 2 images, shared across all four arms
STEPS=300            # phase 2 steps  — do NOT lower this; see above
LADDER_N=4           # phase 1 images
LADDER_STEPS=120     # phase 1 steps  — ranking lrs, not measuring an arm

export BASE="--arch segformer --cityscapes_root $CS $RES"

# --lr_schedule cosine on EVERY arm. Not a preference: attack_image measured a
# 9.6-point spread across four identical un-annealed single-image runs, from
# float-level nondeterminism alone. An ablation hunting an effect smaller than
# its own noise floor finds nothing, whichever way the truth lies.
# --n_panels 0 --no_aggregate: see cut 1 above.
export RUN="--lr_schedule cosine --n_panels 0 --no_aggregate"

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 0 — measure the rate before spending on it
# ═══════════════════════════════════════════════════════════════════════════
# population.sh already says to do this ("the pilot has told you
# seconds/image") and then hardcodes n=100 anyway. One arm, two images, so the
# projection below is measured on THIS node rather than assumed.

echo
echo "═══ PHASE 0 — timing pilot ═══"
T0=$SECONDS
python scripts/overfit_population.py $BASE $RUN \
    --images random --n_images 2 --sample_seed 99 \
    --patch_mode raw --loss_fn ce --placement center \
    --pixel_param sigmoid --optimiser adam --lr 0.01 \
    --steps $LADDER_STEPS --out_root $ROOT --tag pilot
PILOT=$(( SECONDS - T0 ))

python - <<PYEOF
pilot, ladder_steps = $PILOT, $LADDER_STEPS
per_step = pilot / (2 * ladder_steps)
total = 12 * $LADDER_N * ladder_steps + 4 * $N * $STEPS
print()
print(f"  pilot          : {pilot}s for 2 images x {ladder_steps} steps")
print(f"  per attack step: {per_step*1000:.0f} ms  (includes per-image overhead)")
print(f"  planned steps  : {total:,}")
print(f"  PROJECTED      : {total*per_step/60:.1f} min of attack, plus ~16 "
      f"model loads")
print(f"  -> lower N (currently $N) if that overruns. Never lower STEPS.")
print()
PYEOF

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — the lr ladder, per arm, on a small shared sample
# ═══════════════════════════════════════════════════════════════════════════
#
# lr IS NOT IN THE SAME UNITS ACROSS THE PARAMETERISATION AXIS, which is why
# every arm gets its own ladder and is scored at its own best rung. Under
# sigmoid the parameter is a logit: a step of lr moves the pixel by
# lr*sigma'(p), 0.25 at the mid-grey init and falling toward zero as it
# saturates. Under direct the parameter IS the pixel: a step of lr moves it by
# exactly lr. One shared lr would compare two step SIZES and call the
# difference a step RULE.
#
# The ladders are therefore centred differently ON PURPOSE — 0.01 is the
# raw-mode default the existing population runs used, 0.0025 is 0.25*0.01, the
# pixel step that logit-lr 0.01 buys at init. Centring both on one number is
# the mistake this section exists to avoid.

echo "═══ PHASE 1 — lr ladder ═══"
export LADDER_SAMPLE="--images random --n_images $LADDER_N --sample_seed 0"

for LR in 0.005 0.01 0.02; do
  for OPT in adam sign; do
    python scripts/overfit_population.py $BASE $LADDER_SAMPLE $RUN \
        --patch_mode raw --loss_fn ce --placement center \
        --pixel_param sigmoid --optimiser "$OPT" --lr "$LR" \
        --steps $LADDER_STEPS \
        --out_root $ROOT --tag "ladder-sigmoid-${OPT}-lr${LR}"
  done
done

for LR in 0.00125 0.0025 0.005; do
  for OPT in adam sign; do
    python scripts/overfit_population.py $BASE $LADDER_SAMPLE $RUN \
        --patch_mode raw --loss_fn ce --placement center \
        --pixel_param direct --optimiser "$OPT" --lr "$LR" \
        --steps $LADDER_STEPS \
        --out_root $ROOT --tag "ladder-direct-${OPT}-lr${LR}"
  done
done

# ── pick each arm's lr from its own ladder ───────────────────────────────────
# Reads config.optimiser / config.pixel_param / config.lr out of each
# summary.json, not the directory name, so a typo in the loops above cannot
# silently assign the wrong lr. It warns on stderr if an arm's winner is its
# TOP rung — that arm never turned over, is still lr-starved, and its number is
# a floor rather than its best. That is the single most likely way this job
# produces a false "PGD is worse".
eval "$(python analysis/pick_ladder_lr.py $ROOT --glob '*ladder*')"
echo "  picked: A(adam,sigmoid)=$LR_A  B(adam,direct)=$LR_B"
echo "          C(sign,sigmoid)=$LR_C  D(sign,direct)=$LR_D"

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — the four arms at their own best lr, on the shared sample
# ═══════════════════════════════════════════════════════════════════════════

echo
echo "═══ PHASE 2 — the 2x2 ═══"
export SAMPLE="--images random --n_images $N --sample_seed 0"
export ARM="--patch_mode raw --loss_fn ce --placement center --steps $STEPS"

python scripts/overfit_population.py $BASE $SAMPLE $RUN $ARM \
    --pixel_param sigmoid --optimiser adam --lr "$LR_A" \
    --out_root $ROOT --tag "arm-A-adam-sigmoid"

python scripts/overfit_population.py $BASE $SAMPLE $RUN $ARM \
    --pixel_param direct --optimiser adam --lr "$LR_B" \
    --out_root $ROOT --tag "arm-B-adam-direct"

python scripts/overfit_population.py $BASE $SAMPLE $RUN $ARM \
    --pixel_param sigmoid --optimiser sign --lr "$LR_C" \
    --out_root $ROOT --tag "arm-C-sign-sigmoid"

python scripts/overfit_population.py $BASE $SAMPLE $RUN $ARM \
    --pixel_param direct --optimiser sign --lr "$LR_D" \
    --out_root $ROOT --tag "arm-D-sign-direct"

# ── the comparison, paired by image index, A as baseline ─────────────────────
python analysis/compare_populations.py --baseline 0 \
    "$ROOT/segformer_raw_ce_n${N}_arm-A-adam-sigmoid" \
    "$ROOT/segformer_raw_ce_direct_n${N}_arm-B-adam-direct" \
    "$ROOT/segformer_raw_ce_sign_n${N}_arm-C-sign-sigmoid" \
    "$ROOT/segformer_raw_ce_sign_direct_n${N}_arm-D-sign-direct" \
    --out_dir "$ROOT/compare"

echo
echo "DONE in $((SECONDS / 60))m$((SECONDS % 60))s -> $ROOT/compare/"

# ═══════════════════════════════════════════════════════════════════════════
#  HOW TO READ THE RESULT — decide this BEFORE looking, not after
# ═══════════════════════════════════════════════════════════════════════════
#
# D ~ A (within the paired interval)
#     Neither the step rule nor the projection matters at this threat model.
#     The honest write-up says so and stops claiming Adam was NECESSARY: it
#     becomes a choice with no measured cost, not a justified deviation. It
#     also removes the last objection to describing the CE arm as PGD's
#     objective, which is where this question started.
#
# D < A, with B < A and C ~ A
#     The PROJECTION costs. spec.py's claim is the live one and the sigmoid is
#     doing real work. Quote final_frac_at_clip from the direct arms — if it is
#     high, "the patch wants to live on the bound" is confirmed directly and
#     the mechanism is named rather than inferred.
#
# D < A, with C < A and B ~ A
#     The STEP RULE costs. adversarial.py's claim is the live one. CHECK THE
#     LADDER WARNING FIRST: if pick_ladder_lr.py flagged the sign arms as
#     sitting at the ladder edge, they were lr-starved and this branch is an
#     artefact. Widen the ladder and rerun before believing it.
#
# D > A
#     PGD is BETTER here and the deviation cost real attack strength. That is a
#     result, and it means the raw-mode baselines understate what an
#     unconstrained patch can do. It would not touch the CSF arms — those
#     project onto a spectral budget and have their own measured reason for
#     post-step projection — but it would need saying plainly.
#
# IN EVERY BRANCH: the numbers are at 256x512 and do not belong beside the
# 512x1024 tables. If the ablation turns up a real effect, re-run the two arms
# that matter at full resolution before it goes in the thesis.
