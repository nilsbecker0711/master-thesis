#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# UNIVERSAL CSF-BOUNDED PATCH — the non-adaptive control for --patch_mode csf.
#
# ONE residual delta, shared across the whole dataset, added onto whatever
# content it lands on. mode='csf' optimises a fresh residual per image and takes
# its base from that image; this differs in exactly two ways -- delta is shared,
# and the reference luminance is fixed -- and in nothing else.
#
# THE TAU SWEEP IS BUILT IN FROM THE START, deliberately. There is a real chance
# this does close to nothing: a 128x128 footprint at threshold amplitude carries
# roughly an order of magnitude less perturbation energy than a global JND
# perturbation, it has to act through a remote objective on a much larger frame,
# and universality forbids exploiting any particular scene. If it fails, the
# failure has to be legible -- where the optimiser spent its budget, how the
# realised visibility spread across the val set, and at which tau it starts
# working -- rather than a table of zeros bolted onto a single disappointing
# run.
#
# --lr_schedule cosine ON EVERY RUN. PR #15 measured a flat lr swinging 9.6 mIoU
# across four identical single-image invocations, because Adam's step is ~lr per
# coordinate whatever the gradient is, so a non-annealed run is still taking
# full-size steps at the end and the reported number is wherever the walk
# happened to stop. That PR gave the flag to overfit.py and
# overfit_population.py but NOT to train.py; train.py has it now, defaulting to
# 'plateau' so every earlier run keeps its meaning. Universal runs should not
# use the default.

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/universal
exec 1> "slurm/universal/${SLURM_JOB_ID}.out"
exec 2> "slurm/universal/${SLURM_JOB_ID}.err"
module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

python --version

# GEOMETRY IS PINNED AND MUST STAY PINNED. universal_csf refuses to run unless
# --patch_size equals int(img_h * patch_scale), because resampling a residual
# resamples its spectrum and the budget its bins were projected onto would stop
# describing the pasted signal. At 512 x 0.25 that is 128.
BASE="--arch segformer --cityscapes_root $CS --img_h 512 --img_w 1024"
GEOM="--patch_mode universal_csf --patch_size 128 --patch_scale 0.25 --placement center"
# --num_workers 16 MATCHES --cpus-per-task ABOVE. The default is 4, which is
# what every earlier run used and is left alone for that reason -- but
# Cityscapes decodes a 2048x1024 PNG per sample and then resizes it, and at
# 4 workers that can cost more per batch than the segmentor step it feeds.
# Nine runs is the wrong place to discover the loader was the bottleneck.
# --lr 0.01, NOT the 0.2 that works for the single-image overfit. The two modes
# optimise different objects. mode='csf' maps its parameter through a smooth,
# SCALE-INVARIANT squash and then renormalises to exactly tau -- scaling that
# parameter by 1000x moves the rendered patch by 7.5% -- so the parameter is
# effectively a direction and lr sets how fast it rotates. Here the parameter IS
# the residual, in pixel units (rms 0.027 at tau 0.25), and the constraint is a
# projection, so lr sets movement inside a small ball and overshoot is thrown
# away by the projection rather than absorbed.
#
# Measured mean relative change of the applied residual per step:
#     lr      csf      universal_csf
#     0.2     0.50     1.89   <- each step discards the previous residual
#     0.05    0.084    1.09
#     0.01    0.031    0.15
#     0.001   0.0038   0.039
#
# and frac_at_bound after 200 steps: lr 0.2 -> 0.931 (pinned to the boundary,
# phase-only), lr 0.01 -> 0.508 (healthy interior). WATCH frac_at_bound in the
# logs: pinned above ~0.9 means lr is too large whatever this file says.
OPT="--loss_fn cospgd --lr_schedule cosine --lr 0.01 --batch_size 4 --num_workers 16"

# ══════════════════════════════════════════════════════════════════════════════
#  TAU AND VISIBILITY — read this before quoting any rung
# ══════════════════════════════════════════════════════════════════════════════
# tau = 1 is the detection threshold of Barten's average observer under the
# assumed geometry. A perturbation at tau <= 1 is at or below threshold; above
# it, the claim of invisibility is simply gone.
#
# BUT THE ENFORCED TAU IS NOMINAL, AND NOMINAL UNDER-STATES. The budget is
# enforced with the legacy mu = 0.5 convention while Cityscapes footprints sit
# near Y = 0.10, so the realised visibility is ~1.7x the nominal number:
#
#     full val split   Y 0.1029   1.69x   -> realised 1.0 at nominal tau 0.592
#     train centre     Y 0.0971   1.70x   -> realised 1.0 at nominal tau 0.589
#
# SO THE INVISIBILITY CEILING IS NOMINAL TAU ~= 0.59, NOT 1.0. Every rung above
# that is a DIAGNOSTIC, not a result, and must never be quoted as an attack that
# preserves imperceptibility. They exist only to separate "the premise fails"
# from "the plumbing is broken": a null at 0.25 means nothing until a non-null
# exists somewhere above it.
#
# Run --csf_lref to enforce the calibrated budget instead, and the nominal and
# realised numbers converge -- at the cost of no longer being comparable with
# the overfit ladder, which is why it is a control rather than the default.

# ── the ladder that carries the result: BELOW threshold ──────────────────────
# 0.05 0.1 0.25 0.5 are EXACTLY the rungs csf_single.sh ran for the per-image
# overfit, deliberately, so control and method are compared at identical tau
# rather than at whatever each happened to use. All four realise below 1.0
# (0.5 nominal -> ~0.85 realised), so the whole ladder stays sub-threshold.
for TAU in 0.05 0.1 0.25 0.5; do
  LR=$(lr_for "$TAU")
  echo "========== tau = $TAU  (lr $LR)  BELOW THRESHOLD =========="
  python scripts/train.py $BASE $GEOM $OPT --lr "$LR" \
      --csf_threshold "$TAU" --epochs 20 --val_every 5 --val_images 500 \
      --tag sweep
done

# ── diagnostic rungs: ABOVE threshold, not results ───────────────────────────
# Where does the attack start working at all? If nothing moves anywhere on this
# ladder either, the failure is mechanical rather than perceptual.
for TAU in 1.0 2.0 8.0; do
  LR=$(lr_for "$TAU")
  echo "========== tau = $TAU  (lr $LR)  ABOVE THRESHOLD — DIAGNOSTIC =========="
  python scripts/train.py $BASE $GEOM $OPT --lr "$LR" \
      --csf_threshold "$TAU" --epochs 20 --val_every 5 --val_images 500 \
      --tag diagnostic
done

# ── the calibrated-budget control ────────────────────────────────────────────
# --csf_lref 0.0971 is the measured Cityscapes train median linear luminance at
# centre placement, switching the budget from the legacy mu=0.5 convention to
# the calibrated one (~1.7x tighter). At the same nominal tau the difference IS
# the calibration error, expressed as attack strength rather than as a ratio.
python scripts/train.py $BASE $GEOM $OPT --lr "$(lr_for 0.25)" \
    --csf_threshold 0.25 --csf_lref 0.0971 --epochs 20 --val_every 5 \
    --val_images 500 --tag calibrated

# ── the composite control ────────────────────────────────────────────────────
# clip is the default and the honest threat model, but it truncates the residual
# where content has no headroom, violating tau permissively -- reported as
# frac_clipped, which hit 0.098 on the darkest val frame. fit rescales instead,
# preserving the spectrum at the cost of a PER-IMAGE scale, an adaptation a
# universal patch should not have. Agreement means clip can be quoted without a
# caveat; disagreement IS the caveat.
python scripts/train.py $BASE $GEOM $OPT --lr "$(lr_for 0.5)" \
    --csf_threshold 0.5 --csf_composite fit --epochs 20 --val_every 5 \
    --val_images 500 --tag fitcomposite

# ── the schedule control ─────────────────────────────────────────────────────
# One rung without annealing, so the claim that cosine matters here is measured
# on THIS mode rather than inherited from the single-image result.
python scripts/train.py $BASE $GEOM $OPT --lr "$(lr_for 0.5)" --lr_schedule plateau \
    --csf_threshold 0.5 --epochs 20 --val_every 5 --val_images 500 \
    --tag plateau

echo "Done: $(date)"
