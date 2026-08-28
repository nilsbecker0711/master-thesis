#!/bin/bash
#SBATCH -p gpu_a100_il
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem=100000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# ═══════════════════════════════════════════════════════════════════════════
#  THE CSF OVERFIT CONFIG, PROMOTED FROM AN ANECDOTE TO A MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════
#
# The single-image result this validates: --patch_mode csf --from_image,
# tau 0.25, lr 0.2, 1000 steps, cosine schedule. On image 420 that reached a
# remote drop of 50.5 +/- 1.5 over repeats, with a realised visibility running
# ~2.8x over tau.
#
# ONE IMAGE IS AN ANECDOTE and neither of those two numbers survives being
# quoted from it. This runs the IDENTICAL config over n validation images —
# same function, patchreach.patch.optimise.attack_image(), so the population
# and the single-image number cannot drift — and reports the distribution.
#
# WHAT COMES OUT, in ONE shared directory ($OUT):
#
#   summary.json + distribution.png    the per-image drop, its bootstrap CI,
#                                      and whether n was large enough for the
#                                      claim being made.
#   aggregate/                         the diagnostic suite POOLED OVER EVERY
#                                      IMAGE — flows, reach, ring profiles,
#                                      contestability, pooled per-class IoU,
#                                      realised visibility vs tau, and the
#                                      convergence band. This is what a
#                                      mechanism claim gets quoted from.
#   diagnostics/imgNNNN/               panels + the full per-image suite, for
#                                      the BEST 3 ONLY.
#   patches/imgNNNN/best.pt            every image's patch, all n of them.
#
# WHY --panel_select best IS ALLOWED HERE and is not allowed in population.sh:
# the pooled figures now carry every mechanism claim, so the panels illustrate
# rather than evidence. The distribution figure highlights the three chosen
# images inside the full spread, so a reader can see where in the distribution
# the pretty pictures came from — which is the thing that makes showing only
# winners honest rather than selective.

echo "Running on $(hostname)"
echo "Date: $(date)"
mkdir -p slurm/validation
exec 1> "slurm/validation/${SLURM_JOB_ID}.out"
exec 2> "slurm/validation/${SLURM_JOB_ID}.err"
module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/lib
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

python --version

# ── the sample ───────────────────────────────────────────────────────────────
# 100 supports a +/-2 point claim at the sigma this attack has shown so far;
# 500 is the whole Cityscapes val set and is what Gu et al. (ECCV 2024) do.
# Start at 100. summary.json reports images_needed_pm2 against the spread it
# ACTUALLY observed, so raise N only if that number says to — and if it does,
# resume rather than restart (see the bottom of this file).
export N=500
export SEED=68

# ── the config under test. DO NOT edit these three without a reason ──────────
# --csf_threshold 0.25   the rung every existing overfit number is quoted at.
# --lr 0.2               correct for mode='csf' and ONLY for mode='csf'. That
#                        parameter maps through a scale-invariant squash and is
#                        renormalised to exactly tau, so it is effectively a
#                        DIRECTION and lr sets how fast it rotates. The
#                        universal_csf parameter IS the residual in pixel units
#                        and needs lr 0.01; do not carry that number across.
# --steps 1000           400 was measured truncating the run and producing a
#                        fake bimodality in any_flip_rate. aggregate/
#                        convergence.png is the check that 1000 is enough: a
#                        flat tail is the evidence, and the run prints
#                        NOT CONVERGED in words if the tail is still climbing.
export TAU=0.25
export LR=0.2
export STEPS=1000

export BASE="--arch segformer --cityscapes_root $CS --img_h 512 --img_w 1024"
export ATTACK="--patch_mode csf --from_image --loss_fn cospgd --csf_threshold $TAU --placement center"
# --lr_schedule cosine is non-negotiable at this lr. A flat lr was measured
# swinging 9.6 mIoU points across four identical single-image runs, because
# Adam's step is ~lr per coordinate whatever the gradient is, so the run never
# settles and the reported number is wherever the walk stopped.
# --log_every 25 gives the convergence band 40 points across 1000 steps. It is
# also the granularity at which best.pt is checkpointed, so it is not free
# resolution — it is the resolution of "best".
export RUN="--steps $STEPS --lr $LR --lr_schedule cosine --log_every 25"
export PANELS="--n_panels 3 --panel_select best --select_key drop_remote"

# ── ONE shared directory, named explicitly ───────────────────────────────────
# Explicit rather than auto-incremented, because --resume needs a stable path
# and because this is meant to be THE directory for this claim rather than
# run_3 of an accreting pile.
export OUT="results/validation/segformer_csf_tau${TAU}_lr${LR}_s${STEPS}_n${N}"

# ── 0. PILOT — 5 images, so the walltime is arithmetic and not a guess ───────
# 1000 steps is 3.3x population.sh's 300, and this is the first time this
# config runs at length over a population. The pilot prints s/img; multiply by
# N before trusting the -t above.
python scripts/overfit_population.py $BASE $ATTACK $RUN \
    --images random --n_images 5 --sample_seed $SEED \
    --n_panels 1 --panel_select best \
    --out_dir "${OUT}_pilot"

# ── 1. THE RUN ───────────────────────────────────────────────────────────────
python scripts/overfit_population.py $BASE $ATTACK $RUN $PANELS \
    --images random --n_images $N --sample_seed $SEED \
    --out_dir "$OUT"

echo "Done: $(date)"

# ═══════════════════════════════════════════════════════════════════════════
#  IF IT HITS THE WALLTIME, OR IF N HAS TO GO UP
# ═══════════════════════════════════════════════════════════════════════════
#
# Resubmit the SAME command with --resume added. Every image's record, the
# pooled confusion matrices and the pooled diagnostic tensors are checkpointed
# after each image, so nothing is recomputed and nothing is lost:
#
#   python scripts/overfit_population.py $BASE $ATTACK $RUN $PANELS \
#       --images random --n_images $N --sample_seed $SEED \
#       --out_dir "$OUT" --resume
#
# To GROW the sample, raise --n_images and keep --sample_seed: the sample is a
# seeded permutation truncated to n, so n=200 is n=100 plus 100 new images and
# --resume attacks only the new ones. Changing the seed instead throws the
# first 100 away.
#
# ── the comparison that makes the number mean something ──────────────────────
# A drop under a perceptual constraint is only interpretable next to the
# unconstrained ceiling on the SAME images. Run this arm with the same sample
# and pair them by image index:
#
#   python scripts/overfit_population.py $BASE $RUN $PANELS \
#       --patch_mode raw --loss_fn cospgd --placement center \
#       --images random --n_images $N --sample_seed $SEED \
#       --out_dir "${OUT}__raw_ceiling"
#   python analysis/compare_populations.py "${OUT}__raw_ceiling" "$OUT"
#
# compare_populations.py pairs BY IMAGE INDEX, and the paired difference is the
# only test that resolves a small effect under this attack's scene-to-scene
# spread. That is why both arms must keep --sample_seed $SEED.
