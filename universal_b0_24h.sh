#!/bin/bash
#SBATCH -p gpu_a100_il
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm/attack/universal_b0_%j.out
#SBATCH --error=slurm/attack/universal_b0_%j.err

# ── WHY FOUR ARMS AND NOT ONE LONG RUN ──────────────────────────────────────
# The 90-epoch run at lr 0.1 produced a FLAT val curve: 18 checkpoints from
# epoch 5 to 90, drop_remote bouncing in +0.03..+0.23 with the second half
# (+0.122 mean) no better than the first (+0.146), and flip pinned at 0.4% at
# every checkpoint. Training cospgd moved 6e-5 across the last 20 epochs while
# the patch itself kept moving (frac_at_bound 0.73->0.87). That is a plateau,
# not slow learning, so more epochs of the same buy nothing. Depth is not where
# the information is; arms are.
#
# ── THE RUN WAS ALSO NOT AT TAU ─────────────────────────────────────────────
# csf_enforce was 'nominal', and realised_visibility reached 1.89 mean / 2.92
# max against tau=0.25 -- 7.6x and 11.7x the budget. frac_clipped was only
# 0.01, which is the point: a clamp BENDS the residual rather than scaling it,
# and the harmonics land in the low/mid bands where the CSF is 500-700. So no
# number from that run is reportable as a tau=0.25 result.
#
# Every arm here uses --csf_enforce realised. EXPECT IT TO LOOK WEAKER:
# fit_to_visibility only ever scales DOWN (it bisects c in [0,1]), so the
# effective perturbation shrinks. That is the first honest measurement, not a
# regression.
#
# ── THE ARMS ────────────────────────────────────────────────────────────────
# The reason the previous result is uninterpretable is that nothing in it was
# guaranteed to produce signal. A flat curve could mean "the budget is too
# tight" or "the setup cannot learn" and there was no way to tell. The tau=1.0
# arm fixes that: it is ~1.9 JND locally, i.e. VISIBLE, and is included ONLY as
# a positive control. It is never reportable as an invisible attack.
#
#   D  baseline  tau 0.25, lr 0   -- random projected init, 1 epoch. Makes every
#                                    other number mean something.
#   C  claim     tau 0.25         -- the result you actually want.
#   A  control   tau 1.0          -- must move. If it does not, the problem is
#                                    the setup (lr, loss, patch scale), not tau,
#                                    and no tau tuning will save it.
#   B  middle    tau 0.5          -- ~0.95 JND local, at the detection
#                                    threshold. The most likely place for a
#                                    real result that is still defensible.
#
# ORDER IS DELIBERATE: D, C, A, B. If the job dies you keep baseline + claim +
# control, which is already a complete story. B is the nice-to-have.
#
# ── TIMING ──────────────────────────────────────────────────────────────────
# Measured 6.7 min/epoch (90 epochs in the 10h run). 55 epochs x 3 arms is
# ~18.4h, leaving ~5h of margin in the 24h wall. The previous job died to a
# time limit; losing all four arms the same way is the worst outcome, so the
# margin is deliberate. Raise EPOCHS only if you have measured a faster epoch.
#
# Each arm anneals its cosine to completion inside its own budget -- a finished
# 55-epoch schedule beats a truncated 150-epoch one, which is exactly how the
# previous run ended up reporting "wherever the walk happened to be".

echo "Running on $(hostname)"
echo "Date: $(date)"

mkdir -p slurm/attack

module --ignore_cache load "cuda/11.8"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /pfs/work9/workspace/scratch/ma_nilbecke-thesis/miniconda3/envs/thesis_backup3
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CS=/pfs/work9/workspace/scratch/ma_nilbecke-thesis/data/cityscapes

echo "Python path:"; which python
echo "Python version:"; python --version

EPOCHS="${EPOCHS:-55}"
LR="${LR:-0.01}"

# GEOMETRY IS PINNED: universal_csf refuses to run unless patch_size equals
# int(img_h * patch_scale), because resampling a residual resamples its
# spectrum and the budget its bins were projected onto stops describing the
# pasted signal. At 512 x 0.25 that is 128.
BASE="--arch segformer --cityscapes_root $CS --img_h 512 --img_w 1024"
GEOM="--patch_mode universal_csf --patch_size 128 --patch_scale 0.25 --placement center"
COMMON="--loss_fn cospgd --lr_schedule cosine --batch_size 4 --num_workers 16 \
        --csf_enforce realised --val_images 500 --val_every 3 --out_root results/runs"

echo ""
echo "=============================================================="
echo "  D  baseline: random projected init at tau 0.25 (lr 0)"
echo "=============================================================="
# --lr 0 so Adam's step is zero and the patch stays at its init. Under the
# projection that init is the CSF envelope with every bin at its budget, i.e.
# a full-spectrum noise residual at exactly tau -- the right null for "what did
# optimisation actually buy". --lr_schedule none avoids CosineAnnealingLR
# having anything to anneal.
python scripts/train.py $BASE $GEOM \
    --loss_fn cospgd --lr 0 --lr_schedule none --epochs 1 \
    --batch_size 4 --num_workers 16 --csf_enforce realised \
    --val_images 500 --val_every 1 --out_root results/runs \
    --csf_threshold 0.25 --tag "u_baseline_random_t0_25"

for ARM in "C 0.25" "A 1.0" "B 0.5"; do
  set -- $ARM
  NAME="$1"; TAU="$2"
  echo ""
  echo "=============================================================="
  echo "  $NAME  tau=$TAU  lr=$LR  epochs=$EPOCHS"
  echo "=============================================================="
  python scripts/train.py $BASE $GEOM $COMMON \
      --lr "$LR" --epochs "$EPOCHS" \
      --csf_threshold "$TAU" \
      --tag "u_${NAME}_t$(echo $TAU | tr . _)_lr$(echo $LR | tr . _)_e${EPOCHS}"
done

echo ""
echo "=============================================================="
echo "  READ IT LIKE THIS"
echo "=============================================================="
echo "  1. Does cospgd move at all? Compare the 4th decimal between the"
echo "     first and last epoch of each arm. The 90-epoch run moved 6e-5."
echo "  2. Does realised_visibility now stay <= tau, with frac_clipped no"
echo "     longer able to break it? That is the correctness check."
echo "  3. Arm A (tau 1.0) MUST move. If it does not, stop tuning tau --"
echo "     the next axis is patch scale (--patch_scale 0.35 --patch_size 179),"
echo "     not the budget."
echo "  4. Compare arm C against arm D. If trained == random, the honest"
echo "     result is 'no measurable universal attack at tau=0.25'."
echo ""
echo "    grep '^\[epoch' slurm/attack/universal_b0_*.out"
echo "    grep 'VAL remote' slurm/attack/universal_b0_*.out"
echo "=============================================================="
