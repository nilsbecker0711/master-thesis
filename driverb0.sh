#!/bin/bash
#SBATCH -p cpu_il
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#
# Feeds 30-minute dev-GPU slices of benchmarkb0.sh back to back until every
# config is done.
#
#     sbatch driverb0.sh            all configs
#     sbatch driverb0.sh raw        one config
#
# Idempotent: each slice passes --resume and picks up from train_state.pt,
# and a finished config is skipped. Relaunch as often as needed.
#
# 72h, not 5 minutes: sbatch --wait blocks for the slice's whole walltime
# plus queue time, so the driver must outlive everything it submits.
#
# NOT on a timer. --wait returns when the slice TERMINATES and the next one
# goes in immediately.

set -u

# NOT dirname "$BASH_SOURCE" -- sbatch runs a COPY of this out of
# /var/spool/slurmd/job<id>/, which is neither the repo nor writable.
ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$ROOT" || exit 1

JOB="benchmarkb0.sh"
ONLY="${1:-}"
TAB=$'\t'
LOGS="slurm/benchmark"

# Created BEFORE anything writes into it. The previous driver redirected its
# own output there with the mkdir commented out; a failed exec redirect
# kills a non-interactive shell outright, so it died with no log at all.
mkdir -p "$LOGS" || exit 1

echo "driver on $(hostname), $(date)"
echo "root    : $ROOT"

# ── preflight: fail now, loudly, not in three hours ─────────────────────────
for f in "$JOB" scripts/train.py; do
    [ -f "$f" ] && continue
    echo "FATAL: $f not found under $ROOT. sbatch from the repo root, or set"
    echo "       REPO_ROOT=/path/to/repo."
    exit 1
done
command -v sbatch >/dev/null || {
    echo "FATAL: no sbatch on this node. Run the driver on the login node"
    echo "       with nohup instead."
    exit 1; }
bash "$JOB" --list >/dev/null 2>&1 || {
    echo "FATAL: '$JOB --list' failed. Run it by hand and fix it first."
    exit 1; }

# EVERY resource on the command line. sbatch precedence is command line >
# environment > #SBATCH directives, and this driver runs with
# SLURM_CPUS_PER_TASK=1 / SLURM_MEM_PER_NODE=2000 in its environment --
# inherited, those would override the slice's directives and hand it 1 CPU
# and 2 GB.
SLICE="-p dev_gpu_a100_il -t 00:30:00 --gres=gpu:1 -N 1 -n 1 \
       --cpus-per-task=16 --mem=40000"

# name<TAB>out_dir. Read back from the job rather than duplicated here, so
# an out_dir cannot drift between the two.
mapfile -t ROWS < <(bash "$JOB" --list)
[ "${#ROWS[@]}" -gt 0 ] || { echo "FATAL: empty config table."; exit 1; }

# results.json is written once, at the very end, after the final 500-image
# evaluation -- so it is the authoritative done signal. The per-epoch PNGs
# are progress only: a slice killed after the last epoch but before that
# evaluation leaves the count topped out with no results.json, and the next
# slice finishes the tail.
epochs_done () {
    [ -d "$1/intermediate_patches" ] \
        && find "$1/intermediate_patches" -name 'epoch*.png' | wc -l \
        || echo 0
}
finished () { [ -f "$1/results.json" ]; }

# Strip the driver's own SLURM_* in a subshell so nothing is inherited; the
# driver's environment is left intact.
submit_slice () {
    ( for v in ${!SLURM_@}; do unset "$v"; done
      sbatch --wait $SLICE \
             --job-name="b0_$1" \
             --output="$LOGS/b0_$1_%j.out" \
             --error="$LOGS/b0_$1_%j.err" \
             "$JOB" "$1" )
}

echo "job     : $JOB"
echo "slice   : $SLICE"
echo "configs : ${#ROWS[@]}${ONLY:+  (only: $ONLY)}"
echo

slices=0
for row in "${ROWS[@]}"; do
    # Skip anything that is not name<TAB>out_dir. A stray echo or a
    # `python --version` in the job script before its --list branch exits
    # becomes a PHANTOM CONFIG named after that line: finished() cannot
    # match it, the driver submits it, the job says "no config named ...",
    # and two dev-GPU slices are gone. Observed exactly that way.
    case "$row" in
        *"$TAB"*) ;;
        *) echo "IGNORING non-table line from --list: >$row<"
           echo "  something in $JOB writes to stdout before --list exits"
           continue ;;
    esac
    name="${row%%"$TAB"*}"
    out="${row#*"$TAB"}"
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue

    if finished "$out"; then
        echo "$(date +%F\ %H:%M)  $name  already complete, skipping"
        continue
    fi
    echo "$(date +%F\ %H:%M)  === $name -> $out ==="

    stalled=0 prev=-1
    while :; do
        if finished "$out"; then
            echo "$(date +%F\ %H:%M)  $name  COMPLETE at epoch $(epochs_done "$out")"
            break
        fi
        n=$(epochs_done "$out")
        # Two consecutive slices with no new epoch means failing, not slow.
        # ONE is allowed: a slice can legitimately spend its first minutes
        # on conda, the model load and cudnn warmup. NOTE this also fires if
        # a single epoch does not FIT in 30 minutes -- 250 steps at native
        # resolution -- because train.py only checkpoints at epoch
        # boundaries, so such a config makes no progress ever. If that is
        # what happened, give the slice a longer walltime, not more slices.
        if [ "$n" -eq "$prev" ]; then
            stalled=$((stalled + 1))
            if [ "$stalled" -ge 2 ]; then
                echo "$(date +%F\ %H:%M)  $name  no progress in 2 slices at epoch $n"
                echo "  giving up on this config; see $LOGS/b0_${name}_*.err"
                break
            fi
        else
            stalled=0
        fi
        prev=$n

        echo "$(date +%F\ %H:%M)  $name  epoch $n  submitting slice"
        submit_slice "$name"; rc=$?
        slices=$((slices + 1))
        echo "$(date +%F\ %H:%M)  $name  slice returned $rc"
        # --wait exits with the JOB's code. 3 is the slice's environment
        # preflight: no torch, or a GPU it cannot see. That is true of every
        # slice this driver will submit, not just this config.
        if [ "$rc" -eq 3 ]; then
            echo
            echo "FATAL: slice refused to start -- bad python environment."
            echo "  See $LOGS/b0_${name}_*.err. Fix and relaunch; nothing is lost."
            exit 3
        fi
        sleep 10          # never spin, in case sbatch rejects instantly
    done
done

echo
echo "driver done after $slices slices, $(date)"
for row in "${ROWS[@]}"; do
    case "$row" in *"$TAB"*) ;; *) continue ;; esac
    name="${row%%"$TAB"*}"; out="${row#*"$TAB"}"
    finished "$out" && s=done || s=INCOMPLETE
    printf '  %-12s %-11s epoch %-5s %s\n' "$name" "$s" "$(epochs_done "$out")" "$out"
done
echo
echo "Index:  python analysis/build_index.py --runs results/runs/bench_nesti \\"
echo "            --out results/tables/bench_nesti/index.csv"
