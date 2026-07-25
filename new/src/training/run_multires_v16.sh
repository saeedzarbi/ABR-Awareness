#!/usr/bin/env bash
# =============================================================================
# ABR-Awareness V16 : validate the ORIGINAL claim (VMAF-aware shield > index)
# via a REAL non-monotone, per-chunk, multi-resolution VMAF ladder.
# -----------------------------------------------------------------------------
# Rationale: on a single-resolution ladder VMAF is monotone in the bitrate index,
# so a VMAF-aware shield provably equals an index shield (measured gain = 0 at
# every realistic downgrade depth). The perceptual ranking only becomes a live
# choice on a non-monotone per-chunk ladder, which a multi-resolution encoding
# (resolution/quality crossovers, the convex-hull effect) produces. This pipeline
# builds that ladder, VERIFIES crossovers exist, retrains on it, and runs the
# isolation A/B (vmaf_aware vs highest_feasible).
#
# PREREQUISITE: ffmpeg built with libx264 AND libvmaf, plus the source videos.
#   Ubuntu/Debian:  sudo apt-get update && sudo apt-get install -y ffmpeg
#     (verify libvmaf:  ffmpeg -hide_banner -filters | grep libvmaf )
#     If the distro ffmpeg lacks libvmaf, use a static build from
#     https://johnvansickle.com/ffmpeg/  (the *-gpl builds include libvmaf).
#   Place source videos at:  raw_videos/{bigbuckbunny,crowd_run,tearsofsteel_short,sintel}.mp4
#   (override the directory with RAW_DIR=...)
#
# Env vars: PYTHON, RAW_DIR, SEEDS, NUM_ENVS, TS_SCALE, EPISODES, EVAL_SEED, FULL
#
# Quick (build ladder + train `proposed` + isolation sweep):
#   cd new
#   RAW_DIR=raw_videos NUM_ENVS=8 SEEDS=0 bash src/training/run_multires_v16.sh
# Full (all arms + full eval), after the quick signal is positive:
#   RAW_DIR=raw_videos NUM_ENVS=8 SEEDS=0 FULL=1 bash src/training/run_multires_v16.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # -> new/
cd "${ROOT_DIR}"

PY="${PYTHON:-python}"
RAW_DIR="${RAW_DIR:-raw_videos}"
SEEDS="${SEEDS:-0}"
NUM_ENVS="${NUM_ENVS:-8}"
TS_SCALE="${TS_SCALE:-1.0}"
PARALLEL="${PARALLEL:-1}"
EPISODES="${EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-0}"
FULL="${FULL:-0}"

PERCHUNK_CSV="data/vmaf_scores/vmaf_perchunk_multires.csv"
OUT_BB="results/v16_perchunk_shielded_qoe"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=============================================================="
echo "ABR-Awareness V16 (per-chunk multi-resolution VMAF) pipeline"
echo "Root=${ROOT_DIR}  raw=${RAW_DIR}  seeds=${SEEDS}  envs=${NUM_ENVS}  FULL=${FULL}"
echo "=============================================================="

echo ""
echo "[0] Verify ffmpeg + libvmaf"
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[FATAL] ffmpeg not found. Install it (see header) then re-run."; exit 2
fi
# NOTE: capture the filter list into a variable FIRST, then grep the string.
# Do NOT pipe `ffmpeg -filters | grep -q libvmaf`: with `set -o pipefail`, grep -q
# closes the pipe early on match, ffmpeg dies with SIGPIPE, and the pipeline is
# reported as failed even though libvmaf IS present (false negative).
FF_FILTERS="$(ffmpeg -hide_banner -filters 2>/dev/null || true)"
case "${FF_FILTERS}" in
    *libvmaf*) : ;;  # present
    *) echo "[FATAL] this ffmpeg lacks the libvmaf filter. Use a build that includes it."; exit 2 ;;
esac
echo "  ffmpeg + libvmaf OK"

echo ""
echo "[1] Build the non-monotone per-chunk multi-resolution VMAF ladder"
"${PY}" data/build_multires_vmaf.py --raw-dir "${RAW_DIR}" \
    --videos bigbuckbunny,crowd_run,tearsofsteel_short,sintel

if [ ! -f "${PERCHUNK_CSV}" ]; then
    echo "[FATAL] ${PERCHUNK_CSV} was not produced; cannot continue."; exit 1
fi
echo ""
echo ">>> Inspect the inversion report printed above. If mean/max gain is ~0,"
echo ">>> the ladder has no usable crossovers and training will NOT validate the"
echo ">>> claim. Press Ctrl-C now if so. Continuing in 10s..."
sleep 10

echo ""
echo "[2] Train the 'proposed' policy on the per-chunk ladder"
"${PY}" src/training/train_all_models_v16.py \
    --models proposed --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}"

echo ""
echo "[3] Shield isolation sweep (vmaf_aware vs highest_feasible) on the per-chunk ladder"
"${PY}" src/evaluation/eval_shield_vmaf_online_v16.py \
    --policy proposed_v14 --seed "${EVAL_SEED}" --episodes "${EPISODES}" \
    --trace-dir data/standardized/test_traces \
    --out "${OUT_BB}/online_episodes.csv"

echo ""
echo "[4] Paired stats + conditional VMAF equivalence (the isolation test)"
"${PY}" src/evaluation/analyze_v14_shielded_qoe.py \
    --episodes-csv "${OUT_BB}/online_episodes.csv" || true
"${PY}" src/evaluation/compute_vmaf_equivalence_v14.py \
    --decisions-csv "${OUT_BB}/online_decisions.csv" || true

echo ""
echo "=============================================================="
echo "QUICK SIGNAL READY. Inspect:"
echo "  ${OUT_BB}/online_summary.csv     (compare vmaf_aware_* vs highest_feasible_* rows)"
echo "  ${OUT_BB}/paired_stats_v14.csv   (row: 'VMAF-aware vs highest-index [isolation]')"
echo "  ${OUT_BB}/vmaf_equivalence_conditional_v14.csv"
echo "Success = the isolation rows now have n_nonzero > 0 and a positive VMAF"
echo "difference in favour of vmaf_aware (the original claim, on a real ladder)."
echo "=============================================================="

if [ "${FULL}" != "1" ]; then
    echo ""
    echo "QUICK mode done. Re-run with FULL=1 to train all arms + full evaluation."
    exit 0
fi

echo ""
echo "[5] FULL: train all remaining arms on the per-chunk ladder"
"${PY}" src/training/train_all_models_v16.py --all \
    --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}" --parallel "${PARALLEL}"

echo ""
echo "[6] FULL: main multi-method evaluation (per-chunk, seed ${EVAL_SEED})"
"${PY}" src/evaluation/evaluate_all_models_v16.py \
    --episodes "${EPISODES}" --seed "${EVAL_SEED}"

echo ""
echo "DONE (FULL). Key outputs under ${ROOT_DIR}/results/ (v16_perchunk_*, summary_master_v14_v16_perchunk_seed${EVAL_SEED}.csv)"
