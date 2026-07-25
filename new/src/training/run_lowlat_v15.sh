#!/usr/bin/env bash
# =============================================================================
# ABR-Awareness V15 (low-latency operating point) : CPU-server pipeline
# -----------------------------------------------------------------------------
# Purpose: test whether shrinking the playback buffer to a low-latency cap (6 s)
# makes the runtime shield and the binding CMDP constraint actually MATTER.
# Under the V14 broadband setup (30 s buffer) the shield was inert (<0.5%
# intervention, no measurable effect) simply because the operating point was too
# forgiving. V15 changes ONLY the buffer geometry (see abr_multi_env_v15.py);
# everything else (reward scale beta=100, VBR sizes, per-video VMAF, stats) is
# identical, so the two operating points are directly comparable.
#
# Two modes:
#   QUICK (default): train ONLY the `proposed` policy, then run the shield sweep
#     on broadband + harsh-5G traces. This is the decisive, fast signal:
#       * if shield_off now rebuffers a lot while shielded arms do not, and the
#         intervention rate is high  -> the safety story is alive; go FULL.
#       * if shield_off still ~= shielded  -> the regime change did not help;
#         stop and fall back to the honest reframing.
#   FULL  (FULL=1): additionally train all remaining arms and run the full
#     multi-method evaluation for the main results table.
#
# Override via env vars: PYTHON, SEEDS, NUM_ENVS, TS_SCALE, EPISODES, EVAL_SEED,
#                        PARALLEL, FULL
#
# Example (8-core CPU, quick signal):
#   cd new
#   NUM_ENVS=8 SEEDS=0 TS_SCALE=1.0 bash src/training/run_lowlat_v15.sh
#
# Example (full run once the quick signal is positive):
#   NUM_ENVS=8 SEEDS=0 FULL=1 bash src/training/run_lowlat_v15.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # -> new/
cd "${ROOT_DIR}"

PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0}"
NUM_ENVS="${NUM_ENVS:-8}"
TS_SCALE="${TS_SCALE:-1.0}"
PARALLEL="${PARALLEL:-1}"
EPISODES="${EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-0}"
FULL="${FULL:-0}"

HARSH_5G_DIR="data/standardized/test_traces_5g_harsh_v15"
OUT_BB="results/v15_lowlat_shielded_qoe"
OUT_5G="results/v15_lowlat_5g_harsh_shielded_qoe"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=============================================================="
echo "ABR-Awareness V15 (low-latency, buffer cap 6 s) pipeline"
echo "Root=${ROOT_DIR}  seeds=${SEEDS}  num_envs=${NUM_ENVS}  ts_scale=${TS_SCALE}  FULL=${FULL}"
echo "=============================================================="

echo ""
echo "[0] Generate HARSH synthetic 5G traces (around/below the ladder)"
"${PY}" data/generate_5g_standardized_v14.py synth \
    --preset harsh --num 50 --length 300 \
    --out "${HARSH_5G_DIR}" || true

echo ""
echo "[1] Train the low-latency 'proposed' policy (needed for the shield sweep)"
"${PY}" src/training/train_all_models_v15.py \
    --models proposed --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}"

echo ""
echo "[2] Shield sweep (isolation baseline) : broadband (small buffer) + harsh 5G"
"${PY}" src/evaluation/eval_shield_vmaf_online_v15.py \
    --policy proposed_v14 --seed "${EVAL_SEED}" --episodes "${EPISODES}" \
    --trace-dir data/standardized/test_traces \
    --out "${OUT_BB}/online_episodes.csv"
"${PY}" src/evaluation/eval_shield_vmaf_online_v15.py \
    --policy proposed_v14 --seed "${EVAL_SEED}" --episodes "${EPISODES}" \
    --trace-dir "${HARSH_5G_DIR}" \
    --out "${OUT_5G}/online_episodes.csv"

echo ""
echo "[3] Paired stats + conditional VMAF equivalence on the sweep"
"${PY}" src/evaluation/analyze_v14_shielded_qoe.py \
    --episodes-csv "${OUT_BB}/online_episodes.csv" || true
"${PY}" src/evaluation/compute_vmaf_equivalence_v14.py \
    --decisions-csv "${OUT_BB}/online_decisions.csv" || true
"${PY}" src/evaluation/analyze_v14_shielded_qoe.py \
    --episodes-csv "${OUT_5G}/online_episodes.csv" || true
"${PY}" src/evaluation/compute_vmaf_equivalence_v14.py \
    --decisions-csv "${OUT_5G}/online_decisions.csv" || true

echo ""
echo "=============================================================="
echo "QUICK SIGNAL READY. Inspect:"
echo "  ${OUT_BB}/online_summary.csv   (compare shield_off vs shielded rows)"
echo "  ${OUT_5G}/online_summary.csv"
echo "Look for: shield_off Rebuf% high & StallFree_frac low, shielded arms"
echo "much better, and Interv_rate_pct well above 0. That means the shield now"
echo "matters. If shield_off ~= shielded, the regime change did not help."
echo "=============================================================="

if [ "${FULL}" != "1" ]; then
    echo ""
    echo "QUICK mode done. Re-run with FULL=1 to train all arms + full evaluation."
    exit 0
fi

echo ""
echo "[4] FULL: train all remaining low-latency arms"
"${PY}" src/training/train_all_models_v15.py --all \
    --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}" --parallel "${PARALLEL}"

echo ""
echo "[5] FULL: main multi-method evaluation (low-latency, seed ${EVAL_SEED})"
"${PY}" src/evaluation/evaluate_all_models_v15.py \
    --episodes "${EPISODES}" --seed "${EVAL_SEED}"

echo ""
echo "DONE (FULL). Key outputs under: ${ROOT_DIR}/results/"
echo "  summary_master_v14_v15_lowlat_seed${EVAL_SEED}.csv"
echo "  ${OUT_BB}/*.csv , ${OUT_5G}/*.csv"
