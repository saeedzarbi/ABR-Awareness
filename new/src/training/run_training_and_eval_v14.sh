#!/usr/bin/env bash
# =============================================================================
# ABR-Awareness V14 (reviewer-response) : full CPU-server pipeline
# -----------------------------------------------------------------------------
# Runs, in order:
#   0. provenance record (train/test split + ladder monotonicity)
#   1. training of all V14 arms (corrected QoE scale, bindable CMDP)
#   2. main multi-method evaluation (rescaled QoE, tail metrics, seen/unseen)
#   3. shield sweep with the highest-feasible-index isolation baseline
#      (broadband + synthetic 5G stress)
#   4. paired stats (Wilcoxon zsplit + Holm) and conditional VMAF equivalence
#
# Designed for a CPU-only server. Override anything via environment variables:
#   PY, SEEDS, NUM_ENVS, TS_SCALE, PARALLEL, EPISODES
#
# Example (8-core CPU, single seed, ~2M-step budget):
#   cd new
#   NUM_ENVS=8 SEEDS=0 TS_SCALE=1.0 bash src/training/run_training_and_eval_v14.sh
#
# To add seeds later (reviewer asked for >=3), rerun with SEEDS=1,2 — the
# per-seed model dirs mean earlier seeds are not overwritten.
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # -> new/
cd "${ROOT_DIR}"

PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0}"            # comma-separated training seeds
NUM_ENVS="${NUM_ENVS:-8}"     # match server logical cores
TS_SCALE="${TS_SCALE:-1.0}"   # scale factor on timestep budgets
PARALLEL="${PARALLEL:-1}"     # parallel (model,seed) jobs
EPISODES="${EPISODES:-20}"    # eval episodes per video (4 videos -> 4*EPISODES)
EVAL_SEED="${EVAL_SEED:-0}"   # which trained seed to evaluate / replay

# Keep math libs single-threaded so env workers own the cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=============================================================="
echo "ABR-Awareness V14 pipeline"
echo "Root=${ROOT_DIR}  seeds=${SEEDS}  num_envs=${NUM_ENVS}  ts_scale=${TS_SCALE}"
echo "=============================================================="

echo ""
echo "[0/4] Provenance + (optional) 5G stress traces"
"${PY}" src/paper/scripts/make_provenance_v14.py || true
# Regenerate the improved synthetic 5G stress suite (skip if you use real traces).
"${PY}" data/generate_5g_standardized_v14.py synth \
    --num 50 --length 300 \
    --out data/standardized/test_traces_5g_stress_v14 || true

echo ""
echo "[1/4] Training all V14 arms"
"${PY}" src/training/train_all_models_v14.py --all \
    --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}" --parallel "${PARALLEL}"

echo ""
echo "[2/4] Main multi-method evaluation (seed ${EVAL_SEED})"
"${PY}" src/evaluation/evaluate_all_models_v14.py \
    --episodes "${EPISODES}" --seed "${EVAL_SEED}"

echo ""
echo "[3/4] Shield sweep (isolation baseline) : broadband + 5G stress"
"${PY}" src/evaluation/eval_shield_vmaf_online_v14.py \
    --policy proposed_v14 --seed "${EVAL_SEED}" --episodes "${EPISODES}" \
    --trace-dir data/standardized/test_traces \
    --out results/v14_shielded_qoe/online_episodes.csv
"${PY}" src/evaluation/eval_shield_vmaf_online_v14.py \
    --policy proposed_v14 --seed "${EVAL_SEED}" --episodes "${EPISODES}" \
    --trace-dir data/standardized/test_traces_5g_stress_v14 \
    --out results/v14_5g_stress_shielded_qoe/online_episodes.csv

echo ""
echo "[4/4] Paired stats + conditional VMAF equivalence"
"${PY}" src/evaluation/analyze_v14_shielded_qoe.py \
    --episodes-csv results/v14_shielded_qoe/online_episodes.csv
"${PY}" src/evaluation/compute_vmaf_equivalence_v14.py \
    --decisions-csv results/v14_shielded_qoe/online_decisions.csv
"${PY}" src/evaluation/analyze_v14_shielded_qoe.py \
    --episodes-csv results/v14_5g_stress_shielded_qoe/online_episodes.csv || true
"${PY}" src/evaluation/compute_vmaf_equivalence_v14.py \
    --decisions-csv results/v14_5g_stress_shielded_qoe/online_decisions.csv || true

echo ""
echo "DONE. Key outputs under: ${ROOT_DIR}/results/"
echo "  detailed_stats_master_v14_seed${EVAL_SEED}.csv, summary_master_v14_seed${EVAL_SEED}.csv"
echo "  v14_shielded_qoe/{online_episodes,online_summary,online_decisions,paired_stats_v14,vmaf_equivalence_conditional_v14}.csv"
echo "  v14_5g_stress_shielded_qoe/*.csv , PROVENANCE_v14.json"
