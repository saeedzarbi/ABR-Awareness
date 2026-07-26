#!/usr/bin/env bash
# =============================================================================
# ABR-Awareness V18 : Certified Perceptual Shield
# -----------------------------------------------------------------------------
# Novel claim (model-agnostic runtime shield):
#   (1) VMAF-knee bandwidth banking  -- perceptually-lossless downshift on
#       saturated chunks (e.g. 6000->2850 buys ~0 VMAF, ~2x fewer bytes).
#   (2) Conformal throughput lower bound -- distribution-free coverage 1-alpha.
#   (3) Certified feasibility under that bound -> rebuffering guarantee.
#
# Unlike the V12-V16 "VMAF-aware ranking" shield (proven inert on monotone
# ladders), this exploits RATE-LADDER SATURATION, which the data genuinely has.
#
# Quick (no training; greedy + BBA, real + 5G traces):
#   cd new
#   bash src/training/run_certified_v18.sh
#
# Full (train proposed+pensieve overnight, then evaluate with the CPS):
#   FULL=1 NUM_ENVS=8 SEEDS=0 bash src/training/run_certified_v18.sh
#
# Env vars: PYTHON, SEEDS, NUM_ENVS, TS_SCALE, EPISODES, EPSILON, ALPHA, FULL
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0}"
NUM_ENVS="${NUM_ENVS:-8}"
TS_SCALE="${TS_SCALE:-1.0}"
EPISODES="${EPISODES:-200}"
EPSILON="${EPSILON:-1.0}"
ALPHA="${ALPHA:-0.10}"
FULL="${FULL:-0}"

OUT="results/v18_certified"
TRACE_5G="data/standardized/test_traces_5g_v18"
TRACE_BB="data/standardized/test_traces"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=============================================================="
echo "V18 Certified Perceptual Shield"
echo "Root=${ROOT_DIR}  eps=${EPSILON}  alpha=${ALPHA}  FULL=${FULL}"
echo "=============================================================="

mkdir -p "${OUT}"

# ---- [0] Torch-free PoC (isolates banking vs safety-only) ----
echo ""
echo "[0] PoC: bitrate-greedy under RAW / SAFETY / CERTIFIED"
"${PY}" _poc_v18.py | tee "${OUT}/poc_console.txt"

# ---- [1] Generate moderate 5G-like traces (saturated rungs feasible) ----
echo ""
echo "[1] Generate 5G-like evaluation traces (median ~9 Mbps + dips)"
mkdir -p "${TRACE_5G}"
"${PY}" - <<'PY'
import json, numpy as np
from pathlib import Path
out = Path("data/standardized/test_traces_5g_v18")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
for i in range(50):
    tp = np.full(1000, 9000.0)
    for start in range(20 + (i % 17), 1000, 40):
        tp[start:start + 4] = float(rng.uniform(500.0, 1200.0))
    tp *= rng.uniform(0.85, 1.15, size=tp.shape)
    tp = np.clip(tp, 80.0, 50000.0)
    (out / f"5g_v18_{i:03d}.json").write_text(
        json.dumps({"throughput_kbps": tp.tolist()}), encoding="utf-8")
print(f"wrote 50 traces -> {out}")
PY

# ---- [2] Model-agnostic eval: GREEDY (no training needed) ----
echo ""
echo "[2] Eval GREEDY on 5G traces (the adversarial / bitrate-hungry case)"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy greedy --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/greedy_5g"

echo ""
echo "[3] Eval GREEDY on broadband test traces"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy greedy --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_BB}" --buffer 12 \
    --out "${OUT}/greedy_bb"

echo ""
echo "[4] Eval BBA on 5G traces"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy bba --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/bba_5g"

echo ""
echo "=============================================================="
echo "QUICK SIGNAL READY. Inspect:"
echo "  ${OUT}/greedy_5g/summary.json"
echo "  ${OUT}/bba_5g/summary.json"
echo "Success criteria for the novel claim:"
echo "  * CERTIFIED vs SAFETY: bandwidth DOWN, VMAF TOST within +/-eps"
echo "  * CERTIFIED vs RAW:    rebuffer DOWN (p<0.05), conformal cover ~ 1-alpha"
echo "=============================================================="

if [ "${FULL}" != "1" ]; then
    echo ""
    echo "QUICK mode done. Re-run with FULL=1 to train proposed+pensieve overnight"
    echo "and evaluate the CPS on those RL policies."
    exit 0
fi

# ---- FULL: train RL policies (NO shield during training; applied at eval) ----
echo ""
echo "[5] FULL: train proposed + pensieve on v18 env"
"${PY}" src/training/train_all_models_v18.py \
    --models proposed,pensieve --seeds "${SEEDS}" --num-envs "${NUM_ENVS}" \
    --timesteps-scale "${TS_SCALE}"

# SB3 save() writes final_model.zip under results/models (see configs/paths.py).
# Pass path without .zip — PPO.load appends it. Prefer best_model if present.
_resolve_ckpt() {
    local base="$1"
    if [ -f "${base}/best_model/best_model.zip" ]; then
        echo "${base}/best_model/best_model"
    elif [ -f "${base}/final_model.zip" ]; then
        echo "${base}/final_model"
    else
        echo "${base}/final_model"
    fi
}
CKPT_PROP="$(_resolve_ckpt "results/models/master_v18_5g/proposed_v14/seed_${SEEDS%%,*}")"
CKPT_PEN="$(_resolve_ckpt "results/models/master_v18_5g/pensieve_v14/seed_${SEEDS%%,*}")"
echo "CKPT proposed=${CKPT_PROP}"
echo "CKPT pensieve=${CKPT_PEN}"
for _c in "${CKPT_PROP}" "${CKPT_PEN}"; do
    if [ ! -f "${_c}.zip" ]; then
        echo "ERROR: missing checkpoint ${_c}.zip (did FULL training finish?)"
        exit 1
    fi
done

echo ""
echo "[6] FULL: CPS eval on proposed"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy ppo --ckpt "${CKPT_PROP}" --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/proposed_5g"

echo ""
echo "[7] FULL: CPS eval on pensieve (content-blind)"
"${PY}" src/evaluation/eval_certified_shield_v18.py \
    --policy ppo --ckpt "${CKPT_PEN}" --blind --episodes "${EPISODES}" \
    --epsilon "${EPSILON}" --alpha "${ALPHA}" \
    --trace-dir "${TRACE_5G}" --buffer 12 \
    --out "${OUT}/pensieve_5g"

echo ""
echo "DONE (FULL). Key outputs under ${OUT}/"
