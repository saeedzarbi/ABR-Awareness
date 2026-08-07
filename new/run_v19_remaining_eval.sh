#!/usr/bin/env bash
# =============================================================================
# Re-run CPS eval hosts that may still reflect the pre-v19 (4-video) roster.
#
# After the 12-video pipeline, greedy/BBA/bb/improved/proposed were refreshed;
# BOLA, RobustMPC, and co-design (proposed_cps_5g) may still be stale.
#
# Usage (on Linux server, from new/):
#   bash run_v19_remaining_eval.sh
#
# Options:
#   PYTHON=python3          python binary
#   REGEN_PAPER=1           run make_cps_paper_assets.py after eval (default 1)
#   REDO_CODESIGN=1         also re-eval proposed_cps_5g if checkpoint exists
#   EPISODES=204            override episode count (default: configs.videos.CPS_EPISODES)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3
    elif command -v python  >/dev/null 2>&1; then PY=python
    else PY=python3
    fi
fi

OUT="results/v18_certified"
TRACE_5G="data/standardized/test_traces_5g_v18"
EPISODES="${EPISODES:-$("${PY}" -c "from configs.videos import CPS_EPISODES; print(CPS_EPISODES)" 2>/dev/null || echo 204)}"
EPSILON="${EPSILON:-1.0}"
ALPHA="${ALPHA:-0.10}"
REGEN_PAPER="${REGEN_PAPER:-1}"
REDO_CODESIGN="${REDO_CODESIGN:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${OUT}"

echo "=============================================================="
echo "v19 remaining CPS eval  episodes=${EPISODES}  eps=${EPSILON}  alpha=${ALPHA}"
echo "=============================================================="

_run() {
    local label="$1"
    shift
    echo ""
    echo "[*] ${label}"
    "$@"
}

_run "BOLA on 5G" \
    "${PY}" src/evaluation/eval_certified_shield_v18.py \
        --policy bola --episodes "${EPISODES}" \
        --epsilon "${EPSILON}" --alpha "${ALPHA}" \
        --trace-dir "${TRACE_5G}" --buffer 12 \
        --out "${OUT}/bola_5g"

_run "RobustMPC on 5G" \
    "${PY}" src/evaluation/eval_certified_shield_v18.py \
        --policy mpc --episodes "${EPISODES}" \
        --epsilon "${EPSILON}" --alpha "${ALPHA}" \
        --trace-dir "${TRACE_5G}" --buffer 12 \
        --out "${OUT}/mpc_5g"

if [ "${REDO_CODESIGN}" = "1" ]; then
    TAG_CODESIGN="master_v18_5gtrain"
    SEED0="${SEEDS:-0}"
    SEED0="${SEED0%%,*}"
    CKPT_PROP_CPS="${ROOT_DIR}/results/models/${TAG_CODESIGN}/proposed_cps_v18/seed_${SEED0}/final_model"
    if [ -f "${CKPT_PROP_CPS}.zip" ]; then
        _run "Co-design: proposed_cps_5g (IMPROVED shield)" \
            "${PY}" src/evaluation/eval_certified_shield_v18.py \
                --policy ppo --ckpt "${CKPT_PROP_CPS}" --episodes "${EPISODES}" \
                --epsilon "${EPSILON}" --alpha "${ALPHA}" \
                --trace-dir "${TRACE_5G}" --buffer 12 \
                --predictive --forecast \
                --out "${OUT}/proposed_cps_5g"
    else
        echo "[WARN] Skip proposed_cps_5g: missing ${CKPT_PROP_CPS}.zip"
        echo "       Run FULL=1 training first, or set REDO_CODESIGN=0."
    fi
fi

if [ "${REGEN_PAPER}" = "1" ]; then
    echo ""
    echo "[*] Regenerate paper tables/figures/macros"
    "${PY}" src/paper/make_cps_paper_assets.py
fi

echo ""
echo "=============================================================="
echo "DONE. Inspect:"
echo "  ${OUT}/bola_5g/summary.json"
echo "  ${OUT}/mpc_5g/summary.json"
echo "  src/paper/overleaf_upload/tables/macros_cps.tex"
echo "=============================================================="
