#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=============================================================="
echo "ABR-Awareness V13: train + evaluate"
echo "Root: ${ROOT_DIR}"
echo "=============================================================="

PY="${PYTHON:-python}"
PARALLEL="${PARALLEL:-1}"
GPU_IDS="${GPU_IDS:-}"
MODELS="${MODELS:-proposed_v13,proposed_v13_guarded}"

echo ""
echo "[1/2] Training v13 models: ${MODELS}"
if [[ -n "${GPU_IDS}" ]]; then
  "${PY}" "${ROOT_DIR}/src/training/train_all_models_v13.py" --models "${MODELS}" --parallel "${PARALLEL}" --gpu-ids "${GPU_IDS}"
else
  "${PY}" "${ROOT_DIR}/src/training/train_all_models_v13.py" --models "${MODELS}" --parallel "${PARALLEL}"
fi

echo ""
echo "[2/2] Evaluating v13"
"${PY}" "${ROOT_DIR}/src/evaluation/run_dual_eval_v13.py"

echo ""
echo "DONE. Results under: ${ROOT_DIR}/results"
