#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=============================================================="
echo "ABR-Awareness V12: train all + evaluate"
echo "Root: ${ROOT_DIR}"
echo "=============================================================="

PY="${PYTHON:-python}"
PARALLEL="${PARALLEL:-1}"
GPU_IDS="${GPU_IDS:-}"

echo ""
echo "[1/2] Training all v12 learning-based models"
if [[ -n "${GPU_IDS}" ]]; then
  "${PY}" "${ROOT_DIR}/src/training/train_all_models_v12.py" --all --parallel "${PARALLEL}" --gpu-ids "${GPU_IDS}"
else
  "${PY}" "${ROOT_DIR}/src/training/train_all_models_v12.py" --all --parallel "${PARALLEL}"
fi

echo ""
echo "[2/2] Evaluating v12 (policy)"
"${PY}" "${ROOT_DIR}/src/evaluation/run_dual_eval_v12.py"

echo ""
echo "DONE. Results under: ${ROOT_DIR}/results"

