#!/bin/bash

# V11 Training and Evaluation Pipeline
# - Train v11 methods (including Proposed_ShieldedQoE) to master_v11
# - Run v11 evaluation (policy-only)

set -e

echo "=========================================="
echo "🚀 ABR Training and Evaluation Pipeline V11"
echo "   (Shield-aware + Hysteresis)"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "📚 Step 1: Training ALL models (V11)"
echo "=========================================="
python train_all_models_v11.py --all

echo ""
echo "=========================================="
echo "🔬 Step 2: Running Evaluation (V11)"
echo "=========================================="
cd ../evaluation
python run_dual_eval_v11.py

echo ""
echo "=========================================="
echo "🎉 V11 Pipeline completed successfully!"
echo "=========================================="

