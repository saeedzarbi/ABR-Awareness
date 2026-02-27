#!/bin/bash

# V5 Training and Evaluation Pipeline
# - Constrained MDP with Lagrangian primal-dual optimization
# - Balanced reward weights (REBUF=6.0, LYAP_BETA=1.0, BUF_DEV=0.05)
# - Chunk-0 smooth penalty fix

set -e

if [ -f ".env" ]; then
    source .env
fi

SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

send_slack_message() {
    local status=$1
    local step=$2
    local message=$3
    
    local color="good"
    local emoji="✅"
    if [ "$status" = "error" ]; then
        color="danger"
        emoji="❌"
    elif [ "$status" = "info" ]; then
        color="#36a64f"
        emoji="ℹ️"
    fi
    
    local payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "$emoji $step",
            "text": "$message",
            "footer": "ABR Training Pipeline V5 (CMDP)",
            "ts": $(date +%s)
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
        --data "$payload" \
        "$SLACK_WEBHOOK" 2>/dev/null || echo "⚠️ Failed to send Slack notification"
}

echo "=========================================="
echo "🚀 ABR Training and Evaluation Pipeline V5"
echo "   (Constrained MDP + Lagrangian)"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

send_slack_message "info" "Pipeline V5 Started" "ABR Training V5 (CMDP + Lagrangian) pipeline has started"

# Step 1: Train all models with V5 script
echo ""
echo "=========================================="
echo "📚 Step 1: Training all models (V5)"
echo "=========================================="
send_slack_message "info" "Step 1 Started" "Training all models (Proposed with CMDP, Ablations, Pensieve)..."

if python3 train_all_models_v5.py --all --parallel 1; then
    echo "✅ All model training completed!"
    send_slack_message "success" "Step 1 Completed" "All V5 model training completed successfully!"
else
    ERROR_MSG="Model training failed with exit code $?"
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 1 Failed" "$ERROR_MSG"
    exit 1
fi

# Step 2: Run evaluation with V5 script
echo ""
echo "=========================================="
echo "🔬 Step 2: Running Final Evaluation (V5)"
echo "=========================================="
send_slack_message "info" "Step 2 Started" "Running V5 evaluation..."

cd ../evaluation
if python3 evaluate_all_models_v5.py; then
    cd ../training
    echo "✅ Final evaluation completed!"
    send_slack_message "success" "Step 2 Completed" "Final V5 evaluation completed successfully!"
else
    ERROR_MSG="Final evaluation failed with exit code $?"
    cd ../training
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 2 Failed" "$ERROR_MSG"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 V5 Pipeline completed successfully!"
echo "=========================================="
send_slack_message "success" "Pipeline V5 Completed" "All V5 steps completed successfully! 🎉"
