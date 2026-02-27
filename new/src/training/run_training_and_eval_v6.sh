#!/bin/bash

# V6 Training and Evaluation Pipeline
# - Constrained MDP with Lagrangian primal-dual optimization (V6 tuning)
# - Environment V6 (relaxed Lyapunov / buffer dev / rebuffer weights)
# - Safety guard with raw / light / strong modes

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
    
    if [ -z "$SLACK_WEBHOOK" ]; then
        echo "[SLACK-$status][$step] $message"
        return
    fi

    local payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "$emoji $step",
            "text": "$message",
            "footer": "ABR Training Pipeline V6 (CMDP)",
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
echo "🚀 ABR Training and Evaluation Pipeline V6"
echo "   (Constrained MDP + Lagrangian tuning)"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

send_slack_message "info" "Pipeline V6 Started" "ABR Training V6 (CMDP + Lagrangian tuning) pipeline has started"

# Step 1: Train Proposed-V6 model
echo ""
echo "=========================================="
echo "📚 Step 1: Training Proposed model (V6)"
echo "=========================================="
send_slack_message "info" "Step 1 Started" "Training Proposed model with V6 environment and CMDP tuning..."

if python3 train_all_models_v6.py --models proposed; then
    echo "✅ Proposed V6 training completed!"
    send_slack_message "success" "Step 1 Completed" "Proposed V6 training completed successfully!"
else
    ERROR_MSG="V6 model training failed with exit code $?"
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 1 Failed" "$ERROR_MSG"
    exit 1
fi

# Step 2: Run evaluation with V6 script (raw + light + strong guard)
echo ""
echo "=========================================="
echo "🔬 Step 2: Running Final Evaluation (V6)"
echo "=========================================="
send_slack_message "info" "Step 2 Started" "Running V6 evaluation (raw, light-guard, strong-guard)..."

cd ../evaluation
if python3 run_dual_eval_v6.py; then
    cd ../training
    echo "✅ Final V6 evaluation completed!"
    send_slack_message "success" "Step 2 Completed" "Final V6 evaluation (raw/light/strong) completed successfully!"
else
    ERROR_MSG="Final V6 evaluation failed with exit code $?"
    cd ../training
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 2 Failed" "$ERROR_MSG"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 V6 Pipeline completed successfully!"
echo "=========================================="
send_slack_message "success" "Pipeline V6 Completed" "All V6 steps (train + eval) completed successfully! 🎉"

