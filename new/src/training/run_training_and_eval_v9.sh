#!/bin/bash

# V9 Training and Evaluation Pipeline (paper-ready)
# - Train ALL RL models (including Proposed_Shielded) to master_v9
# - Run evaluation suite (policy-only + system light/strong)
# - Optional Slack notifications via SLACK_WEBHOOK_URL in .env

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
            "footer": "ABR Training Pipeline V9 (CMDP + Shielded RL)",
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
echo "🚀 ABR Training and Evaluation Pipeline V9"
echo "   (CMDP + Shielded RL, paper-ready)"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

send_slack_message "info" "Pipeline V9 Started" "ABR Training + Evaluation V9 pipeline has started"

# Step 1: Train ALL v9 models (master_v9)
echo ""
echo "=========================================="
echo "📚 Step 1: Training ALL models (V9)"
echo "=========================================="
send_slack_message "info" "Step 1 Started" "Training all v9 models (proposed, proposed_shielded, ablations, pensieve)..."

if python train_all_models_v9.py --all; then
    echo "✅ All v9 trainings completed!"
    send_slack_message "success" "Step 1 Completed" "All v9 trainings completed successfully!"
else
    ERROR_MSG="V9 training failed with exit code $?"
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 1 Failed" "$ERROR_MSG"
    exit 1
fi

# Step 2: Run v9 evaluation suite (policy-only + system light/strong)
echo ""
echo "=========================================="
echo "🔬 Step 2: Running Final Evaluation (V9)"
echo "=========================================="
send_slack_message "info" "Step 2 Started" "Running v9 evaluation suite (policy-only, system light, system strong)..."

cd ../evaluation
if python run_dual_eval_v9.py; then
    cd ../training
    echo "✅ Final v9 evaluation completed!"
    send_slack_message "success" "Step 2 Completed" "Final v9 evaluation completed successfully!"
else
    ERROR_MSG="V9 evaluation failed with exit code $?"
    cd ../training
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 2 Failed" "$ERROR_MSG"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 V9 Pipeline completed successfully!"
echo "=========================================="
send_slack_message "success" "Pipeline V9 Completed" "All v9 steps (train + eval) completed successfully! 🎉"

