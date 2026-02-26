#!/bin/bash

# Script to run training and evaluation pipeline V4
# 1. Train all models (Proposed + Ablations + Pensieve)
# 2. Run Final Evaluation (per-model env config, VBR-aware baselines)

set -e  # Exit on error

# Load environment variables if .env file exists
if [ -f ".env" ]; then
    source .env
fi

# Slack webhook URL (from environment variable)
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

# Function to send message to Slack
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
            "footer": "ABR Training Pipeline V4",
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
echo "🚀 ABR Training and Evaluation Pipeline V4"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

send_slack_message "info" "Pipeline V4 Started" "ABR Training and Evaluation V4 pipeline has started"

# Step 1: Train all models with V4 script
echo ""
echo "=========================================="
echo "📚 Step 1: Training initial models (V4)"
echo "=========================================="
send_slack_message "info" "Step 1 Started" "Training initial models (Proposed, Ablation_Base, Pensieve) on CPU..."

if python3 train_all_models_v4.py --models proposed,ablation_base,pensieve --parallel 1; then
    echo "✅ Initial model training completed!"
    send_slack_message "success" "Step 1 Completed" "Initial model training completed successfully!"
else
    ERROR_MSG="Model training failed with exit code $?"
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 1 Failed" "$ERROR_MSG"
    exit 1
fi

# Step 2: Run evaluation with V4 script
echo ""
echo "=========================================="
echo "🔬 Step 2: Running Final Evaluation (V4)"
echo "=========================================="
send_slack_message "info" "Step 2 Started" "Running V4 evaluation (per-model env, VBR-aware baselines)..."

cd ../evaluation
if python3 evaluate_all_models_v4.py; then
    cd ../training
    echo "✅ Final evaluation completed!"
    send_slack_message "success" "Step 2 Completed" "Final V4 evaluation completed successfully!"
else
    ERROR_MSG="Final evaluation failed with exit code $?"
    cd ../training
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 2 Failed" "$ERROR_MSG"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 All steps completed successfully!"
echo "=========================================="
send_slack_message "success" "Pipeline V4 Completed" "All V4 steps completed successfully! 🎉"
