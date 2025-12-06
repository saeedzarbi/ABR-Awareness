#!/bin/bash

# Script to run training and evaluation pipeline
# 1. Train PPO Multi-Dynamic
# 2. Train Pensieve Multi
# 3. Run Final Evaluation

set -e  # Exit on error

# Load environment variables if .env file exists
if [ -f ".env" ]; then
    source .env
fi

# Slack webhook URL (from environment variable)
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

# Function to send message to Slack
send_slack_message() {
    # Skip if webhook URL is not set
    if [ -z "$SLACK_WEBHOOK" ]; then
        return 0
    fi
    
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
            "footer": "ABR Training Pipeline",
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
echo "🚀 ABR Training and Evaluation Pipeline"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

send_slack_message "info" "Pipeline Started" "ABR Training and Evaluation pipeline has started"

# Step 1: Train PPO Multi-Dynamic
echo ""
echo "=========================================="
echo "📚 Step 1: Training PPO Multi-Dynamic"
echo "=========================================="
send_slack_message "info" "Step 1 Started" "Training PPO Multi-Dynamic model..."

if python3 train_ppo_multi_dynamic.py; then
    echo "✅ PPO Multi-Dynamic training completed!"
    send_slack_message "success" "Step 1 Completed" "PPO Multi-Dynamic training completed successfully!"
else
    ERROR_MSG="PPO Multi-Dynamic training failed with exit code $?"
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 1 Failed" "$ERROR_MSG"
    exit 1
fi

# Step 2: Train Pensieve Multi
echo ""
echo "=========================================="
echo "📚 Step 2: Training Pensieve Multi"
echo "=========================================="
send_slack_message "info" "Step 2 Started" "Training Pensieve Multi model..."

if python3 train_pensieve_multi.py; then
    echo "✅ Pensieve Multi training completed!"
    send_slack_message "success" "Step 2 Completed" "Pensieve Multi training completed successfully!"
else
    ERROR_MSG="Pensieve Multi training failed with exit code $?"
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 2 Failed" "$ERROR_MSG"
    exit 1
fi

# Step 3: Run Final Evaluation
echo ""
echo "=========================================="
echo "🔬 Step 3: Running Final Evaluation"
echo "=========================================="
send_slack_message "info" "Step 3 Started" "Running final evaluation..."

cd ../evaluation
if python3 final_multi.py; then
    cd ../training
    echo "✅ Final evaluation completed!"
    send_slack_message "success" "Step 3 Completed" "Final evaluation completed successfully!"
else
    ERROR_MSG="Final evaluation failed with exit code $?"
    cd ../training
    echo "❌ $ERROR_MSG"
    send_slack_message "error" "Step 3 Failed" "$ERROR_MSG"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 All steps completed successfully!"
echo "=========================================="
send_slack_message "success" "Pipeline Completed" "All steps completed successfully! 🎉"

