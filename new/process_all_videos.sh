#!/bin/bash

set -e  # Stop on any error

echo "========================================"
echo "ABR Pipeline: Encoding → Training"
echo "========================================"
echo ""

# Create logs directory
mkdir -p logs

# Show downloaded videos
echo "Downloaded videos:"
ls -lh data/raw_videos/
echo ""

# ================================
# Step 1: Encode Videos
# ================================
echo "Step 1/5: Encoding videos..."
echo "  This will create 6 bitrate versions of each video"
echo "  Estimated time: 30-60 minutes"
echo ""

python3 src/data_preparation/video_encoder.py > logs/encode.log 2>&1
echo "✓ Encoding complete"
echo ""

# Show encoded results
echo "Encoded videos:"
ls -lh data/encoded_videos/
echo ""

# ================================
# Step 2: Calculate VMAF
# ================================
echo "Step 2/5: Calculating VMAF scores..."
echo "  This compares each encoded version to the original"
echo "  Estimated time: 1-2 hours"
echo ""

python3 src/data_preparation/vmaf_calculator.py > logs/vmaf.log 2>&1
echo "✓ VMAF calculation complete"
echo ""

# Show VMAF summary
if [ -f "data/vmaf_scores/vmaf_summary.csv" ]; then
    echo "VMAF Summary:"
    cat data/vmaf_scores/vmaf_summary.csv | head -20
    echo ""
fi

# ================================
# Step 3: Extract SI/TI
# ================================
echo "Step 3/5: Extracting SI/TI features..."
echo "  This analyzes spatial and temporal complexity"
echo "  Estimated time: 20-30 minutes"
echo ""

python3 src/data_preparation/si_ti_extractor.py > logs/siti.log 2>&1
echo "✓ SI/TI extraction complete"
echo ""

# Show SI/TI summary
if [ -f "data/content_features/siti_summary.csv" ]; then
    echo "SI/TI Summary:"
    cat data/content_features/siti_summary.csv
    echo ""
fi

# ================================
# Step 4: Verification
# ================================
echo "Step 4/5: Data verification..."
echo ""

num_videos=$(ls data/encoded_videos/ 2>/dev/null | wc -l)
num_vmaf=$(find data/vmaf_scores -name "*.json" 2>/dev/null | wc -l)
num_siti=$(find data/content_features -name "*_siti.json" 2>/dev/null | wc -l)

echo "  ✓ Encoded video folders: $num_videos"
echo "  ✓ VMAF score files: $num_vmaf"
echo "  ✓ SI/TI feature files: $num_siti"
echo ""

if [ $num_videos -eq 0 ] || [ $num_vmaf -eq 0 ] || [ $num_siti -eq 0 ]; then
    echo "✗ Error: Data preparation incomplete!"
    echo "  Check logs in logs/ directory"
    exit 1
fi

echo "✓ All data prepared successfully"
echo ""

# ================================
# Step 5: Training
# ================================
echo "Step 5/5: Starting PPO V3 Training..."
echo ""
echo "Training Configuration:"
echo "  - Algorithm: PPO (Proximal Policy Optimization)"
echo "  - Reward: Balanced (Quality × 2.0, Rebuffer × 6.0)"
echo "  - Total timesteps: 600,000"
echo "  - Parallel environments: 8"
echo "  - Estimated time: 5-7 hours"
echo ""

echo "Training started at: $(date)"
echo "Logs will be saved to: logs/training.log"
echo ""

sleep 3

python3 src/training/train_ppo_v3_balanced.py 2>&1 | tee logs/training.log

# ================================
# Complete
# ================================
echo ""
echo "========================================"
echo "✓ Pipeline Complete!"
echo "========================================"
echo ""
echo "Training finished at: $(date)"
echo ""

# Show final model location
if [ -d "results/models/ppo_abr_v3/best_model" ]; then
    echo "✓ Best model saved at:"
    echo "  results/models/ppo_abr_v3/best_model/"
    echo ""
fi

echo "Next steps:"
echo ""
echo "1. Quick evaluation:"
echo "   python3 src/evaluation/quick_eval.py \\"
echo "     --model results/models/ppo_abr_v3/best_model/best_model \\"
echo "     --compare --episodes 20"
echo ""
echo "2. Compare all versions (V1, V2, V3, BBA):"
echo "   python3 src/evaluation/compare_versions.py"
echo ""
echo "3. View training curves:"
echo "   tensorboard --logdir results/logs/ppo_abr_v3"
echo ""

echo "All logs saved in: logs/"
echo "========================================"