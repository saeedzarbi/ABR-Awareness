#!/bin/bash

echo "=========================================="
echo "🚀 ABR Improvement Pipeline"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Start PPO V3 Training (background)
echo -e "${YELLOW}Step 1: Starting PPO V3 Training (background)...${NC}"
python3 src/training/train_ppo_v3_balanced.py > logs/training_v3.log 2>&1 &
TRAIN_PID=$!
echo -e "${GREEN}✓ Training started (PID: $TRAIN_PID)${NC}"
echo "  Monitor: tail -f logs/training_v3.log"
echo ""

sleep 5  # Give it time to start

# Step 2: Process new videos (parallel)
echo -e "${YELLOW}Step 2: Processing new videos...${NC}"
echo ""

echo "  2a. Encoding videos..."
python3 src/data_preparation/video_encoder.py > logs/encoding.log 2>&1
echo -e "${GREEN}  ✓ Encoding done${NC}"

echo "  2b. Calculating VMAF..."
python3 src/data_preparation/vmaf_calculator.py > logs/vmaf.log 2>&1
echo -e "${GREEN}  ✓ VMAF done${NC}"

echo "  2c. Extracting SI/TI..."
python3 src/data_preparation/si_ti_extractor.py > logs/siti.log 2>&1
echo -e "${GREEN}  ✓ SI/TI done${NC}"

echo ""
echo -e "${GREEN}✓ Video processing complete!${NC}"
echo ""

# Step 3: Check training status
echo -e "${YELLOW}Step 3: Training Status${NC}"
if ps -p $TRAIN_PID > /dev/null; then
   echo -e "${GREEN}  ✓ Training is running (PID: $TRAIN_PID)${NC}"
   echo "  Logs: tail -f logs/training_v3.log"
else
   echo -e "${YELLOW}  ⚠ Training finished or stopped${NC}"
fi

echo ""
echo "=========================================="
echo "Pipeline Status:"
echo "  ✓ Videos processed"
echo "  ⏳ Training in progress..."
echo ""
echo "Check training progress:"
echo "  tail -f logs/training_v3.log"
echo ""
echo "After training completes, evaluate:"
echo "  python3 src/evaluation/compare_versions.py"
echo "=========================================="