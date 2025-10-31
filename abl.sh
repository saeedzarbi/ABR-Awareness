#!/bin/bash
set -e

echo "========================================"
echo "🚀 Starting Real Ablation Experiments"
echo "========================================"

mkdir -p results/ablation_real_logs

# 1️⃣ Full (Default)
echo -e "\n🧠 Training FULL model (content + VMAF)"
python3 train_ablation.py --type all | tee results/ablation_real_logs/full.log

# 2️⃣ No-SITI (بدون ویژگی‌های محتوا)
echo -e "\n🧩 Training NO-SITI model (no content features)"
python3 train_ablation.py --type no_siti | tee results/ablation_real_logs/no_siti.log

# 3️⃣ No-VMAF (بدون پاداش ادراکی)
echo -e "\n🎞️ Training NO-VMAF model (bitrate-only reward)"
python3 train_ablation.py --type no_vmaf | tee results/ablation_real_logs/no_vmaf.log

# 4️⃣ Network-Only baseline (شبیه Pensieve)
echo -e "\n📶 Training NETWORK-ONLY model (no content, no VMAF)"
python3 train_ablation.py --type network_only | tee results/ablation_real_logs/network_only.log

echo -e "\n✅ All ablation experiments finished!"
echo "Logs are saved in results/ablation_real_logs/"
echo "Each trained model checkpoint is inside results/ablation_<type>/"
echo "========================================"
