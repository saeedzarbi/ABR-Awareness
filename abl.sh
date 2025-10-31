#!/bin/bash

mkdir -p results/ablation_logs

echo "policy,avg_reward" > results/ablation_logs/summary.csv

run_exp() {
  NAME=$1
  shift
  echo "\n🚀 Running $NAME..."
  OUT=$(python train_ablation.py "$@" | tee results/ablation_logs/${NAME}.log)
  REWARD=$(echo "$OUT" | grep "Best Val Reward" | awk '{print $5}')
  echo "$NAME,$REWARD" >> results/ablation_logs/summary.csv
}

# Run all ablation configurations
run_exp full
run_exp no_content --no-content
run_exp no_vmaf --no-vmaf
run_exp pensieve_like --no-content --no-vmaf

echo "\n✅ Done. Summary saved to results/ablation_logs/summary.csv"
