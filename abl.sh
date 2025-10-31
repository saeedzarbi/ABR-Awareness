#!/bin/bash
set -e

mkdir -p results/ablation_runs

run(){
  NAME=$1; shift
  echo -e "\n🚀 Running $NAME..."
  LOG=results/ablation_runs/${NAME}.log
  python3 train_ablation.py --tag $NAME "$@" 2>&1 | tee "$LOG"
}

# 4 configs
run full
run no_content --no-content
run no_vmaf --no-vmaf
run pensieve_like --no-content --no-vmaf

# Live stats after runs
python3 ablation_stats_live.py

echo -e "\n✅ All done. See results in results/ablation_runs/"
