"""
Master evaluation (V15, low-latency operating point).

Thin wrapper over evaluate_all_models_v14: the evaluation logic is single-process
(no SubprocVecEnv), so overriding the environment class and the model tag on the
v14 module in this process is sufficient and safe.

  * environment  -> abr_multi_env_v15 (buffer cap 6 s)
  * checkpoints  -> models/master_v15_lowlat/<folder>/seed_<s>/
  * output suffix -> _v15_lowlat_seed<seed>

Usage:
  cd new
  python src/evaluation/evaluate_all_models_v15.py --seed 0 --episodes 20
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import src.evaluation.evaluate_all_models_v14 as e14
from src.environment.abr_multi_env_v15 import ABREnv as LowLatEnv

# Redirect the environment and checkpoint tag used by every helper in e14.
e14.ABREnv = LowLatEnv
e14.MODEL_TAG = "master_v15_lowlat"


def main():
    parser = argparse.ArgumentParser(description="Evaluate all ABR models (v15, low-latency).")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Training-seed checkpoint to evaluate.")
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    suffix = args.suffix if args.suffix is not None else f"_v15_lowlat_seed{args.seed}"
    e14.run_eval(episodes_per_video=args.episodes, seed=args.seed, suffix=suffix)


if __name__ == "__main__":
    main()
