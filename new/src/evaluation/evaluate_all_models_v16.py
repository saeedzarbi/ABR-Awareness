"""
Master evaluation (V16, per-chunk non-monotone VMAF ladder).

Thin wrapper over evaluate_all_models_v14 (single-process, so overriding the env
class and model tag on the v14 module here is sufficient and safe):
  * environment  -> abr_multi_env_v16 (per-chunk multi-resolution VMAF)
  * checkpoints  -> models/master_v16_perchunk/<folder>/seed_<s>/
  * output suffix -> _v16_perchunk_seed<seed>

Usage:
  cd new
  python src/evaluation/evaluate_all_models_v16.py --seed 0 --episodes 20
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import src.evaluation.evaluate_all_models_v14 as e14
from src.environment.abr_multi_env_v16 import ABREnv as PerChunkEnv

e14.ABREnv = PerChunkEnv
e14.MODEL_TAG = "master_v16_perchunk"


def main():
    parser = argparse.ArgumentParser(description="Evaluate all ABR models (v16, per-chunk VMAF).")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Training-seed checkpoint to evaluate.")
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    suffix = args.suffix if args.suffix is not None else f"_v16_perchunk_seed{args.seed}"
    e14.run_eval(episodes_per_video=args.episodes, seed=args.seed, suffix=suffix)


if __name__ == "__main__":
    main()
