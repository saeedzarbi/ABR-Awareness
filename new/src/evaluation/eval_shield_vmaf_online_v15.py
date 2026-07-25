"""
Online shield-sweep evaluation (V15, low-latency operating point).

Thin wrapper over eval_shield_vmaf_online_v14: the sweep is single-process, so
overriding the environment class and checkpoint tag on the v14 module here is
sufficient and safe.

  * environment -> abr_multi_env_v15 (buffer cap 6 s)
  * checkpoints -> models/master_v15_lowlat/<folder>/seed_<s>/

If --out is not supplied it defaults to results/v15_lowlat_shielded_qoe/... so
results never silently land in the v14 folder.

Usage:
  cd new
  python src/evaluation/eval_shield_vmaf_online_v15.py \
      --policy proposed_v14 --seed 0 --episodes 20 \
      --trace-dir data/standardized/test_traces \
      --out results/v15_lowlat_shielded_qoe/online_episodes.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import src.evaluation.eval_shield_vmaf_online_v14 as s14
from src.environment.abr_multi_env_v15 import ABREnv as LowLatEnv

s14.ABREnv = LowLatEnv
s14.MODEL_TAG = "master_v15_lowlat"


def main():
    argv = sys.argv[1:]
    if "--out" not in argv:
        default_out = str(s14.PATHS["results"] / "v15_lowlat_shielded_qoe" / "online_episodes.csv")
        sys.argv += ["--out", default_out]
    s14.main()


if __name__ == "__main__":
    main()
