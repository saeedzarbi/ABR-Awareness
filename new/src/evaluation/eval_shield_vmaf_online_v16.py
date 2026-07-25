"""
Online shield-sweep evaluation (V16, per-chunk non-monotone VMAF ladder).

Thin wrapper over eval_shield_vmaf_online_v14 (single-process => safe to override
the env class and checkpoint tag on the v14 module here):
  * environment -> abr_multi_env_v16 (per-chunk multi-resolution VMAF)
  * checkpoints -> models/master_v16_perchunk/<folder>/seed_<s>/

This is THE experiment for the original hero claim: on the non-monotone per-chunk
ladder the `vmaf_aware` and `highest_feasible` arms can finally diverge, so the
paired `VMAF-aware vs highest-index [isolation]` test measures the real perceptual
value of VMAF-aware shielding. The per-chunk decision log records raw vs executed
VMAF for the conditional equivalence analysis.

If --out is omitted it defaults to results/v16_perchunk_shielded_qoe/.

Usage:
  cd new
  python src/evaluation/eval_shield_vmaf_online_v16.py \
      --policy proposed_v14 --seed 0 --episodes 20 \
      --trace-dir data/standardized/test_traces \
      --out results/v16_perchunk_shielded_qoe/online_episodes.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import src.evaluation.eval_shield_vmaf_online_v14 as s14
from src.environment.abr_multi_env_v16 import ABREnv as PerChunkEnv

s14.ABREnv = PerChunkEnv
s14.MODEL_TAG = "master_v16_perchunk"


def main():
    argv = sys.argv[1:]
    if "--out" not in argv:
        default_out = str(s14.PATHS["results"] / "v16_perchunk_shielded_qoe" / "online_episodes.csv")
        sys.argv += ["--out", default_out]
    s14.main()


if __name__ == "__main__":
    main()
