"""
Quick V13 guard sweep.

This tests QoE-oriented guard parameters on the base policy without retraining.
Use this first; if a setting improves QoE while keeping rebuffer acceptable,
train `proposed_v13_guarded` with the same environment variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from evaluate_all_models_v13 import run_eval

PATHS = get_paths()


SWEEP = [
    {"name": "tight", "risk": 1.15, "stall": 0.10, "tp": 0.95, "down": 3},
    {"name": "balanced", "risk": 1.35, "stall": 0.25, "tp": 0.97, "down": 2},
    {"name": "qoe_hi", "risk": 1.55, "stall": 0.40, "tp": 1.00, "down": 1},
]


def _set_guard_env(cfg: dict):
    os.environ["ABR_SAFETY_GUARD"] = "0"
    os.environ["ABR_V13_RISK_RATIO"] = str(cfg["risk"])
    os.environ["ABR_V13_ALLOWED_STALL"] = str(cfg["stall"])
    os.environ["ABR_V13_TP_SCALE"] = str(cfg["tp"])
    os.environ["ABR_V13_MAX_DOWNGRADE"] = str(cfg["down"])


def main():
    summaries = []
    methods = ["Proposed_V13_Base", "Proposed_V13_SoftGuard", "Proposed_V13_TightGuard", "Pensieve_V13", "RobustMPC"]

    for cfg in SWEEP:
        _set_guard_env(cfg)
        suffix = f"_v13_sweep_{cfg['name']}"
        print("=" * 72)
        print(f"Running V13 sweep: {cfg}")
        print("=" * 72)
        df = run_eval(episodes_per_video=int(os.environ.get("ABR_V13_SWEEP_EPISODES", "20")), suffix=suffix, methods=methods)
        if df is None:
            continue
        grouped = df.groupby("Method").agg(QoE=("QoE", "mean"), Rebuffer=("Rebuffer", "mean"), VMAF=("VMAF", "mean"), Switch=("Switch", "mean")).reset_index()
        grouped.insert(0, "Config", cfg["name"])
        grouped.insert(1, "RiskRatio", cfg["risk"])
        grouped.insert(2, "AllowedStall", cfg["stall"])
        grouped.insert(3, "TPScale", cfg["tp"])
        grouped.insert(4, "MaxDowngrade", cfg["down"])
        summaries.append(grouped)

    if summaries:
        out = pd.concat(summaries, ignore_index=True)
        out_csv = PATHS["results"] / "summary_v13_guard_sweep.csv"
        out.to_csv(out_csv, index=False)
        print(f"\nSaved V13 sweep summary: {out_csv}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
