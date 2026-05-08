"""Paired Wilcoxon analysis for v5g_shielded_qoe.

This mirrors analyze_v123_shielded_qoe.py but reads:
  results/v5g_shielded_qoe/online_episodes.csv

Outputs:
  results/v5g_shielded_qoe/paired_wilcoxon.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
EPISODES_CSV = ROOT / "results" / "v5g_stress_shielded_qoe" / "online_episodes.csv"
OUT_CSV = ROOT / "results" / "v5g_stress_shielded_qoe" / "paired_wilcoxon.csv"

ANCHORS = ["shield_legacy", "shield_off"]
TARGETS = [
    "vmaf_aware_tol1.0_bud08",
    "vmaf_aware_tol0.8_bud08",
    "thresh_cat3.0_vmafFB",
    "thresh_cat4.0_vmafFB",
]


def _paired(df: pd.DataFrame, target: str, anchor: str) -> dict:
    a = df[df.Method == anchor].sort_values(["Video", "Episode"]).reset_index(drop=True)
    b = df[df.Method == target].sort_values(["Video", "Episode"]).reset_index(drop=True)
    if len(a) != len(b) or len(a) == 0:
        return {}

    out: dict = {"target": target, "anchor": anchor, "n": int(len(a))}
    for metric in ("QoE", "Rebuffer", "VMAF"):
        delta = b[metric].to_numpy(dtype=float) - a[metric].to_numpy(dtype=float)
        out[f"delta_mean_{metric}"] = float(delta.mean())
        out[f"delta_median_{metric}"] = float(np.median(delta))
        try:
            _, p = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            out[f"wilcoxon_p_{metric}"] = float(p)
        except ValueError:
            out[f"wilcoxon_p_{metric}"] = float("nan")
    return out


def main() -> None:
    df = pd.read_csv(EPISODES_CSV)
    rows: list[dict] = []
    for tgt in TARGETS:
        for anc in ANCHORS:
            rec = _paired(df, tgt, anc)
            if rec:
                rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")

    cols = [
        "target",
        "anchor",
        "n",
        "delta_mean_QoE",
        "wilcoxon_p_QoE",
        "delta_mean_Rebuffer",
        "wilcoxon_p_Rebuffer",
        "delta_mean_VMAF",
        "wilcoxon_p_VMAF",
    ]
    print(out[cols].to_string(index=False))
    print(f"\nWrote: {OUT_CSV}")


if __name__ == "__main__":
    main()

