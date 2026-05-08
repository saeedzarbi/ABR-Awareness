"""Paired Wilcoxon analysis for v123_shielded_qoe.

Headline question: does VMAF-aware projection strictly Pareto-dominate the
legacy index-decrement shield, on the *same* trained policy and the *same*
trace seeds?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
EPISODES_CSV = ROOT / "results" / "v123_shielded_qoe" / "online_episodes.csv"
OUT_CSV = ROOT / "results" / "v123_shielded_qoe" / "paired_wilcoxon.csv"

ANCHORS = [
    "shield_legacy",
    "shield_off",
]
TARGETS = [
    "vmaf_aware_tol1.0_bud08",
    "vmaf_aware_tol0.8_bud08",
    "vmaf_aware_tol1.2_bud08",
    "vmaf_aware_tol1.5_bud08",
    "lookahead_h3_mb1.0_vmafFB",
    "thresh_cat3.0_vmafFB",
    "thresh_cat4.0_vmafFB",
    "thresh_cat3.0",
    "thresh_cat4.0",
    "thresh_cat5.0",
]


def paired(df: pd.DataFrame, target: str, anchor: str) -> dict:
    a = (
        df[df.Method == anchor]
        .sort_values(["Video", "Episode"])
        .reset_index(drop=True)
    )
    b = (
        df[df.Method == target]
        .sort_values(["Video", "Episode"])
        .reset_index(drop=True)
    )
    if len(a) != len(b) or len(a) == 0:
        return {}
    out: dict = {"target": target, "anchor": anchor, "n": len(a)}
    for metric in ("QoE", "Rebuffer", "VMAF"):
        delta = b[metric].to_numpy() - a[metric].to_numpy()
        out[f"mean_{metric}_target"] = float(b[metric].mean())
        out[f"mean_{metric}_anchor"] = float(a[metric].mean())
        out[f"delta_mean_{metric}"] = float(delta.mean())
        out[f"delta_median_{metric}"] = float(np.median(delta))
        try:
            stat, p = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            out[f"wilcoxon_p_{metric}"] = float(p)
        except ValueError:
            out[f"wilcoxon_p_{metric}"] = float("nan")
    return out


def main() -> None:
    df = pd.read_csv(EPISODES_CSV)
    rows: list[dict] = []
    for tgt in TARGETS:
        for anc in ANCHORS:
            rec = paired(df, tgt, anc)
            if rec:
                rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
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
