"""
Paired statistical analysis for the V14 shield sweep (reviewer-response).

Fixes applied vs. the v12/v123 analysis:
* P0.4 -- report the EFFECTIVE sample size (number of non-zero paired
  differences) alongside n, and use ``zero_method="zsplit"`` so the reported n
  and the test's actual n are consistent. The old code used
  ``zero_method="wilcox"`` (drops zeros) while printing n=80.
* Multiplicity -- Holm-adjusted p-values within each metric family.
* F2 isolation -- an explicit A/B between the VMAF-aware arm and the
  highest-feasible-index arm on the *identical* soft-safe set. On a monotone
  ladder every paired difference should be exactly zero (n_nonzero == 0),
  turning the theoretical inertness argument into a measured result.

Reads : results/v14_shielded_qoe/online_episodes.csv   (or --episodes-csv)
Writes: results/v14_shielded_qoe/paired_stats_v14.csv

Usage:
  cd new
  python src/evaluation/analyze_v14_shielded_qoe.py \
      --episodes-csv results/v14_shielded_qoe/online_episodes.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
except Exception:
    sp_stats = None

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.paths import get_paths

PATHS = get_paths()
METRICS = ["QoE", "Rebuffer", "VMAF"]

# (target, baseline, description). Baselines must exist in the sweep grid.
DEFAULT_PAIRS = [
    ("vmaf_aware_tol0.8_bud08", "shield_legacy", "VMAF-aware(0.8) vs legacy"),
    ("vmaf_aware_tol1.0_bud08", "shield_legacy", "VMAF-aware(1.0) vs legacy"),
    ("vmaf_aware_tol0.8_bud08", "highest_feasible_tol0.8", "VMAF-aware vs highest-index (0.8) [isolation]"),
    ("vmaf_aware_tol1.0_bud08", "highest_feasible_tol1.0", "VMAF-aware vs highest-index (1.0) [isolation]"),
    ("shield_off", "shield_legacy", "shield-off vs legacy [co-design]"),
]


def _paired(df, target, baseline, metric):
    a = df[df.Method == target].sort_values(["Video", "Episode"]).reset_index(drop=True)
    b = df[df.Method == baseline].sort_values(["Video", "Episode"]).reset_index(drop=True)
    if len(a) == 0 or len(a) != len(b):
        return None
    return a[metric].to_numpy(float) - b[metric].to_numpy(float)


def _wilcoxon(diff):
    n = len(diff)
    n_nonzero = int(np.count_nonzero(diff))
    mean = float(np.mean(diff))
    median = float(np.median(diff))
    if sp_stats is None or n_nonzero == 0:
        # All-zero differences: identical behaviour, p is undefined/1.0.
        return {"n": n, "n_nonzero": n_nonzero, "mean": mean, "median": median,
                "stat": float("nan"), "p": 1.0 if n_nonzero == 0 else float("nan")}
    stat, p = sp_stats.wilcoxon(diff, alternative="two-sided", zero_method="zsplit")
    return {"n": n, "n_nonzero": n_nonzero, "mean": mean, "median": median,
            "stat": float(stat), "p": float(p)}


def _holm(pvals):
    """Holm-Bonferroni adjusted p-values (preserving input order)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-csv", type=str,
                        default=str(PATHS["results"] / "v14_shielded_qoe" / "online_episodes.csv"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.episodes_csv)
    if not csv_path.exists():
        print(f"[ERROR] missing {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    available = set(df["Method"].unique())

    rows = []
    for metric in METRICS:
        family = []
        for target, baseline, desc in DEFAULT_PAIRS:
            if target not in available or baseline not in available:
                continue
            diff = _paired(df, target, baseline, metric)
            if diff is None:
                continue
            r = _wilcoxon(diff)
            r.update({"metric": metric, "target": target, "baseline": baseline, "desc": desc})
            family.append(r)
        # Holm within (metric) family across the comparisons present.
        if family:
            adj = _holm([r["p"] for r in family])
            for r, pa in zip(family, adj):
                r["p_holm"] = float(pa)
            rows.extend(family)

    out_df = pd.DataFrame(rows, columns=[
        "metric", "desc", "target", "baseline", "n", "n_nonzero",
        "mean", "median", "stat", "p", "p_holm",
    ])
    out_path = Path(args.out) if args.out else csv_path.with_name("paired_stats_v14.csv")
    out_df.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(out_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")

    iso = out_df[out_df["desc"].str.contains("isolation")]
    if not iso.empty:
        max_nonzero = int(iso["n_nonzero"].max())
        print("\n[Isolation check] max non-zero paired differences between "
              f"VMAF-aware and highest-feasible-index arms: {max_nonzero}")
        if max_nonzero == 0:
            print("  => Perceptual ranking is INERT on this (monotone) ladder, "
                  "as predicted analytically. The measured shield gain is due to "
                  "the feasibility margin, not VMAF-awareness.")


if __name__ == "__main__":
    main()
