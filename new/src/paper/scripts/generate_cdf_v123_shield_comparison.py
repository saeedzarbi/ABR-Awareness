#!/usr/bin/env python3
"""
Empirical CDFs for paired shield comparison (v123 online_episodes.csv).

Usage:
  python new/src/paper/scripts/generate_cdf_v123_shield_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {
    "shield_legacy": "Legacy projection",
    "vmaf_aware_tol1.0_bud08": "VMAF-aware ($\\tau{=}1.0$)",
    "shield_off": "No shield (same policy)",
}

COLORS = {
    "shield_legacy": "#E69F00",
    "vmaf_aware_tol1.0_bud08": "#0072B2",
    "shield_off": "#CC79A7",
}

LINESTYLES = {
    "shield_legacy": "--",
    "vmaf_aware_tol1.0_bud08": "-",
    "shield_off": "-.",
}


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    csv_path = root / "results" / "v123_shielded_qoe" / "online_episodes.csv"
    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.8,
        }
    )

    stack_figsize = (6.8, 3.9)
    stack_label = 13
    stack_tick = 12
    stack_legend = 11

    for col, xlabel, stem in [
        ("QoE", "Session QoE (sum surrogate)", "fig_cdf_qoe_v123_paired"),
        ("Rebuffer", "Rebuffer ratio (% of session)", "fig_cdf_rebuffer_v123_paired"),
    ]:
        fig, ax = plt.subplots(figsize=stack_figsize, constrained_layout=False)
        for key, label in METHODS.items():
            sub = df[df["Method"] == key][col].astype(float).values
            if len(sub) == 0:
                continue
            xs, ys = _ecdf(sub)
            highlight = key.startswith("vmaf_aware")
            ax.step(
                xs,
                ys,
                where="post",
                label=label,
                color=COLORS.get(key, None),
                linewidth=2.4 if highlight else 1.6,
                alpha=1.0 if highlight else 0.85,
                linestyle=LINESTYLES.get(key, "-"),
                zorder=4 if highlight else 2,
            )
        ax.set_xlabel(xlabel, fontsize=stack_label)
        ax.set_ylabel("Empirical CDF", fontsize=stack_label)
        ax.tick_params(axis="both", labelsize=stack_tick)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.35, linestyle=":")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            fontsize=stack_legend,
            ncol=1,
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.24, left=0.10, right=0.98, top=0.97)
        for ext in ("pdf", "png"):
            p = out_dir / f"{stem}.{ext}"
            fig.savefig(p, bbox_inches="tight", facecolor="white", edgecolor="none")
            print("Wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
