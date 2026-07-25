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
            # STIX matches the Times/serif body font; without this, the mathtext
            # legend label "VMAF-aware ($\tau{=}1.0$)" falls back to DejaVu Sans.
            "mathtext.fontset": "stix",
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "legend.fontsize": 18,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.3,
            "lines.linewidth": 2.3,
        }
    )

    stack_figsize = (7.2, 4.7)
    stack_label = 24
    stack_tick = 22
    stack_legend = 20
    for col, xlabel, stem in [
        ("QoE", "Session QoE (sum surrogate)", "fig_cdf_qoe_v123_paired"),
        ("Rebuffer", "Rebuffer ratio (% of session)", "fig_cdf_rebuffer_v123_paired"),
    ]:
        fig, ax = plt.subplots(figsize=stack_figsize, layout="constrained")
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
        # Figure-level legend outside the axes so the constrained layout
        # engine can reserve exact space for it (no manual margin tuning).
        fig.legend(
            loc="outside lower center",
            fontsize=stack_legend,
            ncol=3,
            frameon=True,
        )
        for ext in ("pdf", "png"):
            p = out_dir / f"{stem}.{ext}"
            fig.savefig(p, bbox_inches="tight", facecolor="white", edgecolor="none")
            print("Wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
