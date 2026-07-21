#!/usr/bin/env python3
"""
Empirical CDFs for paired shield comparison (v123 online_episodes.csv).

Plots QoE and rebuffer ratio for identical Video×Episode seeds across methods,
highlighting that VMAF-aware projection shifts mass right (QoE) and toward
near-zero rebuffer versus legacy projection.

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
    "vmaf_aware_tol1.0_bud08": "VMAF-aware (v123, $\\tau{=}1.0$)",
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
            # Sized for legibility after scaling in the manuscript.
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "figure.figsize": (5.0, 3.2),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.8,
            "figure.constrained_layout.use": True,
        }
    )

    for col, xlabel, stem in [
        ("QoE", "Session QoE (sum surrogate)", "fig_cdf_qoe_v123_paired"),
        ("Rebuffer", "Rebuffer ratio (% of session)", "fig_cdf_rebuffer_v123_paired"),
    ]:
        fig, ax = plt.subplots()
        for key, label in METHODS.items():
            sub = df[df["Method"] == key][col].astype(float).values
            if len(sub) == 0:
                continue
            xs, ys = _ecdf(sub)
            ax.step(
                xs,
                ys,
                where="post",
                label=label,
                color=COLORS.get(key, None),
                linewidth=1.9,
                linestyle=LINESTYLES.get(key, "-"),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", framealpha=0.92)
        ax.grid(True, alpha=0.35, linestyle=":")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out_dir / f"{stem}.{ext}"
            fig.savefig(p, bbox_inches="tight")
            print("Wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
