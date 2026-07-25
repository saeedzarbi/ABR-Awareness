#!/usr/bin/env python3
"""
Publication figures from v123 online shield evaluation (VMAF-aware sweep).

Reads:
  new/results/<run>/online_summary.csv  (and optionally online_episodes.csv)

Writes (default out dir: new/src/paper/figures/):
  fig_v123_pareto_qoe_rebuffer.pdf|.png
  fig_v123_empirical_pareto_front.pdf|.png  (non-dominated points + envelope)
  summary_v123_unique_behaviors.csv         (deduped behavioral classes)

If --episodes is passed:
  fig_v123_paired_dQoE_vs_legacy.pdf|.png   (paired delta QoE vs shield_legacy)

This script deliberately does NOT touch macros_v12.tex or any v12 table fragments
(so running it after a v12 regenerate is safe for shield-only iteration).

Usage:
  cd new
  python src/paper/scripts/generate_figures_v123.py
  python src/paper/scripts/generate_figures_v123.py \\
      --summary results/v123_shielded_qoe/online_summary.csv \\
      --episodes results/v123_shielded_qoe/online_episodes.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style aligned with generate_figures_v12.py
# ---------------------------------------------------------------------------
COLORS = {
    "legacy": "#666666",
    "off": "#CC3311",
    "vmaf": "#009E73",
    "thresh": "#0072B2",
    "lookahead": "#E69F00",
    "other": "#999999",
}


def _setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            # STIX matches the Times/serif body font; without this, any mathtext
            # ($...$, e.g. "$\Delta$QoE") silently falls back to DejaVu Sans and
            # mixes a sans-serif face into an otherwise all-serif figure.
            "mathtext.fontset": "stix",
            # Larger fonts for better readability (matching Fig 9/10 style).
            "font.size": 22,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "axes.titleweight": "600",
            "axes.linewidth": 1.3,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "grid.color": "#E0E0E0",
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.3,
            "lines.markersize": 8.5,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "#CCCCCC",
            "legend.framealpha": 0.95,
            "legend.fontsize": 18,
            "figure.constrained_layout.use": True,
        }
    )


STACK_PANEL_FIGSIZE = (7.2, 4.7)
STACK_PANEL_LABELSIZE = 24
STACK_PANEL_TICKSIZE = 22
STACK_PANEL_LEGENDSIZE = 20
def _repo_new() -> Path:
    return Path(__file__).resolve().parents[3]


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Wrote {p}")


def _family(method: str) -> str:
    if method == "shield_legacy":
        return "legacy"
    if method == "shield_off":
        return "off"
    if method.startswith("vmaf_aware") or method.startswith("vmaf_riskgate"):
        return "vmaf"
    if method.startswith("thresh_"):
        return "thresh"
    if method.startswith("lookahead_"):
        return "lookahead"
    if method.startswith("shield_"):
        return "other"
    return "other"


def _dedupe_behaviors(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (QoE, Rebuf, VMAF) behavioral class (same rounding as paper text)."""
    t = df.copy()
    t["_qk"] = np.round(t["QoE_mean"].astype(float), 1)
    t["_rk"] = np.round(t["Rebuf_pct"].astype(float), 2)
    t["_vk"] = np.round(t["VMAF_mean"].astype(float), 2)
    t["_fam"] = t["Method"].map(_family)
    # first row per class; stable sort by Method for reproducibility
    t = t.sort_values("Method")
    u = t.groupby(["_qk", "_rk", "_vk"], as_index=False).first()
    return u.drop(columns=["_qk", "_rk", "_vk"], errors="ignore")


def _non_dominated_mask(rb: np.ndarray, qoe: np.ndarray) -> np.ndarray:
    """Non-dominated under: lower Rebuf better, higher QoE better (strict dominance)."""
    n = len(rb)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rb[j] <= rb[i] and qoe[j] >= qoe[i] and (rb[j] < rb[i] or qoe[j] > qoe[i]):
                keep[i] = False
                break
    return keep


def plot_pareto_scatter(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=STACK_PANEL_FIGSIZE, layout="constrained")
    for _, r in summary.iterrows():
        m = r["Method"]
        fam = _family(m)
        c = COLORS[fam]
        x = float(r["Rebuf_pct"])
        y = float(r["QoE_mean"])
        xlo, xhi = float(r["Rebuf_ci_lo"]), float(r["Rebuf_ci_hi"])
        ylo, yhi = float(r["QoE_ci_lo"]), float(r["QoE_ci_hi"])
        ax.errorbar(
            x,
            y,
            xerr=[[x - xlo], [xhi - x]],
            yerr=[[y - ylo], [yhi - y]],
            fmt="o",
            color=c,
            ecolor=c,
            alpha=0.85,
            markersize=7 if m in ("shield_legacy", "shield_off") else 5,
            capsize=2,
            elinewidth=0.8,
            zorder=3 if fam in ("legacy", "off") else 2,
        )

    # Legend by family (proxy artists)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["legacy"], label="legacy shield", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["off"], label="shield off", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["vmaf"], label="VMAF-aware / risk-gate", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["thresh"], label="thresh sweep", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["lookahead"], label="lookahead", markersize=8),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        fontsize=STACK_PANEL_LEGENDSIZE,
        ncol=3,
        frameon=True,
    )
    ax.set_xlabel("Mean rebuffer ratio (%)", fontsize=STACK_PANEL_LABELSIZE)
    ax.set_ylabel("Mean session QoE", fontsize=STACK_PANEL_LABELSIZE)
    ax.tick_params(axis="both", labelsize=STACK_PANEL_TICKSIZE)
    ax.grid(True, linestyle="--", alpha=0.7)
    _save(fig, out_dir, "fig_v123_pareto_qoe_rebuffer")
    plt.close(fig)


def plot_pareto_front_line(unique: pd.DataFrame, out_dir: Path) -> None:
    rb = unique["Rebuf_pct"].to_numpy(dtype=float)
    qoe = unique["QoE_mean"].to_numpy(dtype=float)
    mask = _non_dominated_mask(rb, qoe)
    front = unique.loc[mask].sort_values("Rebuf_pct")
    fig, ax = plt.subplots(figsize=STACK_PANEL_FIGSIZE, layout="constrained")
    ax.scatter(rb, qoe, c="#BBBBBB", s=40, zorder=1, label="dominated points")
    ax.plot(
        front["Rebuf_pct"],
        front["QoE_mean"],
        color="#000000",
        linewidth=1.2,
        drawstyle="steps-post",
        zorder=2,
        label="empirical envelope",
    )
    ax.scatter(
        front["Rebuf_pct"],
        front["QoE_mean"],
        c="#0072B2",
        s=55,
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
        label="nondominated",
    )
    for idx, (_, r) in enumerate(front.iterrows()):
        offset_y = 6 if idx % 2 == 0 else -10
        ax.annotate(
            str(r["Method"])[:22],
            (float(r["Rebuf_pct"]), float(r["QoE_mean"])),
            textcoords="offset points",
            xytext=(5, offset_y),
            fontsize=16,
            alpha=0.85,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4),
        )
    ax.set_xlabel("Mean rebuffer ratio (%)", fontsize=STACK_PANEL_LABELSIZE)
    ax.set_ylabel("Mean session QoE", fontsize=STACK_PANEL_LABELSIZE)
    ax.tick_params(axis="both", labelsize=STACK_PANEL_TICKSIZE)
    ax.grid(True, linestyle="--", alpha=0.7)
    # Figure-level legend below the axes — an in-axes legend previously sat
    # on top of the "dominated points" cluster in the lower-left corner.
    fig.legend(loc="outside lower center", fontsize=STACK_PANEL_LEGENDSIZE, ncol=3)
    _save(fig, out_dir, "fig_v123_empirical_pareto_front")
    plt.close(fig)


def plot_paired_delta_qoe(episodes_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(episodes_csv)
    if "shield_legacy" not in set(df["Method"].unique()):
        print("[WARN] shield_legacy missing in episodes; skip paired delta figure")
        return
    targets = [
        "vmaf_aware_tol1.0_bud08",
        "vmaf_aware_tol0.8_bud08",
        "vmaf_aware_tol1.2_bud08",
        "lookahead_h3_mb1.0_vmafFB",
        "thresh_cat3.0_vmafFB",
        "thresh_cat4.0_vmafFB",
        "shield_off",
    ]
    legacy = df[df.Method == "shield_legacy"].sort_values(["Video", "Episode"])
    labels = []
    deltas = []
    for m in targets:
        if m not in set(df["Method"].unique()):
            continue
        sub = df[df.Method == m].sort_values(["Video", "Episode"])
        if len(sub) != len(legacy):
            print(f"[WARN] length mismatch for {m}; skip")
            continue
        d = (sub["QoE"].to_numpy(dtype=float) - legacy["QoE"].to_numpy(dtype=float)).astype(float)
        labels.append(m)
        deltas.append(d)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    pos = np.arange(len(labels))
    med = [float(np.median(d)) for d in deltas]
    lo = [float(np.percentile(d, 25)) for d in deltas]
    hi = [float(np.percentile(d, 75)) for d in deltas]
    colors = [COLORS.get(_family(l), "#333333") for l in labels]
    ax.barh(pos, med, color=colors, alpha=0.85, height=0.65)
    ax.errorbar(med, pos, xerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], fmt="none", c="#333333", capsize=2, linewidth=0.8)
    ax.axvline(0.0, color="#111111", linewidth=1.7, linestyle="--", zorder=2)
    ax.set_yticks(pos)
    ax.set_yticklabels([l[:32] for l in labels], fontsize=20)
    ax.set_xlabel("Paired ΔQoE vs. shield_legacy (same Video×Episode)", fontsize=22)
    ax.set_title("Headline shield variants", fontsize=22)
    ax.tick_params(axis="x", labelsize=20)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    _save(fig, out_dir, "fig_v123_paired_dQoE_vs_legacy")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v123 shield figures (does not touch macros).")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to online_summary.csv",
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=None,
        help="Optional path to online_episodes.csv (paired delta figure)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for PDF/PNG/CSV",
    )
    args = parser.parse_args()
    new_root = _repo_new()
    summary_path = args.summary or (new_root / "results" / "v123_shielded_qoe" / "online_summary.csv")
    out_dir = args.out or (new_root / "src" / "paper" / "figures")

    if not summary_path.is_file():
        raise SystemExit(f"Missing summary CSV: {summary_path}")

    _setup_matplotlib()
    df = pd.read_csv(summary_path)
    required = {"Method", "QoE_mean", "Rebuf_pct", "VMAF_mean", "QoE_ci_lo", "QoE_ci_hi", "Rebuf_ci_lo", "Rebuf_ci_hi"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Summary CSV missing columns: {missing}")

    unique = _dedupe_behaviors(df)
    csv_out = out_dir / "summary_v123_unique_behaviors.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    unique.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out} ({len(unique)} unique classes from {len(df)} rows)")

    plot_pareto_scatter(unique, out_dir)
    plot_pareto_front_line(unique, out_dir)

    episodes_path = args.episodes or (summary_path.parent / "online_episodes.csv")
    if episodes_path.is_file():
        plot_paired_delta_qoe(episodes_path, out_dir)
    else:
        print(f"[INFO] No episodes file at {episodes_path}; skip paired delta figure")


if __name__ == "__main__":
    main()
