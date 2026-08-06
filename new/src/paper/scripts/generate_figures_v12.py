#!/usr/bin/env python3
"""
Generate publication-quality figures from v12 evaluation CSVs.

Inputs (default paths relative to repo new/):
  results/detailed_stats_master_v12_v12_policy.csv
  results/decision_log_v12_v12_policy.csv

Outputs:
  new/src/paper/figures/*.pdf and *.png
  new/src/paper/figures/summary_bootstrap_v12.csv
  new/src/paper/tables/table_main_results_ci.tex
  new/src/paper/tables/table_paired_wilcoxon_qoe_headline.tex
  new/src/paper/tables/table_paired_wilcoxon_rebuffer_headline.tex
  new/src/paper/tables/macros_v12.tex (abstract-ready scalars)
  new/src/paper/tables/paired_wilcoxon_v12.csv

Usage:
  python new/src/paper/scripts/generate_figures_v12.py
  python new/src/paper/scripts/generate_figures_v12.py --stats path/to/detailed_stats.csv --decisions path/to/decision_log.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
# Style: modern, colorblind-friendly (Okabe–Ito inspired + neutrals)
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "Genie": "#000000",
    "RobustMPC": "#E69F00",
    "Pensieve": "#56B4E9",
    "BBA": "#CC79A7",
    "Fugu": "#999999",
    "Proposed": "#0072B2",
    "Proposed_Shielded": "#009E73",
    "Proposed_ShieldedRiskGate": "#D55E00",
    "Proposed_ShieldedQoE": "#6A3D9A",
    "Ablation_Base": "#B15928",
    "Ablation_Future": "#FDB462",
    "Ablation_Lyap": "#B2DF8A",
}

DISPLAY_NAME: dict[str, str] = {
    "Genie": "Genie (oracle)",
    "RobustMPC": "RobustMPC",
    "Pensieve": "Pensieve",
    "BBA": "BBA",
    "Fugu": "Fugu",
    "Proposed": "Ours: CMDP (no shield)",
    "Proposed_Shielded": "Ours: + shield",
    "Proposed_ShieldedRiskGate": "Ours: + risk-gate",
    "Proposed_ShieldedQoE": "Ours: + shield-QoE",
    "Ablation_Base": "Abl. base",
    "Ablation_Future": "Abl. + future",
    "Ablation_Lyap": "Abl. + Lyapunov",
}

ORDER_MAIN = [
    "Genie",
    "RobustMPC",
    "Pensieve",
    "Fugu",
    "BBA",
    "Proposed",
    "Proposed_Shielded",
    "Proposed_ShieldedRiskGate",
    "Proposed_ShieldedQoE",
]

ORDER_ABLATION = ["Proposed", "Ablation_Base", "Ablation_Future", "Ablation_Lyap"]

# Ablation bars use the subtractive wording of the ablation table and the running
# text ("- Lagrangian constraints" etc.) so figure and table can be read together.
ABLATION_LABEL: dict[str, str] = {
    "Proposed": "Full CMDP",
    "Ablation_Base": r"$-$ Lagrangian",
    "Ablation_Future": r"$-$ Lookahead",
    "Ablation_Lyap": r"$-$ Lyapunov",
}

# Distinct linestyles for B&W / colorblind-safe CDF overlays (keyed by method).
LINESTYLES_BY_METHOD: dict[str, str | tuple] = {
    "Genie": "-",
    "RobustMPC": (0, (5, 1)),
    "Pensieve": "--",
    "BBA": ":",
    "Fugu": "-.",
    "Proposed": (0, (3, 1, 1, 1)),
    "Proposed_Shielded": "-",  # highlight: solid + thicker
    "Proposed_ShieldedRiskGate": (0, (5, 2, 1, 2)),
    "Proposed_ShieldedQoE": (0, (3, 1, 1, 1, 1, 1)),
}
# Fallback cycle if an unexpected method appears.
LINESTYLES_MAIN: list[str | tuple] = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
]
# Proposed / VMAF-aware shield variants are drawn one step thicker.
HIGHLIGHT_METHODS = {
    "Proposed_Shielded",
    "Proposed_ShieldedQoE",
    "Proposed_ShieldedRiskGate",
}

# LaTeX first-column entries (with citations where useful)
LATEX_METHOD_COL: dict[str, str] = {
    "Genie": r"Genie (oracle)~\cite{mao2017neural}",
    "RobustMPC": r"RobustMPC~\cite{yin2015control}",
    "Pensieve": r"Pensieve~\cite{mao2017neural}",
    "BBA": r"BBA~\cite{huang2014buffer}",
    "Fugu": r"Fugu~\cite{yan2019learning}",
    "Proposed": r"Ours: CMDP (no shield)",
    "Proposed_Shielded": r"Ours: + shield",
    "Proposed_ShieldedRiskGate": r"Ours: + risk-gate",
    "Proposed_ShieldedQoE": r"Ours: + shield-QoE$^{\dagger}$",
    "Ablation_Base": r"Abl.\ base",
    "Ablation_Future": r"Abl.\ + future",
    "Ablation_Lyap": r"Abl.\ + Lyapunov",
}

METHOD_CLASS: dict[str, str] = {
    "Genie": "oracle",
    "RobustMPC": "MPC",
    "Pensieve": "DRL",
    "BBA": "heur.",
    "Fugu": "IL",
    "Proposed": "RL+CMDP",
    "Proposed_Shielded": "RL+CMDP+S",
    "Proposed_ShieldedRiskGate": "RL+CMDP+RG",
    "Proposed_ShieldedQoE": "RL+CMDP+Q",
    "Ablation_Base": "abl.",
    "Ablation_Future": "abl.",
    "Ablation_Lyap": "abl.",
}

# Short axis labels for dense plots (heatmap rows, forest y-axis).
try:
    import sys as _sys
    _NEW = Path(__file__).resolve().parents[3]
    _sys.path.insert(0, str(_NEW))
    from configs.videos import ALL_VIDEOS, DISPLAY_NAMES, LEGACY_ALIASES

    VIDEO_SHORT_LABEL: dict[str, str] = {
        slug: DISPLAY_NAMES.get(slug, slug).replace(" ", "").replace("&", "")
        for slug in ALL_VIDEOS
    }
    for old, new in LEGACY_ALIASES.items():
        VIDEO_SHORT_LABEL.setdefault(old, VIDEO_SHORT_LABEL.get(new, old))
except Exception:
    VIDEO_SHORT_LABEL = {
        "bigbuckbunny": "BigBuckBunny",
        "crowd_run": "CrowdRun",
        "sintel": "Sintel",
        "tearsofsteel_short": "TearsOfSteel",
        "tearsofsteel": "TearsOfSteel",
    }

HEATMAP_ROW_LABEL: dict[str, str] = {
    "Genie": "Genie",
    "RobustMPC": "RobustMPC",
    "Pensieve": "Pensieve",
    "Fugu": "Fugu",
    "BBA": "BBA",
    "Proposed": "Ours: CMDP",
    "Proposed_Shielded": "Ours: + shield",
    "Proposed_ShieldedRiskGate": "Ours: + risk-gate",
    "Proposed_ShieldedQoE": "Ours: + shield-QoE",
}

FOREST_LABEL: dict[str, str] = {
    "Genie": "Genie (oracle)",
    "Proposed": "Ours: CMDP",
    "Proposed_Shielded": "Ours: + shield",
    "Proposed_ShieldedRiskGate": "Ours: + risk-gate",
    "Proposed_ShieldedQoE": "Ours: + shield-QoE",
}

# Compact legend entries for crowded CDF panels.
CDF_LABEL: dict[str, str] = {
    "Genie": "Genie",
    "RobustMPC": "RobustMPC",
    "Pensieve": "Pensieve",
    "BBA": "BBA",
    "Fugu": "Fugu",
    "Proposed": "Ours (no shield)",
    "Proposed_Shielded": "Ours + shield",
    "Proposed_ShieldedRiskGate": "Ours + risk-gate",
    "Proposed_ShieldedQoE": "Ours + shield-QoE",
}


def _setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            # TrueType embedding (42) — Type 3 bitmap fonts look pixelated in PDF viewers.
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
            "axes.labelweight": "500",
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
            "legend.borderaxespad": 0.5,
            "legend.fontsize": 18,
            "figure.constrained_layout.use": True,
        }
    )


# Full-width stacked (a)/(b) panels: one panel per \linewidth row in LaTeX.
STACK_PANEL_FIGSIZE = (7.2, 4.7)
STACK_PANEL_LABELSIZE = 24
STACK_PANEL_TICKSIZE = 22
STACK_PANEL_LEGENDSIZE = 20
FOREST_FIGSIZE_W = 10.5
# Heatmap: smaller type than scatter/CDF panels so axis labels do not dominate the grid.
HEATMAP_TICKSIZE = 13
HEATMAP_AXISLABELSIZE = 14
HEATMAP_CBARSIZE = 13
# Compact bar charts (shield intervention, ablation panels).
COMPACT_TICKSIZE = 13
COMPACT_LABELSIZE = 14
COMPACT_TITLESIZE = 14
COMPACT_BARLABELSIZE = 11


def _repo_new() -> Path:
    return Path(__file__).resolve().parents[3]


def _bootstrap_ci(
    x: np.ndarray, n_boot: int = 4000, ci: float = 95.0, rng: np.random.Generator | None = None
) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.mean(x)), float(lo), float(hi)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Wrote {p}")


def fig_tradeoff_with_ci(df: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in ORDER_MAIN if m in set(df["Method"].unique())]
    rows = []
    rng = np.random.default_rng(0)
    for m in methods:
        sub = df[df["Method"] == m]
        qm, qlo, qhi = _bootstrap_ci(sub["QoE"].values, rng=rng)
        rm, rlo, rhi = _bootstrap_ci(sub["Rebuffer"].values, rng=rng)
        rows.append(
            {
                "Method": m,
                "QoE_m": qm,
                "QoE_lo": qlo,
                "QoE_hi": qhi,
                "Rb_m": rm,
                "Rb_lo": rlo,
                "Rb_hi": rhi,
            }
        )
    summ = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=False)
    for _, r in summ.iterrows():
        m = r["Method"]
        c = COLORS.get(m, "#444444")
        ax.errorbar(
            r["Rb_m"],
            r["QoE_m"],
            xerr=[[r["Rb_m"] - r["Rb_lo"]], [r["Rb_hi"] - r["Rb_m"]]],
            yerr=[[r["QoE_m"] - r["QoE_lo"]], [r["QoE_hi"] - r["QoE_m"]]],
            fmt="o",
            color=c,
            ecolor=c,
            elinewidth=1.3,
            capsize=3.5,
            markersize=9 if m != "Genie" else 10,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
            label=DISPLAY_NAME.get(m, m),
        )

    ax.set_xlabel("Mean rebuffer ratio (% of session)", fontsize=22)
    ax.set_ylabel("Session QoE (sum)", fontsize=22)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=18,
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.0,
        frameon=True,
    )
    fig.subplots_adjust(bottom=0.22, left=0.10, right=0.98, top=0.97)
    _save(fig, out_dir, "fig_tradeoff_qoe_rebuffer")
    plt.close(fig)


def fig_ecdf(df: pd.DataFrame, out_dir: Path, column: str, xlabel: str, stem: str) -> None:
    methods = [m for m in ORDER_MAIN if m in set(df["Method"].unique())]
    fig, ax = plt.subplots(figsize=STACK_PANEL_FIGSIZE, layout="constrained")
    for i, m in enumerate(methods):
        sub = np.sort(df[df["Method"] == m][column].values.astype(float))
        y = np.arange(1, len(sub) + 1) / len(sub)
        ls = LINESTYLES_BY_METHOD.get(m, LINESTYLES_MAIN[i % len(LINESTYLES_MAIN)])
        highlight = m in HIGHLIGHT_METHODS
        lw = 2.4 if highlight else 1.35
        alpha = 1.0 if highlight else 0.82
        ax.step(
            sub,
            y,
            where="post",
            color=COLORS.get(m, "#444444"),
            lw=lw,
            alpha=alpha,
            linestyle=ls,
            label=CDF_LABEL.get(m, DISPLAY_NAME.get(m, m)),
            zorder=4 if highlight else 2,
        )
    ax.set_xlabel(xlabel, fontsize=STACK_PANEL_LABELSIZE)
    ax.set_ylabel("Empirical CDF", fontsize=STACK_PANEL_LABELSIZE)
    ax.tick_params(axis="both", labelsize=STACK_PANEL_TICKSIZE)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.35, linestyle="--")
    # Figure-level legend placed *outside* the axes; the constrained layout
    # engine reserves exactly the space it needs, so it never collides with
    # the x-axis label regardless of font size or entry count.
    fig.legend(
        loc="outside lower center",
        fontsize=STACK_PANEL_LEGENDSIZE,
        ncol=5,
        handletextpad=0.35,
        columnspacing=0.9,
        frameon=True,
    )
    _save(fig, out_dir, stem)
    plt.close(fig)


def fig_shield_intervention(decisions: pd.DataFrame, out_dir: Path) -> None:
    shield_methods = [m for m in decisions["Method"].unique() if "Shield" in m or m == "Proposed"]
    if not shield_methods:
        return
    rows = []
    for m in sorted(shield_methods):
        sub = decisions[decisions["Method"] == m]
        if len(sub) == 0:
            continue
        rate = float(sub["Shield_Intervened"].mean()) if "Shield_Intervened" in sub.columns else 0.0
        rows.append((DISPLAY_NAME.get(m, m), m, rate * 100.0))
    if not rows:
        return
    rows.sort(key=lambda t: t[2])
    labels = [
        HEATMAP_ROW_LABEL.get(r[1], DISPLAY_NAME.get(r[1], r[0])) for r in rows
    ]
    vals = [r[2] for r in rows]
    colors = [COLORS.get(r[1], "#888888") for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=False)
    ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.6, height=0.62)
    ax.set_xlabel("Intervention rate (% of chunk steps)", fontsize=COMPACT_LABELSIZE)
    ax.tick_params(axis="y", labelsize=COMPACT_TICKSIZE)
    ax.tick_params(axis="x", labelsize=COMPACT_TICKSIZE, length=3, pad=2)
    ax.grid(True, axis="x", which="major", alpha=0.55, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, "fig_shield_intervention_rate")
    plt.close(fig)


def fig_ablation_bars(df: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in ORDER_ABLATION if m in set(df["Method"].unique())]
    if len(methods) < 2:
        return
    agg = df[df["Method"].isin(methods)].groupby("Method", as_index=False)[["QoE", "Rebuffer"]].mean()
    agg["Label"] = agg["Method"].map(lambda m: ABLATION_LABEL.get(m, DISPLAY_NAME.get(m, m)))
    order = [ABLATION_LABEL.get(m, DISPLAY_NAME[m]) for m in methods if m in set(agg["Method"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=False)
    for ax, col, title in zip(
        axes,
        ["QoE", "Rebuffer"],
        ["Mean session QoE", "Mean rebuffer (%)"],
    ):
        sub = agg.set_index("Method").loc[[m for m in methods if m in agg["Method"].values]].reset_index()
        sub["Label"] = sub["Method"].map(lambda m: ABLATION_LABEL.get(m, DISPLAY_NAME.get(m, m)))
        sub = sub.set_index("Label").loc[[l for l in order if l in sub["Label"].values]].reset_index()
        cols = [COLORS.get(m, "#888888") for m in sub["Method"]]
        bars = ax.bar(sub["Label"], sub[col], color=cols, edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontsize=COMPACT_TITLESIZE)
        ax.tick_params(axis="both", labelsize=COMPACT_TICKSIZE, length=3, pad=2)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.35, linestyle="--")
        # Bar charts must start at zero to avoid exaggerating small gaps.
        ax.set_ylim(bottom=0)
        # With a zero baseline the QoE bars differ by about 1%, so print the
        # values; the axis stays honest and the numbers stay readable.
        fmt = "{:.0f}" if col == "QoE" else "{:.1f}"
        ax.bar_label(
            bars,
            labels=[fmt.format(v) for v in sub[col]],
            fontsize=COMPACT_BARLABELSIZE,
            padding=2,
        )
        ax.set_ylim(top=float(sub[col].max()) * 1.16)
    fig.tight_layout()
    _save(fig, out_dir, "fig_ablation_qoe_rebuffer")
    plt.close(fig)


def fig_per_video_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in ORDER_MAIN if m in set(df["Method"].unique())]
    pv = (
        df[df["Method"].isin(methods)]
        .groupby(["Video", "Method"], as_index=False)["QoE"]
        .mean()
        .pivot(index="Video", columns="Method", values="QoE")
    )
    pv = pv[[c for c in ORDER_MAIN if c in pv.columns]]
    video_labels = [
        VIDEO_SHORT_LABEL.get(str(v), str(v).replace("_", " ")) for v in pv.index
    ]
    # Transpose: methods = rows (horizontal labels), videos = columns.
    pv_t = pv.T

    n_methods = len(pv_t.index)
    n_videos = len(pv_t.columns)
    if n_methods == 0 or n_videos == 0:
        return
    fig_h = max(4.4, 0.40 * n_methods + 1.5)
    fig_w = max(6.8, 1.05 * n_videos + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
    im = ax.imshow(
        pv_t.values,
        aspect="auto",
        cmap="cividis",
        vmin=float(pv.values.min()),
        vmax=float(pv.values.max()),
    )
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(
        [HEATMAP_ROW_LABEL.get(c, DISPLAY_NAME.get(c, c)) for c in pv_t.index],
        fontsize=HEATMAP_TICKSIZE,
    )
    ax.set_xticks(range(n_videos))
    ax.set_xticklabels(
        video_labels,
        rotation=25,
        ha="right",
        fontsize=HEATMAP_TICKSIZE,
    )
    ax.set_xlabel("Video sequence", fontsize=HEATMAP_AXISLABELSIZE)
    ax.set_ylabel("Method", fontsize=HEATMAP_AXISLABELSIZE)
    ax.tick_params(axis="both", labelsize=HEATMAP_TICKSIZE, length=3, pad=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Mean session QoE", fontsize=HEATMAP_CBARSIZE)
    cbar.ax.tick_params(labelsize=HEATMAP_TICKSIZE - 1, length=2, pad=1)
    fig.tight_layout()
    _save(fig, out_dir, "fig_heatmap_qoe_by_video")
    plt.close(fig)


def fig_timeseries_compare(
    decisions: pd.DataFrame,
    out_dir: Path,
    video: str = "sintel",
    episode: int = 0,
    methods: tuple[str, str] = ("Proposed", "Proposed_Shielded"),
) -> None:
    """Two-row panel: throughput, buffer, selected bitrate, per-chunk rebuffer."""
    # Style map: circle vs cross so overlapping traces remain separable.
    style_by_method = {
        methods[0]: {"ls": "-", "marker": "o", "markevery": 3},
        methods[1]: {"ls": "-.", "marker": "x", "markevery": 3},
    }
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 8.4), sharex=True, layout="constrained")
    for row, m in enumerate(methods):
        if m not in decisions["Method"].values:
            continue
        sub = decisions[
            (decisions["Method"] == m) & (decisions["Video"] == video) & (decisions["Episode"] == episode)
        ].sort_values("Chunk")
        if sub.empty:
            continue
        chunks = sub["Chunk"].values
        tp = sub["Throughput_kbps"].values / 1000.0
        buf = sub["Buffer_Before"].values
        br = sub["Bitrate_kbps"].values
        rb = sub["Rebuffer_s"].values
        color = COLORS.get(m, "#444444")
        st = style_by_method.get(m, {"ls": "-", "marker": "o", "markevery": 3})
        # Mark bitrate switch points so overlapping paths stay trackable.
        if len(br) > 1:
            switch_idx = np.flatnonzero(br[1:] != br[:-1]) + 1
        else:
            switch_idx = np.array([], dtype=int)

        # Throughput is environment-shared; plot once from the first available method.
        if row == 0:
            axes[0].plot(
                chunks,
                tp,
                linestyle="-",
                marker="o",
                markevery=st["markevery"],
                ms=3.2,
                lw=1.4,
                color="#333333",
                label="Throughput",
                alpha=0.7,
            )
        axes[1].step(
            chunks,
            buf,
            where="mid",
            color=color,
            lw=1.6,
            linestyle=st["ls"],
            label=DISPLAY_NAME.get(m, m),
            alpha=0.7,
        )
        axes[1].plot(
            chunks[:: st["markevery"]],
            buf[:: st["markevery"]],
            linestyle="None",
            marker=st["marker"],
            ms=4.0,
            color=color,
            alpha=0.85,
        )
        axes[2].step(
            chunks,
            br,
            where="mid",
            color=color,
            lw=1.6,
            linestyle=st["ls"],
            label=DISPLAY_NAME.get(m, m),
            alpha=0.7,
        )
        axes[2].plot(
            chunks[switch_idx] if len(switch_idx) else chunks[:: st["markevery"]],
            br[switch_idx] if len(switch_idx) else br[:: st["markevery"]],
            linestyle="None",
            marker=st["marker"],
            ms=5.0,
            color=color,
            alpha=0.9,
            zorder=5,
        )
        axes[3].bar(
            chunks + 0.18 * (row - 0.5),
            rb,
            width=0.35,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.35,
            hatch="" if row == 0 else "//",
            label=DISPLAY_NAME.get(m, m),
        )

    axes[0].set_ylabel("Throughput (Mb/s)", fontsize=20)
    axes[1].set_ylabel("Buffer (s)", fontsize=20)
    axes[2].set_ylabel("Bitrate (kbps)", fontsize=20)
    axes[3].set_ylabel("Rebuf. (s)", fontsize=20)
    axes[-1].set_xlabel("Chunk index", fontsize=20)
    axes[0].set_title(f"Representative trace ({video}, episode {episode})", fontsize=22)
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=20)
    axes[1].legend(loc="upper right", fontsize=18, ncol=1, handletextpad=0.3)
    _save(fig, out_dir, "fig_timeseries_proposed_vs_shielded")
    plt.close(fig)


def fig_forest_two_baselines(df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side paired mean ΔQoE + bootstrap CI vs Pensieve / RobustMPC."""
    baselines = [b for b in ("Pensieve", "RobustMPC") if b in set(df["Method"].unique())]
    targets = [
        "Proposed",
        "Proposed_Shielded",
        "Proposed_ShieldedRiskGate",
        "Proposed_ShieldedQoE",
        "Genie",
    ]
    targets = [m for m in targets if m in set(df["Method"].unique())]
    rng = np.random.default_rng(2)
    if not baselines or not targets:
        return

    n_rows = max(1, len([m for m in targets if m not in baselines]) - 0)
    fig_h = max(5.0, 0.55 * n_rows + 1.8)
    fig, axes = plt.subplots(
        1, len(baselines), figsize=(FOREST_FIGSIZE_W, fig_h), squeeze=False, layout="constrained"
    )
    for ax_col, baseline in enumerate(baselines):
        ax = axes[0, ax_col]
        base = df[df["Method"] == baseline][["Video", "Episode", "QoE"]].rename(columns={"QoE": "QoE_b"})
        ys, pos = [], 0
        for m in targets:
            if m == baseline:
                continue
            a = df[df["Method"] == m][["Video", "Episode", "QoE"]].rename(columns={"QoE": "QoE_m"})
            merged = a.merge(base, on=["Video", "Episode"])
            if len(merged) < 10:
                continue
            delta = (merged["QoE_m"] - merged["QoE_b"]).values.astype(float)
            mean_d = float(np.mean(delta))
            _, lo, hi = _bootstrap_ci(delta, rng=rng)
            c = COLORS.get(m, "#444444")
            ax.barh(
                pos,
                mean_d,
                xerr=[[mean_d - lo], [hi - mean_d]],
                height=0.58,
                color=c,
                ecolor=c,
                capsize=3.5,
                edgecolor="white",
                linewidth=0.35,
            )
            ys.append(FOREST_LABEL.get(m, DISPLAY_NAME.get(m, m)))
            pos += 1
        ax.axvline(0.0, color="#111111", lw=1.7, linestyle="--", zorder=2)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys, fontsize=STACK_PANEL_TICKSIZE)
        # Baseline name already appears in the title above, so the x-axis
        # label stays short — a repeated long label was overflowing the
        # narrow per-panel width and visually colliding with the neighboring
        # panel's label.
        ax.set_xlabel(r"Paired $\Delta$QoE", fontsize=STACK_PANEL_LABELSIZE)
        ax.set_title("vs. " + baseline, fontsize=STACK_PANEL_LABELSIZE, pad=10)
        ax.tick_params(axis="x", labelsize=STACK_PANEL_TICKSIZE)
        ax.grid(True, axis="x", alpha=0.4, linestyle="--")
        if ys:
            ax.set_ylim(-0.6, len(ys) - 0.4)
    _save(fig, out_dir, "fig_forest_delta_qoe_dual_baseline")
    plt.close(fig)


def export_summary_csv(df: pd.DataFrame, out_dir: Path) -> None:
    rng = np.random.default_rng(0)
    lines = []
    for m in sorted(df["Method"].unique()):
        sub = df[df["Method"] == m]
        for col in ("QoE", "Rebuffer", "Switch", "VMAF"):
            mu, lo, hi = _bootstrap_ci(sub[col].values.astype(float), rng=rng)
            lines.append(
                {
                    "Method": m,
                    "Metric": col,
                    "Mean": mu,
                    "CI95_lo": lo,
                    "CI95_hi": hi,
                    "n_episodes": len(sub),
                }
            )
    out = pd.DataFrame(lines)
    p = out_dir / "summary_bootstrap_v12.csv"
    out.to_csv(p, index=False)
    print(f"Wrote {p}")


def _summarize_wide_bootstrap(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for m in sorted(df["Method"].unique()):
        sub = df[df["Method"] == m]
        row: dict = {"Method": m}
        for col in ("QoE", "VMAF", "Rebuffer", "Switch"):
            mu, lo, hi = _bootstrap_ci(sub[col].values.astype(float), rng=rng)
            row[f"{col}_mean"], row[f"{col}_lo"], row[f"{col}_hi"] = mu, lo, hi
            row["n"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_cell(mu: float, lo: float, hi: float, digits: int) -> str:
    """Two-line CI cell via \\cici (matches headline tables; avoids column overflow)."""
    if digits <= 0:
        mtxt = f"{mu:.0f}"
        lotxt = f"{lo:.0f}"
        hitxt = f"{hi:.0f}"
    else:
        mtxt = f"{mu:.{digits}f}"
        lotxt = f"{lo:.{digits}f}"
        hitxt = f"{hi:.{digits}f}"
    return rf"\cici{{{mtxt}}}{{{lotxt}}}{{{hitxt}}}"


def export_latex_main_table(summary_wide: pd.DataFrame, tex_dir: Path) -> None:
    tex_dir.mkdir(parents=True, exist_ok=True)
    order = [m for m in ORDER_MAIN if m in set(summary_wide["Method"])]
    # Table style is shared with the hand-written fragments in main.ltx:
    # tabularx at \linewidth, bold header cells, and no embedded note row
    # (main.ltx supplies every note through \tablenote).
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xccccl@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{QoE}$\uparrow$ & \textbf{VMAF}$\uparrow$ & "
        r"\textbf{Rebuf.\ (\%)}$\downarrow$ & \textbf{Switches}$\downarrow$ & \textbf{Type} \\",
        r"\midrule",
    ]
    sw = summary_wide.set_index("Method")
    for m in order:
        r = sw.loc[m]
        q_cell = _fmt_cell(r["QoE_mean"], r["QoE_lo"], r["QoE_hi"], 0)
        v_cell = _fmt_cell(r["VMAF_mean"], r["VMAF_lo"], r["VMAF_hi"], 2)
        rb_cell = _fmt_cell(r["Rebuffer_mean"], r["Rebuffer_lo"], r["Rebuffer_hi"], 2)
        sw_cell = _fmt_cell(r["Switch_mean"], r["Switch_lo"], r["Switch_hi"], 1)
        row_tex = " & ".join(
            [
                LATEX_METHOD_COL.get(m, m.replace("_", r"\_")),
                q_cell,
                v_cell,
                rb_cell,
                sw_cell,
                METHOD_CLASS.get(m, "---"),
            ]
        )
        lines.append(row_tex + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    path = tex_dir / "table_main_results_ci.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def _paired_wilcoxon(
    df: pd.DataFrame, method: str, baseline: str, metric: str
) -> dict | None:
    try:
        from scipy import stats
    except ImportError:
        return None
    a = df[df["Method"] == method][["Video", "Episode", metric]].rename(columns={metric: "v_m"})
    b = df[df["Method"] == baseline][["Video", "Episode", metric]].rename(columns={metric: "v_b"})
    m = a.merge(b, on=["Video", "Episode"])
    if len(m) < 10:
        return None
    d_m = m["v_m"].values.astype(float)
    d_b = m["v_b"].values.astype(float)
    diff = d_m - d_b
    med = float(np.median(diff))
    try:
        stat, p = stats.wilcoxon(d_m, d_b, alternative="two-sided", zero_method="wilcox")
    except Exception:
        return None
    return {
        "Method": method,
        "Baseline": baseline,
        "Metric": metric,
        "n_pairs": len(m),
        "median_delta": med,
        "statistic": float(stat),
        "p_value": float(p),
    }


def _fmt_p_tex_val(p: float) -> str:
    if p >= 1e-3:
        return f"${p:.3f}$"
    e = int(math.floor(math.log10(p)))
    mant = p / (10**e)
    return rf"${mant:.2f}\times10^{{{e}}}$"


def _write_headline_wilcoxon_tex(
    *,
    headline_df: pd.DataFrame,
    all_episodes_df: pd.DataFrame,
    path_out: Path,
    delta_header: str,
    median_decimals: int,
) -> None:
    headline_methods = ["Proposed", "Proposed_Shielded", "Proposed_ShieldedRiskGate", "Proposed_ShieldedQoE"]
    headline_df = headline_df[headline_df["Method"].isin(headline_methods)]
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xcccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \multicolumn{2}{c}{\textbf{vs.\ Pensieve}} & "
        r"\multicolumn{2}{c}{\textbf{vs.\ RobustMPC}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        f" & \\textbf{{{delta_header}}} & \\textbf{{$p$}} & \\textbf{{{delta_header}}} & \\textbf{{$p$}} \\\\",
        r"\midrule",
    ]
    for m in [x for x in headline_methods if x in all_episodes_df["Method"].values]:
        row = [LATEX_METHOD_COL.get(m, m)]
        for b in ("Pensieve", "RobustMPC"):
            sub = headline_df[(headline_df["Method"] == m) & (headline_df["Baseline"] == b)]
            if sub.empty:
                row.extend(["---", "---"])
            else:
                p = float(sub["p_value"].values[0])
                med = float(sub["median_delta"].values[0])
                pv = _fmt_p_tex_val(p)
                ms = f"{med:.{median_decimals}f}"
                row.append(f"${ms}$")
                row.append(pv)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    path_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path_out}")


def export_paired_statistics(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    baselines = [b for b in ("Pensieve", "RobustMPC", "Genie") if b in df["Method"].values]
    methods_compare = [
        m
        for m in ORDER_MAIN
        if m not in baselines and m in df["Method"].values and not m.startswith("Ablation")
    ]
    methods_compare += ["Ablation_Base", "Ablation_Future", "Ablation_Lyap"]
    methods_compare = list(dict.fromkeys(m for m in methods_compare if m in df["Method"].values))

    records = []
    for metric in ("QoE", "Rebuffer"):
        for b in baselines:
            for m in methods_compare:
                if m == b:
                    continue
                row = _paired_wilcoxon(df, m, b, metric)
                if row:
                    records.append(row)
    out_df = pd.DataFrame(records)
    csv_p = tables_dir / "paired_wilcoxon_v12.csv"
    out_df.to_csv(csv_p, index=False)
    print(f"Wrote {csv_p}")

    if out_df.empty or "Metric" not in out_df.columns:
        print("WARNING: paired Wilcoxon table empty (is scipy installed?). Skipping headline TeX.")
        return

    hq = out_df[(out_df["Metric"] == "QoE") & (out_df["Baseline"].isin(["Pensieve", "RobustMPC"]))]
    _write_headline_wilcoxon_tex(
        headline_df=hq,
        all_episodes_df=df,
        path_out=tables_dir / "table_paired_wilcoxon_qoe_headline.tex",
        delta_header=r"med.\ $\Delta$QoE",
        median_decimals=1,
    )

    hb = out_df[(out_df["Metric"] == "Rebuffer") & (out_df["Baseline"].isin(["Pensieve", "RobustMPC"]))]
    _write_headline_wilcoxon_tex(
        headline_df=hb,
        all_episodes_df=df,
        path_out=tables_dir / "table_paired_wilcoxon_rebuffer_headline.tex",
        delta_header=r"med.\ $\Delta$Rebuf.\ (\%)",
        median_decimals=2,
    )


def export_abstract_macros(wide: pd.DataFrame, tables_dir: Path) -> None:
    """Write LaTeX ``providecommand`` macros for prose / abstract sync."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    sw = wide.set_index("Method")

    def gm(method: str, key: str) -> float:
        return float(sw.loc[method, f"{key}_mean"])

    rb_shield_sets = []
    for m in ("Proposed_Shielded", "Proposed_ShieldedRiskGate", "Proposed_ShieldedQoE"):
        if m in sw.index:
            rb_shield_sets.append(gm(m, "Rebuffer"))
    band_lo = min(rb_shield_sets) if rb_shield_sets else 0.0
    band_hi = max(rb_shield_sets) if rb_shield_sets else 0.0

    lines = [
        "% macros_v12.tex — auto-generated by scripts/generate_figures_v12.py",
        r"\makeatletter",
        rf"\providecommand{{\VNepisodes}}{{80}}",
        rf"\providecommand{{\VQoNoShield}}{{{gm('Proposed', 'QoE'):.0f}}}",
        rf"\providecommand{{\VQoShield}}{{{gm('Proposed_Shielded', 'QoE'):.0f}}}",
        rf"\providecommand{{\VQoRiskGate}}{{{gm('Proposed_ShieldedRiskGate', 'QoE'):.0f}}}",
        rf"\providecommand{{\VQoPen}}{{{gm('Pensieve', 'QoE'):.0f}}}",
        rf"\providecommand{{\VQoMPC}}{{{gm('RobustMPC', 'QoE'):.0f}}}",
        rf"\providecommand{{\VQoGenie}}{{{gm('Genie', 'QoE'):.0f}}}",
        rf"\providecommand{{\VRbNoShield}}{{{gm('Proposed', 'Rebuffer'):.2f}}}",
        rf"\providecommand{{\VRbShieldLo}}{{{band_lo:.2f}}}",
        rf"\providecommand{{\VRbShieldHi}}{{{band_hi:.2f}}}",
        rf"\providecommand{{\VRbPen}}{{{gm('Pensieve', 'Rebuffer'):.2f}}}",
        rf"\providecommand{{\VRbMPC}}{{{gm('RobustMPC', 'Rebuffer'):.2f}}}",
        r"\makeatother",
    ]
    mp = tables_dir / "macros_v12.tex"
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {mp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--decisions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figure assets without rewriting tables, statistics, or macros.",
    )
    args = parser.parse_args()

    new_root = _repo_new()
    stats_path = args.stats or (new_root / "results" / "detailed_stats_master_v12_v12_policy.csv")
    dec_path = args.decisions or (new_root / "results" / "decision_log_v12_v12_policy.csv")
    paper_dir = Path(__file__).resolve().parents[1]
    out_dir = args.out or (paper_dir / "figures")
    tables_dir = paper_dir / "tables"

    _setup_matplotlib()
    df = pd.read_csv(stats_path)
    decisions = pd.read_csv(dec_path)

    print(f"Loaded {len(df)} episode rows from {stats_path}")
    if not args.figures_only:
        rng_tab = np.random.default_rng(0)
        wide = _summarize_wide_bootstrap(df, rng_tab)

    # Figures first so a later table export failure cannot skip high-quality redraws.
    fig_tradeoff_with_ci(df, out_dir)
    fig_ecdf(df, out_dir, "QoE", "Session QoE (sum)", "fig_cdf_qoe")
    fig_ecdf(df, out_dir, "Rebuffer", "Rebuffer ratio (% of session)", "fig_cdf_rebuffer")
    fig_shield_intervention(decisions, out_dir)
    fig_ablation_bars(df, out_dir)
    fig_per_video_heatmap(df, out_dir)
    fig_forest_two_baselines(df, out_dir)
    fig_timeseries_compare(decisions, out_dir, video="sintel", episode=0)

    if args.figures_only:
        print("Done (figures only).")
        return

    export_summary_csv(df, out_dir)

    export_latex_main_table(wide, tables_dir)
    export_paired_statistics(df, tables_dir, out_dir)
    export_abstract_macros(wide, tables_dir)
    print("Done.")


if __name__ == "__main__":
    main()
