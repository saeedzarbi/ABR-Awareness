#!/usr/bin/env python3
"""
Generate publication-quality figures from v12 evaluation CSVs.

Inputs (default paths relative to repo new/):
  results/detailed_stats_master_v12_v12_policy.csv
  results/decision_log_v12_v12_policy.csv

Outputs:
  new/src/paper/figures_v12/*.pdf and *.png
  new/src/paper/figures_v12/summary_bootstrap_v12.csv
  new/src/paper/tables_v12/table_main_results_ci.tex
  new/src/paper/tables_v12/table_paired_wilcoxon_qoe_headline.tex
  new/src/paper/tables_v12/table_paired_wilcoxon_rebuffer_headline.tex
  new/src/paper/tables_v12/macros_v12.tex (abstract-ready scalars)
  new/src/paper/tables_v12/paired_wilcoxon_v12.csv

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

ORDER_ABLATION = ["Ablation_Base", "Ablation_Future", "Ablation_Lyap", "Proposed"]

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


def _setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "axes.titleweight": "600",
            "axes.labelweight": "500",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": "#E0E0E0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.borderaxespad": 0.5,
        }
    )


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

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
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
            elinewidth=1.0,
            capsize=2.5,
            markersize=7 if m != "Genie" else 8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
            label=DISPLAY_NAME.get(m, m),
        )

    ax.set_xlabel("Mean rebuffer ratio (% of session)")
    ax.set_ylabel("Session QoE (sum)")
    ax.set_title("Safety–QoE trade-off (bootstrap 95% CI)")
    ax.grid(True, alpha=0.45, linestyle="--")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fontsize=7.2,
        ncol=1,
        handletextpad=0.4,
    )
    fig.tight_layout()
    _save(fig, out_dir, "fig_tradeoff_qoe_rebuffer")
    plt.close(fig)


def fig_ecdf(df: pd.DataFrame, out_dir: Path, column: str, xlabel: str, stem: str) -> None:
    methods = [m for m in ORDER_MAIN if m in set(df["Method"].unique())]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for m in methods:
        sub = np.sort(df[df["Method"] == m][column].values.astype(float))
        y = np.arange(1, len(sub) + 1) / len(sub)
        ax.step(sub, y, where="post", color=COLORS.get(m, "#444444"), lw=1.45, label=DISPLAY_NAME.get(m, m))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ttl = "Episode QoE" if column == "QoE" else "Episode rebuffer %"
    ax.set_title(f"CDF of {ttl}")
    ax.grid(True, alpha=0.45, linestyle="--")
    ax.legend(loc="lower right", fontsize=7.0, ncol=1)
    fig.tight_layout()
    _save(fig, out_dir, stem)
    plt.close(fig)


def fig_switches_violin(df: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal mean switches per episode with bootstrap 95% CI (matplotlib-only).

    Filename remains ``fig_switches_violin`` for stable LaTeX includes.
    """
    methods = [m for m in ORDER_MAIN if m in set(df["Method"].unique())]
    if not methods:
        return

    rng = np.random.default_rng(42)
    labels = [DISPLAY_NAME[m] for m in methods]
    means: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []
    for m in methods:
        sub = df[df["Method"] == m]["Switch"].values.astype(float)
        mu, lo, hi = _bootstrap_ci(sub, rng=rng)
        means.append(mu)
        err_lo.append(mu - lo)
        err_hi.append(hi - mu)

    y = np.arange(len(methods), dtype=float)
    fig_h = max(4.2, 0.38 * len(methods) + 1.2)
    fig, ax = plt.subplots(figsize=(5.85, fig_h))

    for i, m in enumerate(methods):
        c = COLORS.get(m, "#444444")
        ax.errorbar(
            means[i],
            y[i],
            xerr=np.array([[err_lo[i]], [err_hi[i]]]),
            fmt="o",
            color=c,
            ecolor=c,
            capsize=2.8,
            markersize=5.8,
            elinewidth=1.05,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlabel("Bitrate switches per episode")
    ax.set_title("Mean bitrate switches (bootstrap 95% CI)")
    ax.grid(True, axis="x", alpha=0.45, linestyle="--")
    pad = 0.55
    ax.set_ylim(y.min() - pad, y.max() + pad)
    fig.tight_layout()
    _save(fig, out_dir, "fig_switches_violin")
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
    labels = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    colors = [COLORS.get(r[1], "#888888") for r in rows]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.6, height=0.65)
    ax.set_xlabel("Shield intervention rate (% of chunk steps)")
    ax.set_title("Runtime shield activity")
    ax.grid(True, axis="x", alpha=0.45, linestyle="--")
    fig.tight_layout()
    _save(fig, out_dir, "fig_shield_intervention_rate")
    plt.close(fig)


def fig_ablation_bars(df: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in ORDER_ABLATION if m in set(df["Method"].unique())]
    if len(methods) < 2:
        return
    agg = df[df["Method"].isin(methods)].groupby("Method", as_index=False)[["QoE", "Rebuffer"]].mean()
    agg["Label"] = agg["Method"].map(lambda m: DISPLAY_NAME.get(m, m))
    order = [DISPLAY_NAME[m] for m in methods if m in set(agg["Method"])]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for ax, col, title in zip(
        axes,
        ["QoE", "Rebuffer"],
        ["Mean session QoE", "Mean rebuffer (%)"],
    ):
        sub = agg.set_index("Method").loc[[m for m in methods if m in agg["Method"].values]].reset_index()
        sub["Label"] = sub["Method"].map(lambda m: DISPLAY_NAME.get(m, m))
        sub = sub.set_index("Label").loc[[l for l in order if l in sub["Label"].values]].reset_index()
        cols = [COLORS.get(m, "#888888") for m in sub["Method"]]
        ax.bar(sub["Label"], sub[col], color=cols, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18, labelsize=7.5)
        ax.grid(True, axis="y", alpha=0.45, linestyle="--")
    fig.suptitle("Ablations vs. full proposed (episode means)", fontsize=11, fontweight="600", y=1.02)
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
    # short video names
    pv.index = [str(i)[:10] for i in pv.index]

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    im = ax.imshow(pv.values, aspect="auto", cmap="magma", vmin=pv.values.min(), vmax=pv.values.max())
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([DISPLAY_NAME.get(c, c) for c in pv.columns], rotation=35, ha="right", fontsize=6.8)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(list(pv.index), fontsize=8)
    ax.set_title("Mean QoE by test video")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("QoE")
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
    fig, axes = plt.subplots(4, 1, figsize=(6.8, 5.4), sharex=True, constrained_layout=True)
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

        axes[0].plot(chunks, tp, "-o", ms=2.2, lw=1.0, color=color, label=DISPLAY_NAME.get(m, m), alpha=0.9)
        axes[1].step(chunks, buf, where="mid", color=color, lw=1.2, label=DISPLAY_NAME.get(m, m), alpha=0.9)
        axes[2].step(chunks, br, where="mid", color=color, lw=1.2, label=DISPLAY_NAME.get(m, m), alpha=0.9)
        axes[3].bar(
            chunks + 0.18 * (row - 0.5),
            rb,
            width=0.35,
            color=color,
            alpha=0.65,
            label=DISPLAY_NAME.get(m, m),
        )

    axes[0].set_ylabel("Throughput (Mb/s)")
    axes[1].set_ylabel("Buffer (s)")
    axes[2].set_ylabel("Bitrate (kbps)")
    axes[3].set_ylabel("Rebuf. (s)")
    axes[-1].set_xlabel("Chunk index")
    axes[0].set_title(f"Representative trace ({video}, episode {episode})")
    for ax in axes:
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.legend(loc="upper right", fontsize=6.5)
    _save(fig, out_dir, "fig_timeseries_proposed_vs_shielded")
    plt.close(fig)


def fig_paired_delta_vs_baseline(df: pd.DataFrame, out_dir: Path, baseline: str = "Pensieve") -> None:
    if baseline not in df["Method"].values:
        return
    base = df[df["Method"] == baseline][["Video", "Episode", "QoE"]].rename(columns={"QoE": "QoE_b"})
    targets = ["Proposed", "Proposed_Shielded", "Proposed_ShieldedRiskGate", "Proposed_ShieldedQoE"]
    rows = []
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    positions = []
    labels = []
    pos = 0
    for m in targets:
        if m not in df["Method"].values:
            continue
        a = df[df["Method"] == m][["Video", "Episode", "QoE"]].rename(columns={"QoE": "QoE_m"})
        merged = a.merge(base, on=["Video", "Episode"])
        delta = (merged["QoE_m"] - merged["QoE_b"]).values.astype(float)
        if len(delta) == 0:
            continue
        _, lo, hi = _bootstrap_ci(delta, rng=rng)
        mean = float(np.mean(delta))
        ax.barh(pos, mean, xerr=[[mean - lo], [hi - mean]], color=COLORS.get(m, "#444444"), capsize=3, height=0.55)
        positions.append(pos)
        labels.append(DISPLAY_NAME.get(m, m))
        pos += 1
    if not labels:
        plt.close(fig)
        return
    ax.axvline(0, color="#333333", lw=0.9, linestyle="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"ΔQoE vs. {baseline} (paired episodes)")
    ax.set_title("Effect size with bootstrap 95% CI")
    ax.grid(True, axis="x", alpha=0.45, linestyle="--")
    fig.tight_layout()
    safe = "".join(ch if ch.isalnum() else "_" for ch in baseline.lower())
    _save(fig, out_dir, f"fig_paired_delta_qoe_vs_{safe}")
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

    fig, axes = plt.subplots(1, len(baselines), figsize=(5.6 * len(baselines), 3.35), squeeze=False)
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
                height=0.52,
                color=c,
                ecolor=c,
                capsize=2.8,
                edgecolor="white",
                linewidth=0.35,
            )
            ys.append(DISPLAY_NAME.get(m, m))
            pos += 1
        ax.axvline(0.0, color="#222222", lw=0.9, linestyle="--")
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys, fontsize=8)
        ax.set_xlabel(r"Paired $\Delta$QoE vs.\ " + baseline)
        ax.set_title("vs. " + baseline)
        ax.grid(True, axis="x", alpha=0.4, linestyle="--")
    fig.suptitle(r"Paired QoE shifts (bootstrap 95% CI on mean $\Delta$)", fontsize=11, fontweight="600")
    fig.tight_layout()
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
    if digits <= 0:
        mtxt = f"{mu:.0f}"
        bracket = rf"\scriptsize $[{lo:.0f},\,{hi:.0f}]$"
    else:
        mtxt = f"{mu:.{digits}f}"
        bracket = rf"\scriptsize $[{lo:.{digits}f},\,{hi:.{digits}f}]$"
    return f"{mtxt}\\,{bracket}"


def export_latex_main_table(summary_wide: pd.DataFrame, tex_dir: Path) -> None:
    tex_dir.mkdir(parents=True, exist_ok=True)
    order = [m for m in ORDER_MAIN if m in set(summary_wide["Method"])]
    lines = [
        r"\begin{tabular}{@{}lccccl@{}}",
        r"\toprule",
        r"Method & QoE$\uparrow$ & VMAF$\uparrow$ & Rebuf.\ (\%)$\downarrow$ & Switches$\downarrow$ & Type \\",
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
        r"\multicolumn{6}{l}{\footnotesize Interval: bootstrap 95\% CI on episode means ($n{=}80$ per row). "
        r"$^{\dagger}$Shield-aware variant (elevated bitrate switches).}\\",
        r"\end{tabular}",
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
    footer: str,
) -> None:
    headline_methods = ["Proposed", "Proposed_Shielded", "Proposed_ShieldedRiskGate", "Proposed_ShieldedQoE"]
    headline_df = headline_df[headline_df["Method"].isin(headline_methods)]
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & \multicolumn{2}{c}{vs.\ Pensieve} & \multicolumn{2}{c}{vs.\ RobustMPC} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        f" & {delta_header} & $p$ & {delta_header} & $p$ \\\\",
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
    lines += [r"\bottomrule", rf"\multicolumn{{5}}{{l}}{{\footnotesize {footer}}} \\", r"\end{tabular}"]
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

    hq = out_df[(out_df["Metric"] == "QoE") & (out_df["Baseline"].isin(["Pensieve", "RobustMPC"]))]
    _write_headline_wilcoxon_tex(
        headline_df=hq,
        all_episodes_df=df,
        path_out=tables_dir / "table_paired_wilcoxon_qoe_headline.tex",
        delta_header=r"med.\ $\Delta$QoE",
        median_decimals=1,
        footer=(
            r"Two-sided paired Wilcoxon signed-rank; $\Delta$QoE is the per-episode "
            r"paired difference (method minus baseline)."
        ),
    )

    hb = out_df[(out_df["Metric"] == "Rebuffer") & (out_df["Baseline"].isin(["Pensieve", "RobustMPC"]))]
    _write_headline_wilcoxon_tex(
        headline_df=hb,
        all_episodes_df=df,
        path_out=tables_dir / "table_paired_wilcoxon_rebuffer_headline.tex",
        delta_header=r"med.\ $\Delta$Rebuf.\ (\%)",
        median_decimals=2,
        footer=(
            r"Same paired episodes; $\Delta$Rebuffer is the per-episode difference in reported "
            r"mean rebuffer ratio (\%). Negative values favor the row method (fewer stalls)."
        ),
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
    args = parser.parse_args()

    new_root = _repo_new()
    stats_path = args.stats or (new_root / "results" / "detailed_stats_master_v12_v12_policy.csv")
    dec_path = args.decisions or (new_root / "results" / "decision_log_v12_v12_policy.csv")
    paper_dir = Path(__file__).resolve().parents[1]
    out_dir = args.out or (paper_dir / "figures_v12")
    tables_dir = paper_dir / "tables_v12"

    _setup_matplotlib()
    df = pd.read_csv(stats_path)
    decisions = pd.read_csv(dec_path)

    print(f"Loaded {len(df)} episode rows from {stats_path}")
    rng_tab = np.random.default_rng(0)
    wide = _summarize_wide_bootstrap(df, rng_tab)
    export_latex_main_table(wide, tables_dir)
    export_paired_statistics(df, tables_dir, out_dir)
    export_abstract_macros(wide, tables_dir)

    fig_tradeoff_with_ci(df, out_dir)
    fig_ecdf(df, out_dir, "QoE", "Session QoE (sum)", "fig_cdf_qoe")
    fig_ecdf(df, out_dir, "Rebuffer", "Rebuffer ratio (% of session)", "fig_cdf_rebuffer")
    fig_switches_violin(df, out_dir)
    fig_shield_intervention(decisions, out_dir)
    fig_ablation_bars(df, out_dir)
    fig_per_video_heatmap(df, out_dir)
    fig_paired_delta_vs_baseline(df, out_dir, baseline="Pensieve")
    fig_forest_two_baselines(df, out_dir)
    fig_timeseries_compare(decisions, out_dir, video="sintel", episode=0)
    export_summary_csv(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
