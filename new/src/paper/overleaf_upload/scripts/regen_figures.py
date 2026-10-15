"""Re-render paper figures at single-column width (3.5in), Times-like serif, 8pt.
Data loading reuses the repo's own functions; plotted values unchanged."""
import sys, csv, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

NEW = Path("/workspace/ABR-Awareness/new")
sys.path.insert(0, str(NEW)); sys.path.insert(0, str(NEW/"src/paper")); sys.path.insert(0, str(NEW/"src/paper/scripts"))
OUT = Path("/workspace/figregen/out"); OUT.mkdir(exist_ok=True)

import make_cps_paper_assets as A
import analyze_vmaf_ladder_perchunk as L

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "grid.linewidth": 0.4, "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "legend.handlelength": 1.6, "legend.borderpad": 0.3, "legend.labelspacing": 0.3,
}
mpl.rcParams.update(RC)
W = 3.5
C, AL = A.COLORS, A.ARM_LABEL

def save(fig, name):
    fig.savefig(OUT/name, bbox_inches="tight", pad_inches=0.02); plt.close(fig); print("wrote", name)

def tag(ax, s):
    ax.set_title(s, loc="left", fontsize=8, pad=2)

# ---------- overview ----------
def overview():
    groups = []
    for label, t in [("5G greedy", "greedy_5g"), ("5G BBA", "bba_5g"), ("BB greedy", "greedy_bb")]:
        s = A.load(t)
        if s is None: continue
        cs = A.cmp_(s, "certified_vs_safety")
        groups.append({"label": label, "bw": -cs["bandwidth_reduction_pct"],
                       "reb": -cs["rebuffer_change_pct"], "vmaf_loss": max(0.0, -cs["vmaf_mean_diff"])})
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(W, 2.3), constrained_layout=True)
    b1 = ax.bar(x - w, [g["bw"] for g in groups], w, color="#0072B2", label="Bitrate saved (%)")
    b2 = ax.bar(x, [g["reb"] for g in groups], w, color="#009E73", label="Rebuffer reduced (%)")
    ax2 = ax.twinx()
    b3 = ax2.bar(x + w, [g["vmaf_loss"] for g in groups], w, color="#E69F00", edgecolor="#A05A00", linewidth=0.8, label="VMAF cost (pts, right axis)")
    ax2.plot(x + w, [g["vmaf_loss"] for g in groups], ls="none", marker="D", ms=3.5, color="#A05A00")
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("VMAF cost (pts)", color="#A05A00"); ax2.tick_params(axis="y", colors="#A05A00")
    for b in (b1, b2): ax.bar_label(b, fmt="%.1f", padding=1, fontsize=7)
    ax2.bar_label(b3, fmt="%.2f", padding=3, fontsize=7, color="#A05A00")
    ax.set_xticks(x); ax.set_xticklabels([g["label"] for g in groups])
    ax.set_ylabel("Certified vs. Safety change (%)")
    ax.set_ylim(0, max(g["reb"] for g in groups) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.legend(handles=[b1,b2,b3], loc="outside lower center", ncol=2, frameon=False, columnspacing=0.8, handlelength=1.0)
    save(fig, "fig_cps_overview.pdf")

# ---------- tradeoff / cdf (2 panels side by side) ----------
def tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15), constrained_layout=True)
    handles = None
    for ax, (title, t), lab in zip(axes, [("Greedy", "greedy_5g"), ("BBA", "bba_5g")], "ab"):
        rows = A.load_episodes(t)
        for arm in ("raw", "safety", "certified"):
            xs = [r["reb"] for r in rows[arm]]; ys = [r["vmaf"] for r in rows[arm]]
            ax.scatter(xs, ys, s=5, alpha=0.55, c=C[arm], label=AL[arm], edgecolors="none")
            ax.scatter([np.mean(xs)], [np.mean(ys)], s=30, c=C[arm], marker="D", edgecolors="k", linewidths=0.5, zorder=5)
        ax.set_xlabel("Rebuffering (s)"); ax.grid(alpha=0.25); tag(ax, f"({lab}) {title}")
        if handles is None: handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Mean VMAF")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, markerscale=2)
    save(fig, "fig_cps_tradeoff.pdf")

def cdf():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.0), sharey=True, constrained_layout=True)
    handles = None
    for ax, (title, t), lab in zip(axes, [("Greedy", "greedy_5g"), ("BBA", "bba_5g")], "ab"):
        rows = A.load_episodes(t)
        for arm in ("raw", "safety", "certified"):
            vals = np.sort([r["reb"] for r in rows[arm]]); y = np.arange(1, len(vals)+1)/len(vals)
            ax.plot(vals, y, color=C[arm], lw=1.2, label=AL[arm])
        ax.set_xlabel("Rebuffering (s)"); ax.set_xlim(left=0); ax.grid(alpha=0.25); tag(ax, f"({lab}) {title}")
        if handles is None: handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("CDF")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "fig_cps_cdf_reb.pdf")

# ---------- codesign (stacked) ----------
def codesign():
    def col(name, key):
        out = {}
        with open(A.RES/name/"episodes.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] == "certified": out[int(r["episode"])] = float(r[key])
        return out
    vA, rA = col("proposed_5g_regime", "vmaf_mean"), col("proposed_5g_regime", "rebuffer_total")
    vB, rB = col("proposed_cps_5g", "vmaf_mean"), col("proposed_cps_5g", "rebuffer_total")
    eps = sorted(set(vA) & set(vB))
    va, vb = np.array([vA[e] for e in eps]), np.array([vB[e] for e in eps])
    ra, rb = np.array([rA[e] for e in eps]), np.array([rB[e] for e in eps])
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.9), constrained_layout=True)
    ax = axes[0]
    ax.scatter(ra, va, s=7, alpha=0.55, c=C["eval"], label="Eval-only shield", edgecolors="none")
    ax.scatter(rb, vb, s=7, alpha=0.55, c=C["codesign"], label="Co-trained CPS", edgecolors="none")
    ax.scatter([ra.mean()], [va.mean()], s=36, c=C["eval"], marker="D", edgecolors="k", linewidths=0.5, zorder=5)
    ax.scatter([rb.mean()], [vb.mean()], s=36, c=C["codesign"], marker="D", edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlabel("Rebuffering (s)"); ax.set_ylabel("Mean VMAF")
    ax.legend(loc="center right", framealpha=0.92, markerscale=1.8); ax.grid(alpha=0.25); tag(ax, "(a)")
    ax = axes[1]
    ws = np.linspace(0, 20, 201)
    dQ = (vb.mean() - ws*rb.mean()) - (va.mean() - ws*ra.mean())
    wstar = (vb.mean() - va.mean()) / max(rb.mean() - ra.mean(), 1e-12)
    ax.plot(ws, dQ, color="#009E73", lw=1.4, label=r"$\Delta$QoE (co $-$ eval)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(wstar, color="#D55E00", ls="--", lw=1.0, label=rf"$w^\star={wstar:.2f}$")
    ax.set_xlabel(r"Stall weight $w$ (VMAF pts / s)"); ax.set_ylabel(r"$\Delta$QoE")
    ax.legend(loc="upper right", framealpha=0.92); ax.grid(alpha=0.25); tag(ax, "(b)")
    save(fig, "fig_cps_codesign.pdf")

# ---------- coverage violins ----------
def coverage():
    def per_ep(name):
        vals = []; p = A.RES/name/"episodes.csv"
        if not p.exists(): return vals
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["arm"] != "certified": continue
                c = r.get("conformal_coverage", "")
                if c not in ("", "nan"):
                    try: vals.append(float(c))
                    except ValueError: pass
        return vals
    hosts = [("Greedy", "greedy_5g"), ("BBA", "bba_5g"), ("BOLA", "bola_5g"), ("MPC", "mpc_5g"),
             ("Pensieve", "pensieve_5g"), ("Proposed", "proposed_5g_improved"), ("Co-trained", "proposed_cps_5g")]
    data, labels = [], []
    for label, t in hosts:
        v = per_ep(t)
        if v: data.append(v); labels.append(label)
    target = 0.90
    fig, ax = plt.subplots(figsize=(W, 2.3), constrained_layout=True)
    parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.8)
    for pc in parts["bodies"]: pc.set_facecolor("#0072B2"); pc.set_alpha(0.30)
    bp = ax.boxplot(data, widths=0.28, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.0), boxprops=dict(lw=0.6),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for box in bp["boxes"]: box.set(facecolor="#009E73", alpha=0.65)
    ax.axhline(target, color="#D55E00", ls="--", lw=1.0, label=rf"Target $1-\alpha={target:.2f}$")
    ax.set_xticks(np.arange(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Per-episode coverage"); ax.set_ylim(min(0.86, target-0.02), 1.005)
    ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower right", framealpha=0.92)
    save(fig, "fig_cps_coverage.pdf")

# ---------- coverage drift (same aggregation as original) + alternative ----------
def drift():
    d = pd.read_csv(NEW/"results/v18_certified/coverage_drift/chunks.csv")
    alpha, s0, sl, warm = 0.10, 28, 10, 20
    rc = d.dropna(subset=["rolling_coverage"]).groupby("chunk").rolling_coverage.mean()
    def base(ax):
        ax.axvspan(0, warm, color="#d9d9d9", alpha=0.85, label="Conformal warm-up")
        ax.axvspan(s0, s0+sl, color="#f4cccc", alpha=0.75, label="Injected shock")
        ax.axhline(1-alpha, color="#888888", ls="--", lw=0.9, label=rf"Target $1-\alpha={1-alpha:.2f}$")
        ax.set_xlabel("Chunk index"); ax.set_ylabel("Conformal coverage")
        ax.set_ylim(0.0, 1.02); ax.set_xlim(left=0)
    fig, ax = plt.subplots(figsize=(W, 2.1), constrained_layout=True)
    base(ax)
    ax.plot(rc.index, rc.values, color="#1f4e79", lw=1.2, label=r"Rolling mean ($W{=}20$)")
    h, l = ax.get_legend_handles_labels(); order = [0, 1, 3, 2]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", framealpha=0.92)
    save(fig, "fig_cps_coverage_drift.pdf")
    # alternative: per-chunk cross-episode coverage (no temporal window)
    v = d[d.valid == 1]; pc = v.groupby("chunk").covered.mean()
    fig, ax = plt.subplots(figsize=(W, 2.1), constrained_layout=True)
    base(ax)
    ax.plot(rc.index, rc.values, color="#1f4e79", lw=0.9, ls=":", label=r"Rolling mean ($W{=}20$)")
    ax.plot(pc.index, pc.values, color="#1f4e79", lw=1.2, marker="o", ms=2, label="Per-chunk mean (204 ep.)")
    ax.legend(loc="lower right", framealpha=0.92)
    save(fig, "fig_cps_coverage_drift_perchunk_ALT.pdf")

# ---------- per-chunk ablation (side by side) ----------
def perchunk():
    df = pd.read_csv(NEW/"results/perchunk_ablation/perchunk_ladder_ablation.csv")
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.7), constrained_layout=True)
    ax1, ax2 = axes
    x = np.arange(len(df)); w = 0.38
    ax1.bar(x - w/2, df.bitrate_saved_pct_pooled, w, label="pooled", color="#4C72B0")
    ax1.bar(x + w/2, df.bitrate_saved_pct_perchunk, w, label="per-chunk", color="#DD8452")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{e:.1f}" for e in df.epsilon])
    ax1.set_xlabel(r"perceptual budget $\varepsilon$ (VMAF pts)"); ax1.set_ylabel("bitrate saved (%)")
    ax1.legend(frameon=False); tag(ax1, "(a)")
    ax2.plot(df.epsilon, df.vmaf_cost_perchunk_shield, "o-", ms=3, label="per-chunk shield", color="#DD8452")
    ax2.plot(df.epsilon, df.vmaf_cost_pooled_shield, "s--", ms=3, label="pooled shield", color="#4C72B0")
    ax2.plot(df.epsilon, df.epsilon, ":", color="gray", label=r"budget $\varepsilon$")
    ax2.set_xlabel(r"perceptual budget $\varepsilon$ (VMAF pts)"); ax2.set_ylabel("realized per-chunk cost (pts)")
    ax2.legend(frameon=False); tag(ax2, "(b)")
    save(fig, "fig_cps_perchunk.pdf")

# ---------- eps/alpha ablation ----------
def ablation():
    import ablation_eps_alpha as E   # importable via path below
    def summ(sub):
        return E.HEADLINE/"summary.json" if sub is None else E.OUT/sub/"summary.json"
    eps_rows = []
    for eps in E.EPS_LIST:
        r = E.load(summ(None if abs(eps-E.ALPHA_FIXED_EPS) < 1e-9 else f"eps_{eps}")); r["epsilon"] = eps; eps_rows.append(r)
    alpha_rows = []
    for a in E.ALPHA_LIST:
        r = E.load(summ(None if abs(a-E.EPS_FIXED_ALPHA) < 1e-9 else f"alpha_{a}")); r["alpha"] = a; alpha_rows.append(r)
    mpl.rcParams.update(RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.0), layout="constrained")
    eps = [r["epsilon"] for r in eps_rows]; bw = [-r["bw_cs"] for r in eps_rows]
    ax1.plot(eps, bw, "o-", ms=3, color="#4C72B0")
    ax1.set_xlabel(r"budget $\varepsilon$ (VMAF pts)"); ax1.set_ylabel("bitrate saved (%)")
    ax1.grid(axis="y", alpha=0.3); tag(ax1, "(a)")
    al = [r["alpha"] for r in alpha_rows]; cov = [r["coverage"] for r in alpha_rows]; tgt = [1-a for a in al]
    x = np.arange(len(al)); w = 0.38
    ax2.bar(x - w/2, tgt, w, label=r"target $1{-}\alpha$", color="#BBBBBB")
    ax2.bar(x + w/2, cov, w, label="empirical", color="#4C72B0")
    ax2.set_xticks(x); ax2.set_xticklabels([f"{a:.2f}" for a in al])
    ax2.set_xlabel(r"miscoverage $\alpha$ ($\varepsilon{=}1.0$)"); ax2.set_ylabel("conformal coverage")
    ax2.set_ylim(0.70, 1.02); ax2.legend(frameon=False, loc="upper right"); ax2.grid(axis="y", alpha=0.3); tag(ax2, "(b)")
    save(fig, "fig_cps_ablation.pdf")
    print("eps", [(r["epsilon"], round(-r["bw_cs"],2)) for r in eps_rows], "alpha", [(r["alpha"], r["coverage"]) for r in alpha_rows])

# ---------- ladder ----------
def ladder():
    lad, pooled = L.build_ladders()
    mpl.rcParams.update(RC)
    B = L.BITRATES
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.1), layout="constrained")
    ax = axes[0]
    wide = lad.pivot_table(index=["video", "chunk"], columns="bitrate_kbps", values="vmaf")
    pw = pooled.pivot_table(index="video", columns="bitrate_kbps", values="frame_mean")
    x = np.arange(len(B)); colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, v in enumerate(L.VIDEOS):
        blk = wide.loc[v]
        ax.fill_between(x, blk[B].min(axis=0).to_numpy(), blk[B].max(axis=0).to_numpy(), alpha=0.20, color=colors[i], linewidth=0)
        ax.plot(x, pw.loc[v, B].to_numpy(), marker="o", markersize=3, color=colors[i], label=L.VIDEO_LABEL[v])
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("Representation bitrate (kb/s)"); ax.set_ylabel("VMAF"); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.92); tag(ax, "(a)")
    ax = axes[1]
    gaps, labels = [], []
    for lo, hi in zip(B[:-1], B[1:]):
        gaps.append((wide[hi]-wide[lo]).to_numpy()); labels.append(f"{lo}\u2192{hi}")
    pos = list(range(len(gaps)))
    parts = ax.boxplot(gaps, positions=pos, widths=0.6, showfliers=True, patch_artist=True,
                       medianprops={"color": "black", "lw": 0.9}, boxprops=dict(lw=0.6),
                       whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                       flierprops=dict(markersize=2, markeredgewidth=0.4))
    for p in parts["boxes"]: p.set_facecolor("#9ecae1"); p.set_alpha(0.8)
    ax.axhline(0.0, color="crimson", linewidth=0.9, linestyle="--")
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Adjacent representation pair (kb/s)"); ax.set_ylabel("Per-chunk VMAF gap"); ax.grid(alpha=0.3)
    tag(ax, "(b)")
    save(fig, "fig_ladder_perchunk.pdf")

if __name__ == "__main__":
    sys.path.insert(0, str(NEW/"src/evaluation"))
    for f in (overview, tradeoff, cdf, codesign, coverage, drift, perchunk, ablation, ladder):
        f()
