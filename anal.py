import re
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from tabulate import tabulate

# --- Paths ---
LOGS = {
    "Full": "results/ablation_logs/full.log",
    "No Content": "results/ablation_logs/no_content.log",
    "No VMAF": "results/ablation_logs/no_vmaf.log",
    "Pensieve-Like": "results/ablation_logs/pensieve_like.log",
}
SUMMARY_CSV = "results/ablation_logs/summary.csv"  # optional
OUT_DIR = "results/ablation_logs"

os.makedirs(OUT_DIR, exist_ok=True)

# --- Helpers ---
VAL_REWARD_RE = re.compile(r"Val Reward:\s*([+-]?\d+(?:\.\d+)?)")
BEST_REWARD_RE = re.compile(r"Best Val Reward:\s*([+-]?\d+(?:\.\d+)?)")


def parse_log_rewards(path):
    """Parse a training log and return the list of validation rewards per update.
    Falls back to best-value-only if no per-update values are present.
    """
    vals = []
    best = None
    if not os.path.exists(path):
        return vals, best
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = VAL_REWARD_RE.search(line)
            if m:
                vals.append(float(m.group(1)))
            mb = BEST_REWARD_RE.search(line)
            if mb:
                best = float(mb.group(1))
    return vals, best

# Parse all logs
series = {}
best_map = {}
for name, p in LOGS.items():
    vals, best = parse_log_rewards(p)
    series[name] = np.array(vals, dtype=float)
    best_map[name] = best

# Align lengths for paired tests
min_len = min((len(v) for v in series.values() if len(v) > 0), default=0)
if min_len == 0:
    raise SystemExit("No per-update validation rewards found in logs. Ensure logs contain 'Val Reward: <num>'.")

series_aligned = {k: v[:min_len] for k, v in series.items()}

# Build DataFrame of per-update rewards
df = pd.DataFrame({k: v for k, v in series_aligned.items()})
df.index.name = "update"

a_results = []
comparisons = [("Full", "No Content"), ("Full", "No VMAF"), ("Full", "Pensieve-Like")]
for a, b in comparisons:
    t, p = ttest_rel(df[a], df[b])
    a_results.append({
        "Comparison": f"{a} vs {b}",
        "n(updates)": len(df),
        "mean_A": df[a].mean(),
        "mean_B": df[b].mean(),
        "t": t,
        "p": p,
        "Significant(p<0.05)": p < 0.05,
    })

# Optional: read summary best values
summary_df = None
if os.path.exists(SUMMARY_CSV):
    try:
        summary_df = pd.read_csv(SUMMARY_CSV)
    except Exception:
        summary_df = None

# Save outputs
stats_path = os.path.join(OUT_DIR, "ablation_stats_per_update.csv")
df.to_csv(stats_path)

report_rows = []
for row in a_results:
    report_rows.append({
        "Comparison": row["Comparison"],
        "n": row["n(updates)"],
        "mean_A": f"{row['mean_A']:.2f}",
        "mean_B": f"{row['mean_B']:.2f}",
        "t": f"{row['t']:.2f}",
        "p": f"{row['p']:.3g}",
        "Significant": "Yes" if row["Significant(p<0.05)"] else "No",
    })

report_df = pd.DataFrame(report_rows)
report_path = os.path.join(OUT_DIR, "ablation_ttests.csv")
report_df.to_csv(report_path, index=False)

print("\n📊 Paired t-test across updates (validation reward per update)")
print(tabulate(report_df, headers="keys", tablefmt="github", showindex=False))

if summary_df is not None:
    print("\n📄 Best Val Reward (from summary.csv), if available:")
    print(summary_df.to_string(index=False))

# Optional: emit a LaTeX table for the paper
latex_path = os.path.join(OUT_DIR, "ablation_ttests.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(report_df.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\n✅ Saved: {stats_path}, {report_path}, {latex_path}")