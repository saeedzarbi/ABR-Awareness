import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from tabulate import tabulate

BASE = "results/ablation_runs"
FILES = {
    "Full": os.path.join(BASE, "full", "val_per_update.csv"),
    "No Content": os.path.join(BASE, "no_content", "val_per_update.csv"),
    "No VMAF": os.path.join(BASE, "no_vmaf", "val_per_update.csv"),
    "Pensieve-Like": os.path.join(BASE, "pensieve_like", "val_per_update.csv"),
}

series = {}
for name, path in FILES.items():
    if not os.path.exists(path):
        raise SystemExit(f"Missing file: {path}. Did you run train_ablation with --tag {name.lower().replace(' ','_')}?")
    df = pd.read_csv(path)
    if not set(["update","val_reward"]).issubset(df.columns):
        raise SystemExit(f"Bad CSV format in {path}")
    series[name] = df.sort_values("update")["val_reward"].to_numpy()

# Align lengths
min_len = min(len(v) for v in series.values())
for k in list(series.keys()):
    series[k] = series[k][:min_len]

# Build dataframe of aligned series
A = pd.DataFrame(series)
A.index.name = "update"

# Paired tests vs Full
def paired(a, b):
    n = len(a)
    if n < 2 or np.allclose(a, b) or np.isclose(np.std(a-b), 0):
        return n, a.mean(), b.mean(), np.nan, np.nan, False
    t, p = ttest_rel(a, b)
    return n, a.mean(), b.mean(), t, p, (p < 0.05)

rows = []
for other in ["No Content","No VMAF","Pensieve-Like"]:
    n, ma, mb, t, p, sig = paired(A["Full"].values, A[other].values)
    rows.append({
        "Comparison": f"Full vs {other}",
        "n(updates)": n,
        "mean_Full": f"{ma:.2f}",
        "mean_{other}": f"{mb:.2f}",
        "t": f"{t:.2f}",
        "p": f"{p:.3g}",
        "Significant": "Yes" if sig else "No",
    })

print("\n📊 Paired t-test over validation reward per update")
print(tabulate(rows, headers="keys", tablefmt="github", showindex=False))

# Export CSV + LaTeX
out_dir = BASE
A.to_csv(os.path.join(out_dir, "val_reward_per_update_aligned.csv"))
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ablation_ttests_over_updates.csv"), index=False)
with open(os.path.join(out_dir, "ablation_ttests_over_updates.tex"), "w", encoding="utf-8") as f:
    f.write(pd.DataFrame(rows).to_latex(index=False))

print(f"\n✅ Saved CSV/LaTeX to {out_dir}")
