"""
Conditional VMAF-equivalence analysis for the V14 shield (reviewer-response).

The review's most incisive statistical objection (P0.5 / 3.2): the "quality-
preserving" claim was tested on the *session-mean* VMAF difference, averaged over
all chunks including the ~97% the shield never touches. With a ~3% intervention
rate a +-2 VMAF-point margin on the session mean corresponds to a +-60-point
margin on the chunks that were actually repaired, making the equivalence result
close to vacuous.

This script conditions on the intervened chunks and reports the estimand that
the claim is really about:

    VMAF_given_up  =  V_k(a_raw) - V_k(a_exec)   for chunks with Intervened == 1

For each VMAF-aware / legacy / highest-feasible arm it reports, on intervened
chunks only:
  * number of interventions,
  * mean VMAF given up + bootstrap 95% CI,
  * fraction of interventions exceeding a JND-scale margin,
  * a one-sided non-inferiority check (mean loss < margin),
and, for the isolation A/B, the per-chunk executed-VMAF difference between the
VMAF-aware and highest-feasible-index arms (expected to be exactly 0 on a
monotone ladder).

The margin (default 6.0 VMAF points, ~1 JND) is on the CONDITIONAL scale and is
still post-hoc; it is reported as exploratory, not preregistered.

Reads : results/v14_shielded_qoe/online_decisions.csv  (or --decisions-csv)
Writes: results/v14_shielded_qoe/vmaf_equivalence_conditional_v14.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.paths import get_paths

PATHS = get_paths()

# ~1 just-noticeable-difference on the VMAF scale, on the CONDITIONAL (intervened
# chunk) scale. Post-hoc / exploratory, not preregistered.
MARGIN = 6.0

ARMS = [
    "vmaf_aware_tol0.8_bud08",
    "vmaf_aware_tol1.0_bud08",
    "shield_legacy",
    "highest_feasible_tol0.8",
    "highest_feasible_tol1.0",
]


def _bootstrap_ci(values, n_boot=5000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([values[rng.integers(0, values.size, values.size)].mean() for _ in range(n_boot)])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions-csv", type=str,
                        default=str(PATHS["results"] / "v14_shielded_qoe" / "online_decisions.csv"))
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.decisions_csv)
    if not path.exists():
        print(f"[ERROR] missing {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    available = set(df["Method"].unique())

    rows = []
    print(f"Conditional VMAF-equivalence (margin = +-{args.margin} VMAF pts, intervened chunks only)\n")
    for arm in ARMS:
        if arm not in available:
            continue
        sub = df[(df["Method"] == arm) & (df["Intervened"] == 1)]
        loss = sub["VMAF_given_up"].to_numpy(float)
        n_int = int(len(loss))
        if n_int == 0:
            rows.append({"arm": arm, "n_interventions": 0, "mean_vmaf_given_up": 0.0,
                         "ci_lo": 0.0, "ci_hi": 0.0, "frac_exceed_margin": 0.0,
                         "noninferior": True})
            print(f"  {arm:26s} n_int=0 (never intervenes)")
            continue
        mean = float(loss.mean())
        lo, hi = _bootstrap_ci(loss)
        frac_exceed = float((loss > args.margin).mean())
        # Non-inferiority: upper 95% bootstrap bound of the mean loss < margin.
        noninferior = bool(hi < args.margin)
        rows.append({
            "arm": arm, "n_interventions": n_int, "mean_vmaf_given_up": round(mean, 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "frac_exceed_margin": round(frac_exceed, 3), "noninferior": noninferior,
        })
        verdict = "NON-INFERIOR" if noninferior else "not established"
        print(f"  {arm:26s} n_int={n_int:4d}  mean_loss={mean:6.3f}  "
              f"95% CI=[{lo:.3f},{hi:.3f}]  exceed>{args.margin:.0f}={frac_exceed:.3f}  -> {verdict}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else path.with_name("vmaf_equivalence_conditional_v14.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Isolation A/B: per (video, episode, chunk) executed-VMAF difference between
    # the VMAF-aware and highest-feasible-index arms. Should be exactly 0.
    for tol in ["0.8", "1.0"]:
        va = f"vmaf_aware_tol{tol}_bud08"
        hf = f"highest_feasible_tol{tol}"
        if va in available and hf in available:
            keys = ["Video", "Episode", "Chunk"]
            a = df[df.Method == va].set_index(keys)["Exec_VMAF_ladder"]
            b = df[df.Method == hf].set_index(keys)["Exec_VMAF_ladder"]
            joined = a.subtract(b, fill_value=np.nan).dropna()
            max_abs = float(np.abs(joined.to_numpy()).max()) if len(joined) else float("nan")
            print(f"[Isolation tol={tol}] max |exec-VMAF difference| between "
                  f"VMAF-aware and highest-index = {max_abs} over {len(joined)} chunks")


if __name__ == "__main__":
    main()
