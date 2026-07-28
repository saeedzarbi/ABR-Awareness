"""Formal QoE-weighted comparison of the co-design (item 5), paired per episode.

Compares the CERTIFIED arm of:
  A = proposed_5g_regime  (policy trained WITHOUT the shield; shield only at eval)
  B = proposed_cps_5g     (policy CO-TRAINED with the shield in the loop)

Both were evaluated on the same 5G test traces with paired episode seeds, so the
episode index pairs identical (video, trace). We define a fidelity-vs-stall QoE in
VMAF points:

    QoE(w) = mean_VMAF  -  w * rebuffer_seconds

and sweep the stall weight w (VMAF points lost per second of rebuffering) to find
the crossover w* below which the bolder co-trained policy wins and above which the
conservative eval-only policy wins. Smoothness is omitted (not logged per episode);
both policies use the same action granularity so this is a minor, symmetric term.
"""
import csv
from pathlib import Path

import numpy as np

base = Path("results/v18_certified")


def load_arm(name, arm="certified"):
    rows = {}
    with open(base / name / "episodes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] != arm:
                continue
            rows[int(r["episode"])] = {
                "vmaf": float(r["vmaf_mean"]),
                "reb": float(r["rebuffer_total"]),
                "bitrate": float(r["bitrate_mean_kbps"]),
                "stallfree": float(r["stallfree"]),
            }
    return rows


A = load_arm("proposed_5g_regime")   # eval-only shield (conservative baseline)
B = load_arm("proposed_cps_5g")      # co-trained (bolder)
eps = sorted(set(A) & set(B))
assert eps, "no paired episodes"

vA = np.array([A[e]["vmaf"] for e in eps]);  rA = np.array([A[e]["reb"] for e in eps])
vB = np.array([B[e]["vmaf"] for e in eps]);  rB = np.array([B[e]["reb"] for e in eps])
print(f"paired episodes: {len(eps)}")
print(f"  A eval-only : VMAF {vA.mean():6.2f}  reb {rA.mean():5.2f}s  "
      f"br {np.mean([A[e]['bitrate'] for e in eps]):5.0f}  stallfree {np.mean([A[e]['stallfree'] for e in eps]):.3f}")
print(f"  B co-trained: VMAF {vB.mean():6.2f}  reb {rB.mean():5.2f}s  "
      f"br {np.mean([B[e]['bitrate'] for e in eps]):5.0f}  stallfree {np.mean([B[e]['stallfree'] for e in eps]):.3f}")
print(f"  deltas (B-A): VMAF {vB.mean()-vA.mean():+.2f}   reb {rB.mean()-rA.mean():+.2f}s")


def wilcoxon_p(d):
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        rng = np.random.default_rng(0)
        obs = abs(d.mean())
        perm = np.abs((rng.choice([-1.0, 1.0], size=(20000, d.size)) * d).mean(1))
        return float(max((perm >= obs - 1e-12).mean(), 1 / 20000))


# ---- crossover: solve E[QoE_B - QoE_A] = 0 for w ----
dV = vB.mean() - vA.mean()          # fidelity gain of co-design
dR = rB.mean() - rA.mean()          # extra stall of co-design (s)
w_star = dV / dR if dR != 0 else float("inf")
print(f"\ncrossover w* = dVMAF/dReb = {dV:+.3f} / {dR:+.3f} = {w_star:.2f} VMAF-points per second")
print("  -> co-design (B) has higher mean QoE when the stall weight w < w*.")

print(f"\n{'w (VMAF/s)':>11}{'QoE_A':>9}{'QoE_B':>9}{'B-A':>9}{'wilcoxon_p':>12}  winner")
for w in [0, 1, 2, 5, round(w_star, 2), 10, 20, 43, 50, 100]:
    qA = vA - w * rA
    qB = vB - w * rB
    d = qB - qA
    p = wilcoxon_p(d)
    win = "co-design(B)" if d.mean() > 0 else "eval-only(A)"
    star = "  <-- crossover" if abs(w - round(w_star, 2)) < 1e-9 else ""
    print(f"{w:>11.2f}{qA.mean():>9.2f}{qB.mean():>9.2f}{d.mean():>9.2f}{p:>12.1e}  {win}{star}")

print("\nReference stall weights (VMAF points lost per second of stall):")
print("  w~1-3   : fidelity-favoring QoE (typical for 'quality-first' streaming UX)")
print("  w~5-10  : balanced")
print("  w>=~20  : stall-averse (rebuffering strongly penalized)")
