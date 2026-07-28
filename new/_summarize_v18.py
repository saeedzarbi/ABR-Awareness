import json
from pathlib import Path

base = Path("results/v18_certified")
order = ["greedy_5g", "greedy_5g_improved", "bba_5g", "bba_5g_improved",
         "greedy_bb", "pensieve_5g", "proposed_5g", "proposed_5g_improved",
         "proposed_5g_regime", "proposed_cps_5g"]


def load(name):
    p = base / name / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def arm_row(a):
    g = lambda k: a[k]["mean"] if isinstance(a.get(k), dict) else a.get(k, float("nan"))
    return (g("rebuffer_total"), g("rebuffer_p95_chunk"), g("stallfree"),
            g("vmaf_mean"), g("bitrate_mean_kbps"), g("buffer_mean"),
            g("interv_rate"), a.get("conformal_coverage", float("nan")))


for name in order:
    s = load(name)
    if s is None:
        print(f"\n##### {name}: MISSING")
        continue
    print(f"\n##### {name}  (eps={s.get('epsilon')} alpha={s.get('alpha')} tag={s.get('tag')})")
    print(f"{'arm':>10}{'reb':>10}{'reb_p95':>9}{'stallfr':>8}{'vmaf':>8}{'bitrate':>9}{'buf':>7}{'interv':>8}{'cover':>7}")
    for arm in ["raw", "safety", "certified"]:
        if arm in s["arms"]:
            r = arm_row(s["arms"][arm])
            print(f"{arm:>10}{r[0]:>10.2f}{r[1]:>9.2f}{r[2]:>8.3f}{r[3]:>8.2f}{r[4]:>9.0f}{r[5]:>7.2f}{r[6]:>8.3f}{r[7]:>7.3f}")
    for cname, c in s.get("comparisons", {}).items():
        print(f"   [{cname}] BW {c['bandwidth_reduction_pct']:+.1f}% (p={c['bandwidth_wilcoxon_p']:.1e})  "
              f"reb {c['rebuffer_change_pct']:+.1f}% (p={c['rebuffer_wilcoxon_p']:.1e})  "
              f"VMAFd {c['vmaf_mean_diff']:+.3f} TOST_p={c['vmaf_tost_p_equiv']:.1e} within_eps={c['vmaf_within_epsilon']}")


# ---- Co-design head-to-head: certified arm of regime baseline vs co-trained ----
print("\n\n========== CO-DESIGN (item 5): certified arm, 5G test ==========")
b = load("proposed_5g_regime")
c = load("proposed_cps_5g")
if b and c:
    for lbl, s in [("proposed_5g_regime (shield@eval)", b), ("proposed_cps_5g (co-trained)", c)]:
        a = s["arms"]["certified"]
        r = arm_row(a)
        print(f"  {lbl:36s} reb={r[0]:7.2f}  vmaf={r[3]:6.2f}  bitrate={r[4]:5.0f}  "
              f"stallfree={r[2]:.3f}  cover={r[7]:.3f}")
