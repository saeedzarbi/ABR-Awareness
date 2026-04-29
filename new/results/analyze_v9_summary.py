import csv
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).parent
FILES = {
    "policy": BASE / "detailed_stats_master_v9_v9_policy.csv",
    "light": BASE / "detailed_stats_master_v9_v9_light.csv",
    "safe": BASE / "detailed_stats_master_v9_v9_safe.csv",
}

VIDEOS = ["bigbuckbunny", "crowd_run", "tearsofsteel_short", "sintel"]
METRICS = ["QoE", "VMAF", "Rebuffer", "Switch"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def summarize(rows):
    overall = defaultdict(lambda: defaultdict(list))
    by_video = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        m = r["Method"]
        v = r["Video"]
        for k in METRICS:
            val = float(r[k])
            overall[m][k].append(val)
            by_video[m][v][k].append(val)
    return overall, by_video


def print_mode(mode: str, rows):
    overall, by_video = summarize(rows)
    methods = sorted(overall.keys(), key=lambda m: _mean(overall[m]["QoE"]), reverse=True)

    print("\n" + "=" * 100)
    print(f"V9 SUMMARY – MODE: {mode.upper()}")
    print("=" * 100)
    print(f"{'Method':<24} {'QoE':>12} {'VMAF':>9} {'Rebuf%':>9} {'Switch':>8}  {'Rank':>4}")
    print("-" * 100)
    for i, m in enumerate(methods, start=1):
        print(
            f"{m:<24} "
            f"{_mean(overall[m]['QoE']):>12.2f} "
            f"{_mean(overall[m]['VMAF']):>9.2f} "
            f"{_mean(overall[m]['Rebuffer']):>9.2f} "
            f"{_mean(overall[m]['Switch']):>8.1f}  "
            f"{i:>4}"
        )

    # Focus comparison: Proposed vs Proposed_Shielded
    for target in ["Proposed", "Proposed_Shielded"]:
        if target not in by_video:
            continue
        print(f"\n-- Per-video: {target} --")
        for vid in VIDEOS:
            if vid not in by_video[target]:
                continue
            print(
                f"  {vid:<18} "
                f"QoE={_mean(by_video[target][vid]['QoE']):.2f}  "
                f"VMAF={_mean(by_video[target][vid]['VMAF']):.2f}  "
                f"Rebuf%={_mean(by_video[target][vid]['Rebuffer']):.2f}  "
                f"Switch={_mean(by_video[target][vid]['Switch']):.1f}"
            )

    if "Proposed" in overall and "Proposed_Shielded" in overall:
        dq = _mean(overall["Proposed"]["QoE"]) - _mean(overall["Proposed_Shielded"]["QoE"])
        dr = _mean(overall["Proposed"]["Rebuffer"]) - _mean(overall["Proposed_Shielded"]["Rebuffer"])
        dv = _mean(overall["Proposed"]["VMAF"]) - _mean(overall["Proposed_Shielded"]["VMAF"])
        print("\n-- Proposed minus Proposed_Shielded --")
        print(f"  dQoE={dq:.2f}   dRebuf%={dr:.2f}   dVMAF={dv:.2f}")


def main():
    for mode, path in FILES.items():
        if not path.exists():
            continue
        rows = load_rows(path)
        print_mode(mode, rows)


if __name__ == "__main__":
    main()

