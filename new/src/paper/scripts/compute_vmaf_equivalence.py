#!/usr/bin/env python3
"""
Post-hoc TOST (two one-sided tests) equivalence analysis for the paired VMAF
differences between VMAF-aware and legacy shield projection.

Motivation: main.ltx explicitly states (Sec. 6.3 and 6.5.1) that the
"quality-preserving" claim is descriptive, not a formal non-inferiority/
equivalence result, "because the study did not prespecify a perceptual
margin or perform an equivalence test." This script performs that test.

Equivalence margin: the study did not prespecify one, so we adopt a
conservative post-hoc margin of +/-2.0 VMAF points (roughly one third of the
~6-point value commonly cited as a just-noticeable difference for VMAF).
This choice is reported as exploratory/post-hoc, not confirmatory.

Reads:
  new/results/v123_shielded_qoe/online_episodes.csv   (primary broadband suite)
  new/results/v5g_stress_shielded_qoe/online_episodes.csv (synthetic 5G/mmWave stress suite)

Writes:
  new/src/paper/tables/macros_equivalence.tex

Usage:
  cd new
  python src/paper/scripts/compute_vmaf_equivalence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EQUIV_MARGIN = 2.0  # VMAF points; post-hoc SESOI, see module docstring
ALPHA = 0.05


def _repo_new() -> Path:
    return Path(__file__).resolve().parents[3]


def _pair_diff(df: pd.DataFrame, method: str, baseline: str = "shield_legacy") -> np.ndarray:
    a = df[df.Method == method].sort_values(["Video", "Episode"]).reset_index(drop=True)
    b = df[df.Method == baseline].sort_values(["Video", "Episode"]).reset_index(drop=True)
    assert len(a) == len(b), f"length mismatch: {method}={len(a)} vs {baseline}={len(b)}"
    assert (a["Video"].to_numpy() == b["Video"].to_numpy()).all()
    assert (a["Episode"].to_numpy() == b["Episode"].to_numpy()).all()
    return (a["VMAF"].to_numpy(dtype=float) - b["VMAF"].to_numpy(dtype=float))


def tost_paired(diff: np.ndarray, low: float, high: float, alpha: float = ALPHA) -> dict:
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    df_ = n - 1

    # H0_lower: true mean <= low   vs  H1: true mean > low
    t_low = (mean - low) / se
    p_low = float(1.0 - stats.t.cdf(t_low, df_))
    # H0_upper: true mean >= high  vs  H1: true mean < high
    t_high = (mean - high) / se
    p_high = float(stats.t.cdf(t_high, df_))
    p_tost = max(p_low, p_high)

    tcrit = stats.t.ppf(1.0 - alpha, df_)
    ci_lo = mean - tcrit * se
    ci_hi = mean + tcrit * se
    equivalent = bool(p_tost < alpha)

    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "p_low": p_low, "p_high": p_high, "p_tost": p_tost,
        "ci90_lo": ci_lo, "ci90_hi": ci_hi, "equivalent": equivalent,
    }


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _fmt_p(p: float) -> str:
    """Human-readable p-value for console printing."""
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _fmt_p_latex(p: float) -> str:
    """LaTeX-math-safe p-value string, matching the style of the other
    macros in macros_v12.tex (e.g. ``5.0\\!\\times\\!10^{-3}``)."""
    if p < 1e-3:
        mantissa, exp = f"{p:.1e}".split("e")
        exp = int(exp)
        return f"{float(mantissa):.1f}\\!\\times\\!10^{{{exp}}}"
    return f"{p:.3f}"


def main() -> None:
    new_root = _repo_new()
    broadband_path = new_root / "results" / "v123_shielded_qoe" / "online_episodes.csv"
    fiveg_path = new_root / "results" / "v5g_stress_shielded_qoe" / "online_episodes.csv"

    df_bb = pd.read_csv(broadband_path)
    df_5g = pd.read_csv(fiveg_path)

    configs = [
        ("bb_08", df_bb, "vmaf_aware_tol0.8_bud08", "broadband, sigma_soft=0.8, tau=8"),
        ("bb_10", df_bb, "vmaf_aware_tol1.0_bud08", "broadband, sigma_soft=1.0, tau=8"),
        ("5g_08", df_5g, "vmaf_aware_tol0.8_bud08", "5G/mmWave stress, sigma_soft=0.8, tau=8"),
        ("5g_10", df_5g, "vmaf_aware_tol1.0_bud08", "5G/mmWave stress, sigma_soft=1.0, tau=8"),
    ]

    results = {}
    print(f"TOST equivalence margin: +/-{EQUIV_MARGIN} VMAF points (post-hoc SESOI), alpha={ALPHA}\n")
    for key, df, method, label in configs:
        diff = _pair_diff(df, method)
        res = tost_paired(diff, -EQUIV_MARGIN, EQUIV_MARGIN)
        results[key] = res
        verdict = "EQUIVALENT" if res["equivalent"] else "not established"
        print(f"[{key}] {label} (n={res['n']})")
        print(f"    mean diff = {_fmt(res['mean'])}  sd = {_fmt(res['sd'])}  "
              f"90% CI = [{_fmt(res['ci90_lo'])}, {_fmt(res['ci90_hi'])}]")
        print(f"    p_lower = {_fmt_p(res['p_low'])}  p_upper = {_fmt_p(res['p_high'])}  "
              f"p_TOST = {_fmt_p(res['p_tost'])}  -> {verdict}\n")

    out_dir = new_root / "src" / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "macros_equivalence.tex"

    def macro(name: str, value: str) -> str:
        return f"\\providecommand{{\\{name}}}{{{value}}}"

    lines = [
        "% macros_equivalence.tex --- auto-generated by scripts/compute_vmaf_equivalence.py",
        "% Post-hoc TOST equivalence test for paired VMAF differences (VMAF-aware vs. legacy shield).",
        "\\makeatletter",
        macro("VEqMargin", _fmt(EQUIV_MARGIN, 1)),
    ]

    label_map = {
        "bb_08": "Eight", "bb_10": "One",
        "5g_08": "FiveGEight", "5g_10": "FiveGOne",
    }
    for key, suffix in label_map.items():
        r = results[key]
        lines.append(macro(f"VEqD{suffix}", _fmt(r["mean"])))
        lines.append(macro(f"VEqCILo{suffix}", _fmt(r["ci90_lo"])))
        lines.append(macro(f"VEqCIHi{suffix}", _fmt(r["ci90_hi"])))
        lines.append(macro(f"VEqP{suffix}", _fmt_p_latex(r["p_tost"])))
    lines.append("\\makeatother")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
