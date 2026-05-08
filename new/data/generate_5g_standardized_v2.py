"""
Generate synthetic 5G/mmWave-like traces (standardized JSON) with tunable stress.

This is a *separate* generator from generate_5g_standardized.py (no changes to the old one).

Output format (compatible with the emulator):
  {"throughput_kbps": [ ... per-second samples ... ]}

Default output directory:
  new/data/standardized/test_traces_5g/

Profiles
--------
--profile realistic:
  Mostly high throughput with occasional NLoS dips and rare outages.

--profile stress:
  More frequent/longer NLoS + outages (intended to create stall events).

Usage
-----
  cd new
  python data/generate_5g_standardized_v2.py --profile stress --num 50 --length 300 --seed 123
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StateCfg:
    name: str
    kbps_lo: float
    kbps_hi: float
    mean_dur_s: float


def _sample_duration(mean_dur_s: float, rng: np.random.Generator) -> int:
    # Exponential with minimum 1s.
    d = float(rng.exponential(max(0.1, mean_dur_s)))
    return max(1, int(round(d)))


def _gen_trace_kbps(
    length_s: int,
    rng: np.random.Generator,
    p_outage: float,
    los: StateCfg,
    nlos: StateCfg,
    outage: StateCfg,
) -> list[float]:
    """
    3-state generator:
      - LoS (high throughput)
      - NLoS (moderate/low throughput)
      - Outage (very low throughput)

    Transition logic:
      - LoS -> NLoS always
      - NLoS -> Outage with probability p_outage else LoS
      - Outage -> LoS always
    """
    t = 0
    state = "los"
    out: list[float] = []

    while t < length_s:
        if state == "los":
            cfg = los
            next_state = "nlos"
        elif state == "nlos":
            cfg = nlos
            next_state = "outage" if float(rng.random()) < float(p_outage) else "los"
        else:
            cfg = outage
            next_state = "los"

        bw_kbps = float(rng.uniform(cfg.kbps_lo, cfg.kbps_hi))
        dur = _sample_duration(cfg.mean_dur_s, rng)

        take = min(dur, length_s - t)
        out.extend([bw_kbps] * take)
        t += take
        state = next_state

    return out[:length_s]


def _profile_cfg(profile: str) -> tuple[float, StateCfg, StateCfg, StateCfg]:
    profile = profile.strip().lower()

    if profile == "stress":
        # Designed to create noticeable stalls: frequent/long NLoS and non-trivial outages.
        p_outage = 0.45
        los = StateCfg("los", 30_000.0, 80_000.0, mean_dur_s=5.0)       # 30–80 Mbps, shorter
        nlos = StateCfg("nlos", 300.0, 2_500.0, mean_dur_s=6.0)         # 0.3–2.5 Mbps, longer
        outage = StateCfg("outage", 0.0, 250.0, mean_dur_s=3.0)         # 0–0.25 Mbps
        return p_outage, los, nlos, outage

    # realistic (default)
    p_outage = 0.15
    los = StateCfg("los", 50_000.0, 150_000.0, mean_dur_s=8.0)          # 50–150 Mbps
    nlos = StateCfg("nlos", 2_000.0, 10_000.0, mean_dur_s=2.5)          # 2–10 Mbps
    outage = StateCfg("outage", 200.0, 1_000.0, mean_dur_s=1.2)         # rare but not zero
    return p_outage, los, nlos, outage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="realistic", choices=["realistic", "stress"])
    parser.add_argument("--num", type=int, default=20, help="Number of traces to generate.")
    parser.add_argument("--length", type=int, default=300, help="Trace length in seconds (samples).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "standardized" / "test_traces_5g"),
        help="Output directory (relative to new/ when run from new/).",
    )
    parser.add_argument("--prefix", type=str, default="", help="Optional filename prefix override.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    p_outage, los, nlos, outage = _profile_cfg(args.profile)
    prefix = args.prefix.strip() or f"synthetic_5g_{args.profile}"

    for i in range(int(args.num)):
        thr = _gen_trace_kbps(
            int(args.length),
            rng,
            p_outage=p_outage,
            los=los,
            nlos=nlos,
            outage=outage,
        )
        payload = {"throughput_kbps": thr}
        p = out_dir / f"{prefix}_{i:04d}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Generated {args.num} traces ({args.profile}) -> {out_dir}")
    print(f"  p_outage={p_outage}")
    print(f"  los    : {los.kbps_lo:.0f}-{los.kbps_hi:.0f} kbps, mean_dur={los.mean_dur_s}s")
    print(f"  nlos   : {nlos.kbps_lo:.0f}-{nlos.kbps_hi:.0f} kbps, mean_dur={nlos.mean_dur_s}s")
    print(f"  outage : {outage.kbps_lo:.0f}-{outage.kbps_hi:.0f} kbps, mean_dur={outage.mean_dur_s}s")


if __name__ == "__main__":
    main()

