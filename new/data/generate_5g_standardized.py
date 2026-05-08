"""
Generate synthetic 5G/mmWave-like traces in the project's *standardized* trace format.

Why this exists
--------------
The repository's emulator expects JSON traces with a single key:
  {"throughput_kbps": [ ... per-second samples ... ]}

We keep these traces in:
  new/data/standardized/test_traces_5g/

Usage
-----
  cd new
  python data/generate_5g_standardized.py --num 50 --length 300

This is intentionally separate from train/test traces used in the main paper
so we can run an out-of-distribution (OOD) evaluation on 5G-like dynamics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _gen_trace_kbps(length_s: int, rng: np.random.Generator) -> list[float]:
    """
    Two-state LoS/NLoS model:
      - LoS: 80–150 Mbps (80_000–150_000 kbps), mean duration ~8s
      - NLoS: 2–10 Mbps (2_000–10_000 kbps), mean duration ~2s
    """
    t = 0
    state = 0  # 0=LoS, 1=NLoS
    out: list[float] = []
    while t < length_s:
        if state == 0:
            bw_kbps = float(rng.uniform(80_000.0, 150_000.0))
            duration = float(rng.exponential(8.0))
            state = 1
        else:
            bw_kbps = float(rng.uniform(2_000.0, 10_000.0))
            duration = float(rng.exponential(2.0))
            state = 0

        dur_i = max(1, int(duration))
        take = min(dur_i, length_s - t)
        out.extend([bw_kbps] * take)
        t += take

    return out[:length_s]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=20, help="Number of traces to generate.")
    parser.add_argument("--length", type=int, default=300, help="Trace length in seconds (samples).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "standardized" / "test_traces_5g"),
        help="Output directory (relative to new/ when run from new/).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="synthetic_5g_mmwave",
        help="Filename prefix.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    for i in range(int(args.num)):
        thr = _gen_trace_kbps(int(args.length), rng)
        payload = {"throughput_kbps": thr}
        p = out_dir / f"{args.prefix}_{i:04d}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Generated {args.num} traces -> {out_dir}")


if __name__ == "__main__":
    main()

