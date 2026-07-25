"""
Synthetic 5G/mmWave stress traces (V14, reviewer-response) + real-trace loader.

Why this replaces generate_5g_standardized.py
---------------------------------------------
The review (F4) flagged that the previous synthetic generator was (a) piecewise-
constant within each LoS/NLoS dwell (no per-second variation), (b) had no deep
outage state, and (c) shipped no validation of its statistics against measured
5G data. It also urged using public *measured* traces where possible.

This script does two things:

1. ``synth`` (default): generate a more realistic three-state trace
   (LoS / NLoS / outage) with:
     - log-normal within-state throughput (heavy right tail, as measured),
     - AR(1) temporal correlation within a state (no piecewise-constant steps),
     - an explicit deep-outage state (near-zero throughput) modelling blockage,
     - exponential dwell times with literature-scale means.
   It writes a ``trace_stats.json`` with marginal mean/CV, outage fraction, and
   dwell-time summaries so the paper can report calibration instead of asserting
   "5G-like".

2. ``real``: standardize a directory of MEASURED 5G traces (e.g. Lumos5G,
   Ghent 4G/5G, or Raca et al.) into the project's
   ``{"throughput_kbps": [...]}`` per-second format. This lets the server run the
   OOD evaluation on real data with no code change:
     python data/generate_5g_standardized_v14.py real \
         --src /path/to/lumos5g_csv --col throughput_mbps --unit mbps \
         --out data/standardized/test_traces_5g_real

Default calibration targets (order-of-magnitude, mmWave UE):
  LoS   : median ~120 Mbps, CV ~0.35, mean dwell ~6 s
  NLoS  : median ~12 Mbps,  CV ~0.6,  mean dwell ~3 s
  outage: ~0.2 Mbps,        mean dwell ~1 s, entered ~from NLoS
These are deliberately conservative and documented as synthetic; they are a
stress harness, not a claim about a specific deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_KBPS = 10.0  # emulator throughput floor


def _lognormal_kbps(median_mbps: float, cv: float, rng: np.random.Generator) -> float:
    """Draw one throughput sample (kbps) from a log-normal with the given median
    (Mbps) and coefficient of variation."""
    sigma = float(np.sqrt(np.log(1.0 + cv * cv)))
    mu = float(np.log(median_mbps))
    val_mbps = float(rng.lognormal(mean=mu, sigma=sigma))
    return max(MIN_KBPS, val_mbps * 1000.0)


def _gen_trace_kbps(length_s: int, rng: np.random.Generator, cfg: dict) -> tuple[list[float], dict]:
    """Three-state (LoS=0, NLoS=1, outage=2) generator with AR(1) within state."""
    state = 0
    out: list[float] = []
    dwell_records = {0: [], 1: [], 2: []}
    ar = float(cfg["ar_rho"])
    prev = _lognormal_kbps(cfg["los_median_mbps"], cfg["los_cv"], rng)
    t = 0
    while t < length_s:
        if state == 0:
            median, cv, mean_dwell = cfg["los_median_mbps"], cfg["los_cv"], cfg["los_dwell_s"]
        elif state == 1:
            median, cv, mean_dwell = cfg["nlos_median_mbps"], cfg["nlos_cv"], cfg["nlos_dwell_s"]
        else:
            median, cv, mean_dwell = cfg["outage_median_mbps"], cfg["outage_cv"], cfg["outage_dwell_s"]

        dwell = max(1, int(round(rng.exponential(mean_dwell))))
        dwell_records[state].append(dwell)
        for _ in range(dwell):
            if t >= length_s:
                break
            target = _lognormal_kbps(median, cv, rng)
            # AR(1) smoothing for realistic short-term correlation.
            sample = ar * prev + (1.0 - ar) * target
            sample = max(MIN_KBPS, sample)
            out.append(float(sample))
            prev = sample
            t += 1

        # State transition.
        if state == 0:
            state = 1 if rng.random() < cfg["p_los_to_nlos"] else 0
        elif state == 1:
            r = rng.random()
            if r < cfg["p_nlos_to_outage"]:
                state = 2
            elif r < cfg["p_nlos_to_outage"] + cfg["p_nlos_to_los"]:
                state = 0
            else:
                state = 1
        else:
            state = 1  # outage recovers to NLoS

    arr = np.asarray(out[:length_s], dtype=float)
    stats = {
        "mean_kbps": float(arr.mean()),
        "median_kbps": float(np.median(arr)),
        "cv": float(arr.std() / (arr.mean() + 1e-9)),
        "outage_frac": float((arr <= cfg["outage_median_mbps"] * 1000.0 * 2).mean()),
        "min_kbps": float(arr.min()),
        "p05_kbps": float(np.percentile(arr, 5)),
    }
    return out[:length_s], stats


DEFAULT_CFG = {
    "los_median_mbps": 120.0, "los_cv": 0.35, "los_dwell_s": 6.0,
    "nlos_median_mbps": 12.0, "nlos_cv": 0.60, "nlos_dwell_s": 3.0,
    "outage_median_mbps": 0.2, "outage_cv": 0.5, "outage_dwell_s": 1.0,
    "p_los_to_nlos": 0.35, "p_nlos_to_los": 0.45, "p_nlos_to_outage": 0.12,
    "ar_rho": 0.6,
}


def cmd_synth(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    all_stats = []
    for i in range(int(args.num)):
        thr, stats = _gen_trace_kbps(int(args.length), rng, DEFAULT_CFG)
        (out_dir / f"{args.prefix}_{i:04d}.json").write_text(
            json.dumps({"throughput_kbps": thr}), encoding="utf-8"
        )
        all_stats.append(stats)

    def _agg(key):
        vals = np.array([s[key] for s in all_stats], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    summary = {
        "generator": "v14_three_state_ar1",
        "config": DEFAULT_CFG,
        "n_traces": int(args.num), "length_s": int(args.length),
        "validation": {k: _agg(k) for k in
                       ["mean_kbps", "median_kbps", "cv", "outage_frac", "min_kbps", "p05_kbps"]},
    }
    (out_dir / "trace_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated {args.num} synthetic 5G traces -> {out_dir}")
    print(f"Validation summary -> {out_dir / 'trace_stats.json'}")
    print(json.dumps(summary["validation"], indent=2))


def _read_real_series(path: Path, col: str | None, unit: str) -> list[float] | None:
    """Read one measured trace file (csv/json/txt) into kbps-per-second."""
    scale = {"kbps": 1.0, "mbps": 1000.0, "bps": 1e-3}[unit]
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            series = data.get("throughput_kbps") or (data.get(col) if col else None)
            if series is None:
                return None
            return [max(MIN_KBPS, float(v) * (1.0 if "kbps" in (col or "throughput_kbps") else scale)) for v in series]
        # csv / whitespace-delimited txt
        import csv as _csv
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            sniff = fh.readline()
            fh.seek(0)
            if "," in sniff and col:
                reader = _csv.DictReader(fh)
                for r in reader:
                    if col in r and r[col] not in ("", None):
                        rows.append(float(r[col]) * scale)
            else:
                for line in fh:
                    parts = line.split()
                    if parts:
                        try:
                            rows.append(float(parts[-1]) * scale)
                        except ValueError:
                            continue
        return [max(MIN_KBPS, v) for v in rows] if len(rows) > 5 else None
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to read {path.name}: {exc}")
        return None


def cmd_real(args):
    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in src.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".txt", ".log", ""}]
    n = 0
    for p in files:
        series = _read_real_series(p, args.col, args.unit)
        if series:
            (out_dir / f"{args.prefix}_{n:04d}.json").write_text(
                json.dumps({"throughput_kbps": series}), encoding="utf-8"
            )
            n += 1
    print(f"Standardized {n} measured traces from {src} -> {out_dir}")
    if n == 0:
        print("[WARN] no usable traces found. Check --col and --unit for your dataset.")


def main():
    parser = argparse.ArgumentParser(description="5G/mmWave trace tooling (V14).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synth", help="Generate synthetic three-state 5G traces.")
    ps.add_argument("--num", type=int, default=50)
    ps.add_argument("--length", type=int, default=300)
    ps.add_argument("--seed", type=int, default=123)
    ps.add_argument("--out", type=str, default=str(Path("data") / "standardized" / "test_traces_5g_stress_v14"))
    ps.add_argument("--prefix", type=str, default="synthetic_5g_v14")
    ps.set_defaults(func=cmd_synth)

    pr = sub.add_parser("real", help="Standardize measured 5G traces.")
    pr.add_argument("--src", type=str, required=True, help="Directory of measured trace files.")
    pr.add_argument("--col", type=str, default=None, help="Throughput column name (for CSV/JSON).")
    pr.add_argument("--unit", type=str, default="mbps", choices=["kbps", "mbps", "bps"])
    pr.add_argument("--out", type=str, default=str(Path("data") / "standardized" / "test_traces_5g_real"))
    pr.add_argument("--prefix", type=str, default="real_5g")
    pr.set_defaults(func=cmd_real)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
