"""
Split mixed 5G trace folder into separate profile folders.

Why
---
If `data/standardized/test_traces_5g/` contains both:
  - synthetic_5g_mmwave_*.json
  - synthetic_5g_stress_*.json
then evaluation results can silently mix profiles.

This script creates:
  data/standardized/test_traces_5g_mmwave/
  data/standardized/test_traces_5g_stress/

Usage
-----
  cd new
  python data/split_5g_traces.py

Options
-------
  --src  <dir>   source dir (default: data/standardized/test_traces_5g)
  --mode copy|move   default: copy
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        default=str(Path("data") / "standardized" / "test_traces_5g"),
        help="Source directory containing mixed 5G traces.",
    )
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"])
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[ERROR] source dir not found: {src}")

    dst_mm = src.parent / "test_traces_5g_mmwave"
    dst_st = src.parent / "test_traces_5g_stress"
    dst_mm.mkdir(parents=True, exist_ok=True)
    dst_st.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"[ERROR] no .json traces found in: {src}")

    n_mm = 0
    n_st = 0
    n_other = 0

    for f in files:
        name = f.name
        if name.startswith("synthetic_5g_mmwave_"):
            dst = dst_mm / name
            n_mm += 1
        elif name.startswith("synthetic_5g_stress_") or name.startswith("synthetic_5g_stress"):
            dst = dst_st / name
            n_st += 1
        else:
            n_other += 1
            continue

        if args.mode == "move":
            shutil.move(str(f), str(dst))
        else:
            shutil.copy2(str(f), str(dst))

    print(f"Source: {src}")
    print(f"Mode  : {args.mode}")
    print(f"mmWave: {n_mm} -> {dst_mm}")
    print(f"stress: {n_st} -> {dst_st}")
    if n_other:
        print(f"[WARN] Unrecognized prefix: {n_other} files left untouched in source.")


if __name__ == "__main__":
    main()

