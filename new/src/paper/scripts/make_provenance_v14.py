"""
Generate a train/test provenance record for the V14 pipeline (review P0.6 / F5.1).

Reviewers cannot tell from the manuscript whether the evaluation videos and
traces overlap with training. This script scans the on-disk data and writes a
machine-readable provenance file documenting:

  * which videos are seen (training) vs. unseen (held out),
  * whether any evaluation video is also a training video,
  * trace counts per split and whether train/test trace files overlap by name,
  * per-video VMAF-ladder monotonicity (the property that makes the perceptual
    budget inert; see safety_shield_v14.py).

Writes: new/results/PROVENANCE_v14.json

Usage:
  cd new
  python src/paper/scripts/make_provenance_v14.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from configs.paths import get_paths
from configs.videos import EVAL_VIDEOS, HELD_OUT_VIDEOS, TRAIN_VIDEOS

PATHS = get_paths()

TRAIN_VIDEOS = list(TRAIN_VIDEOS)
TEST_VIDEOS = list(EVAL_VIDEOS)
HELD_OUT = list(HELD_OUT_VIDEOS)


def _trace_names(d: Path) -> set[str]:
    return {p.name for p in d.glob("*.json")} if d.exists() else set()


def main():
    train_dir = PATHS["train_traces"]
    test_dir = PATHS["test_traces"]
    train_names = _trace_names(train_dir)
    test_names = _trace_names(test_dir)
    overlap = sorted(train_names & test_names)

    # VMAF ladder monotonicity per video.
    ladder = {}
    vmaf_csv = PATHS["vmaf_scores"] / "vmaf_summary.csv"
    if vmaf_csv.exists():
        df = pd.read_csv(vmaf_csv)
        for vid, g in df.sort_values("bitrate_kbps").groupby("video"):
            v = g["vmaf"].to_numpy(dtype=float)
            monotone = bool((v[1:] >= v[:-1]).all())
            ladder[str(vid)] = {
                "vmaf_by_rung": [float(x) for x in v],
                "monotone_nondecreasing": monotone,
            }

    prov = {
        "pipeline": "v14",
        "videos": {
            "train_videos": TRAIN_VIDEOS,
            "eval_videos": TEST_VIDEOS,
            "unseen_eval_videos": HELD_OUT,
            "eval_videos_also_in_training": [v for v in TEST_VIDEOS if v in TRAIN_VIDEOS],
            "note": (f"{len(TRAIN_VIDEOS)} training videos; {len(HELD_OUT)} held-out "
                     f"({', '.join(HELD_OUT)}). Eval reports seen/unseen/pooled separately."),
        },
        "traces": {
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "n_train": len(train_names),
            "n_test": len(test_names),
            "n_name_overlap": len(overlap),
            "overlap_examples": overlap[:10],
            "disjoint_by_name": len(overlap) == 0,
        },
        "vmaf_ladder": ladder,
        "vmaf_ladder_all_monotone": all(v["monotone_nondecreasing"] for v in ladder.values()) if ladder else None,
    }

    out = PATHS["results"] / "PROVENANCE_v14.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(prov, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
