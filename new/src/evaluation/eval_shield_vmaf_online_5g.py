"""
Standalone 5G/mmWave OOD evaluation (does NOT modify the existing v121/v12 pipelines).

Goal
----
Evaluate a trained ABR policy under a synthetic 5G/mmWave-like trace set and
produce the same type of outputs as eval_shield_vmaf_online_v121.py:
  - per-episode CSV
  - summary CSV with bootstrap CIs

Workflow
--------
1) Generate synthetic traces into the standardized format:
     cd new
     python data/generate_5g_standardized.py --num 50 --length 300 --seed 123
   This writes:
     data/standardized/test_traces_5g/*.json

2) Run this evaluator:
     cd new
     python src/evaluation/eval_shield_vmaf_online_5g.py \
       --policy proposed_shielded_qoe_v12 \
       --episodes 20 \
       --trace_dir data/standardized/test_traces_5g \
       --out results/v5g_shielded_qoe/online_episodes.csv

Notes
-----
- This script is intentionally separate so the original scripts remain unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.environment.abr_multi_env_v12 import ABREnv
from src.training.safety_shield_v12 import ShieldConfig, safe_adjust_action

PATHS = get_paths()
EVAL_REBUF_PENALTY = 4.3
EVAL_SMOOTH_PENALTY = 1.0


def _resolve_model_path(folder_name: str) -> Path | None:
    base = PATHS["models"] / "master_v12" / folder_name
    best = base / "best_model" / "best_model"
    if best.with_suffix(".zip").exists():
        return best
    final = base / "final_model"
    if final.with_suffix(".zip").exists():
        return final
    return None


def _make_env(video: str, trace_dir: str) -> ABREnv:
    return ABREnv(
        video_names=[video],
        trace_dir=str(trace_dir),
        vmaf_dir=str(PATHS["vmaf_scores"]),
        siti_dir=str(PATHS["content_features"]),
        max_chunks=48,
        random_seed=12345,
        use_future=True,
        use_lyapunov=True,
    )


def _bootstrap_ci(x: np.ndarray, n_boot: int = 4000, ci: float = 95.0) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    n = len(x)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def run_episode(model: PPO, env: ABREnv, shield_cfg: ShieldConfig, seed: int) -> dict:
    obs, info = env.reset(seed=int(seed))
    qoe = 0.0
    total_vmaf = 0.0
    total_rebuf_s = 0.0
    last_vmaf = float(getattr(env, "last_vmaf", 35.0))
    last_action = -1
    switches = 0
    interventions = 0
    chunks = 0
    done = False
    k = 0

    while not done and env.chunk_idx < env.max_chunks:
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = int(raw_action)
        raw_action = max(0, min(raw_action, len(env.BITRATE_LEVELS) - 1))

        safe_action, intervened = safe_adjust_action(env, raw_action, shield_cfg)
        interventions += int(intervened)

        obs, _, done, _, info = env.step(int(safe_action))

        cur_vmaf = float(info.get("vmaf", last_vmaf))
        rebuf = float(info.get("rebuffer", 0.0))
        smooth = 0.0 if k == 0 else abs(cur_vmaf - last_vmaf)
        step_qoe = cur_vmaf - EVAL_REBUF_PENALTY * rebuf - EVAL_SMOOTH_PENALTY * smooth

        qoe += step_qoe
        total_vmaf += cur_vmaf
        total_rebuf_s += rebuf
        if last_action >= 0 and int(safe_action) != last_action:
            switches += 1
        last_action = int(safe_action)
        last_vmaf = cur_vmaf
        chunks += 1
        k += 1

    duration = chunks * env.CHUNK_DURATION
    rebuf_pct = (total_rebuf_s / duration * 100.0) if duration > 0 else 0.0
    avg_vmaf = total_vmaf / max(1, chunks)
    intervention_rate = interventions / max(1, chunks)

    return {
        "QoE": float(qoe),
        "VMAF": float(avg_vmaf),
        "Rebuffer": float(rebuf_pct),
        "Switch": int(switches),
        "Intervention_Rate": float(intervention_rate),
        "Chunks": int(chunks),
    }


def build_shield_grid() -> dict[str, ShieldConfig]:
    # Keep a compact, interpretable grid for OOD 5G testing.
    return {
        "shield_off": ShieldConfig(level="off"),
        "shield_legacy": ShieldConfig(level="light", vmaf_aware=False),
        # Headline v123 points
        "vmaf_aware_tol1.0_bud08": ShieldConfig(level="light", vmaf_aware=True, soft_tolerance=1.0, vmaf_loss_budget=8.0),
        "vmaf_aware_tol0.8_bud08": ShieldConfig(level="light", vmaf_aware=True, soft_tolerance=0.8, vmaf_loss_budget=8.0),
        # Optional threshold knob for stress
        "thresh_cat3.0_vmafFB": ShieldConfig(level="light", catastrophic_ratio=3.0, vmaf_aware=True, soft_tolerance=1.0, vmaf_loss_budget=8.0),
        "thresh_cat4.0_vmafFB": ShieldConfig(level="light", catastrophic_ratio=4.0, vmaf_aware=True, soft_tolerance=1.0, vmaf_loss_budget=8.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="proposed_shielded_qoe_v12",
                        help="Folder name under results/models/master_v12/")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--videos", type=str,
                        default="bigbuckbunny,crowd_run,tearsofsteel_short,sintel")
    parser.add_argument("--trace_dir", type=str, required=True,
                        help="5G trace directory (standardized JSON traces). Example: data/standardized/test_traces_5g")
    parser.add_argument("--out", type=str,
                        default=str(PATHS["results"] / "v5g_eval" / "online_episodes.csv"),
                        help="Per-episode CSV path (saved under results/ by default).")
    args = parser.parse_args()

    model_path = _resolve_model_path(args.policy)
    if model_path is None:
        print(f"[ERROR] No trained model found for policy '{args.policy}'.")
        print(f"        Looked under: {PATHS['models'] / 'master_v12' / args.policy}")
        raise SystemExit(1)

    trace_dir = args.trace_dir.strip()
    if not Path(trace_dir).exists():
        raise SystemExit(f"[ERROR] trace_dir not found: {trace_dir}")

    model = PPO.load(str(model_path))
    cfgs = build_shield_grid()
    test_videos = [v.strip() for v in args.videos.split(",") if v.strip()]

    print(f"Policy         : {args.policy}")
    print(f"Trace dir      : {trace_dir}")
    print(f"Shield configs : {len(cfgs)}")
    print(f"Videos         : {test_videos}")
    print(f"Episodes/video : {args.episodes}")
    print()

    rows: list[dict] = []
    for cfg_name, cfg in cfgs.items():
        for video in test_videos:
            env = _make_env(video, trace_dir=trace_dir)
            for ep in range(args.episodes):
                m = run_episode(model, env, cfg, seed=ep)
                m.update({"Method": cfg_name, "Video": video, "Episode": ep})
                rows.append(m)

        sub = [r for r in rows if r["Method"] == cfg_name]
        if sub:
            mq = float(np.mean([r["QoE"] for r in sub]))
            mv = float(np.mean([r["VMAF"] for r in sub]))
            mr = float(np.mean([r["Rebuffer"] for r in sub]))
            mi = float(np.mean([r["Intervention_Rate"] for r in sub])) * 100.0
            print(
                f"  {cfg_name:26s} n={len(sub):3d}  "
                f"QoE={mq:7.1f}  VMAF={mv:5.2f}  Rebuf={mr:6.2f}%  Interv={mi:5.2f}%"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-episode results -> {out_path}")

    summary_rows = []
    for cfg_name in out_df["Method"].unique():
        sub = out_df[out_df["Method"] == cfg_name]
        qlo, qhi = _bootstrap_ci(sub["QoE"].values.astype(float))
        rlo, rhi = _bootstrap_ci(sub["Rebuffer"].values.astype(float))
        summary_rows.append({
            "Method": cfg_name,
            "n": int(len(sub)),
            "QoE_mean": round(float(sub["QoE"].mean()), 2),
            "QoE_ci_lo": round(float(qlo), 2),
            "QoE_ci_hi": round(float(qhi), 2),
            "VMAF_mean": round(float(sub["VMAF"].mean()), 2),
            "Rebuf_pct": round(float(sub["Rebuffer"].mean()), 2),
            "Rebuf_ci_lo": round(float(rlo), 2),
            "Rebuf_ci_hi": round(float(rhi), 2),
            "Switch_mean": round(float(sub["Switch"].mean()), 2),
            "Interv_rate_pct": round(float(sub["Intervention_Rate"].mean()) * 100.0, 2),
        })

    summary = pd.DataFrame(summary_rows).sort_values("QoE_mean", ascending=False)
    summary_path = out_path.with_name(out_path.stem.replace("episodes", "summary") + ".csv")
    if summary_path == out_path:
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary             -> {summary_path}")


if __name__ == "__main__":
    main()

