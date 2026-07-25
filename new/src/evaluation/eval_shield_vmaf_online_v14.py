"""
Online shield-sweep evaluation (V14, reviewer-response).

This replaces eval_shield_vmaf_online_v121.py and directly answers the review's
two most damaging objections:

* F2 / P0.2  -- the perceptual (VMAF) ranking is never isolated. V14 adds a
  ``highest_feasible`` arm that uses the *identical* soft-safe feasible set as
  the VMAF-aware arm but ranks it by index only. On a monotone ladder the two
  arms MUST coincide; the paired difference between them is the clean, direct
  measurement of "how much does VMAF-awareness actually change".

* F3 / P0.3  -- everything here runs on ONE fixed policy checkpoint and ONE run,
  so every contrast (off / legacy / highest-feasible / VMAF-aware) is paired on
  the same policy, traces and seeds.

* P0.5       -- a per-chunk decision log (raw vs executed action and their VMAF)
  is written so the equivalence test can be computed on the *conditional*
  estimand (VMAF given up on intervened chunks), not the diluted session mean.

The same script covers broadband and OOD 5G by pointing ``--trace-dir`` at the
relevant standardized trace folder.

Usage:
  cd new
  # broadband
  python src/evaluation/eval_shield_vmaf_online_v14.py \
      --policy proposed_v14 --seed 0 --episodes 20 \
      --trace-dir data/standardized/test_traces \
      --out results/v14_shielded_qoe/online_episodes.csv
  # 5G stress
  python src/evaluation/eval_shield_vmaf_online_v14.py \
      --policy proposed_v14 --seed 0 --episodes 20 \
      --trace-dir data/standardized/test_traces_5g_stress \
      --out results/v14_5g_stress_shielded_qoe/online_episodes.csv
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.environment.abr_multi_env_v14 import ABREnv
from src.training.safety_shield_v14 import ShieldConfig, _vmaf_for_idx, safe_adjust_action

PATHS = get_paths()
EVAL_REBUF_PENALTY = 100.0
EVAL_SMOOTH_PENALTY = 1.0
MODEL_TAG = "master_v14"

SEEN_VIDEOS = {"bigbuckbunny", "crowd_run", "tearsofsteel_short"}


def _make_env(video: str, trace_dir: str) -> ABREnv:
    return ABREnv(
        video_names=[video],
        trace_dir=trace_dir,
        vmaf_dir=str(PATHS["vmaf_scores"]),
        siti_dir=str(PATHS["content_features"]),
        max_chunks=48,
        random_seed=12345,
        use_future=True,
        use_lyapunov=True,
    )


def _resolve_model_path(folder_name: str, seed: int) -> Path | None:
    base = PATHS["models"] / MODEL_TAG / folder_name / f"seed_{seed}"
    best = base / "best_model" / "best_model"
    if best.with_suffix(".zip").exists():
        return best
    final = base / "final_model"
    if final.with_suffix(".zip").exists():
        return final
    return None


def run_episode(model, env, shield_cfg, seed, cfg_name, video, ep, chunk_sink):
    obs, info = env.reset(seed=int(seed))
    qoe = 0.0
    total_vmaf = 0.0
    total_rebuf_s = 0.0
    last_vmaf = float(getattr(env, "last_vmaf", 35.0))
    last_action = -1
    switches = 0
    interventions = 0
    chunks = 0
    max_stall = 0.0
    done = False
    k = 0

    while not done and env.chunk_idx < env.max_chunks:
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = int(raw_action)
        raw_action = max(0, min(raw_action, len(env.BITRATE_LEVELS) - 1))

        # VMAF of the raw proposal on this chunk's ladder (before projection).
        raw_vmaf = _vmaf_for_idx(env, raw_action)

        safe_action, intervened = safe_adjust_action(env, raw_action, shield_cfg)
        interventions += int(intervened)
        exec_vmaf_ladder = _vmaf_for_idx(env, int(safe_action))

        obs, _, done, _, info = env.step(int(safe_action))

        cur_vmaf = float(info.get("vmaf", last_vmaf))
        rebuf = float(info.get("rebuffer", 0.0))
        max_stall = max(max_stall, rebuf)
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

        chunk_sink.append({
            "Method": cfg_name, "Video": video, "Episode": ep, "Chunk": k,
            "Raw_Action": raw_action, "Exec_Action": int(safe_action),
            "Raw_VMAF": round(raw_vmaf, 3), "Exec_VMAF_ladder": round(exec_vmaf_ladder, 3),
            "VMAF_given_up": round(max(0.0, raw_vmaf - exec_vmaf_ladder), 3),
            "Intervened": int(intervened), "Rebuffer_s": round(rebuf, 3),
        })
        k += 1

    duration = chunks * env.CHUNK_DURATION
    return {
        "QoE": float(qoe),
        "VMAF": float(total_vmaf / max(1, chunks)),
        "Rebuffer": float(total_rebuf_s / duration * 100.0) if duration > 0 else 0.0,
        "Switch": int(switches),
        "Intervention_Rate": float(interventions / max(1, chunks)),
        "Max_Stall_s": round(max_stall, 3),
        "Any_Stall": int(total_rebuf_s > 1e-6),
        "Chunks": int(chunks),
        "Seen": int(video in SEEN_VIDEOS),
    }


def build_shield_grid() -> dict[str, ShieldConfig]:
    cfgs: dict[str, ShieldConfig] = {}
    cfgs["shield_off"] = ShieldConfig(level="off")
    cfgs["shield_legacy"] = ShieldConfig(level="light", vmaf_aware=False)

    # Core A/B that isolates VMAF-awareness: same soft-safe set, two rankings.
    for tol in [0.8, 1.0, 1.2, 1.5]:
        cfgs[f"vmaf_aware_tol{tol:.1f}_bud08"] = ShieldConfig(
            level="light", vmaf_aware=True, selection="vmaf",
            soft_tolerance=float(tol), vmaf_loss_budget=8.0,
        )
        cfgs[f"highest_feasible_tol{tol:.1f}"] = ShieldConfig(
            level="light", vmaf_aware=True, selection="index",
            soft_tolerance=float(tol),
        )

    # Budget-inertness demonstration (same tol, three tau values).
    for tol in [0.8, 1.0]:
        for bud in [4.0, 8.0, 12.0]:
            cfgs[f"vmaf_aware_tol{tol:.1f}_bud{int(bud):02d}"] = ShieldConfig(
                level="light", vmaf_aware=True, selection="vmaf",
                soft_tolerance=float(tol), vmaf_loss_budget=float(bud),
            )

    # Catastrophic-ratio sweep (legacy threshold behaviour).
    for cat in [2.0, 3.0, 4.0, 5.0]:
        cfgs[f"thresh_cat{cat:.1f}"] = ShieldConfig(level="light", catastrophic_ratio=float(cat))
    return cfgs


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([values[rng.integers(0, values.size, values.size)].mean() for _ in range(n_boot)])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def _cvar_low(values: np.ndarray, alpha: float = 0.1) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * values.size)))
    return float(values[:k].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="proposed_v14")
    parser.add_argument("--seed", type=int, default=0, help="Training-seed checkpoint to replay.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--videos", type=str,
                        default="bigbuckbunny,crowd_run,tearsofsteel_short,sintel")
    parser.add_argument("--trace-dir", type=str, default=str(PATHS["test_traces"]))
    parser.add_argument("--out", type=str,
                        default=str(PATHS["results"] / "v14_shielded_qoe" / "online_episodes.csv"))
    args = parser.parse_args()

    model_path = _resolve_model_path(args.policy, args.seed)
    if model_path is None:
        print(f"[ERROR] No trained model for policy '{args.policy}' seed {args.seed}.")
        print(f"        Looked under: {PATHS['models'] / MODEL_TAG / args.policy / f'seed_{args.seed}'}")
        sys.exit(1)

    print(f"Loading policy: {model_path}")
    print(f"Trace dir     : {args.trace_dir}")
    model = PPO.load(str(model_path))

    cfgs = build_shield_grid()
    test_videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    print(f"Shield configs : {len(cfgs)} | Videos: {test_videos} | Episodes/video: {args.episodes}\n")

    rows = []
    chunk_rows = []
    for cfg_name, cfg in cfgs.items():
        for video in test_videos:
            env = _make_env(video, args.trace_dir)
            for ep in range(args.episodes):
                m = run_episode(model, env, cfg, ep, cfg_name, video, ep, chunk_rows)
                m.update({"Method": cfg_name, "Video": video, "Episode": ep})
                rows.append(m)
        sub = [r for r in rows if r["Method"] == cfg_name]
        mq = float(np.mean([r["QoE"] for r in sub]))
        mr = float(np.mean([r["Rebuffer"] for r in sub]))
        mi = float(np.mean([r["Intervention_Rate"] for r in sub])) * 100.0
        print(f"  {cfg_name:28s} n={len(sub):3d}  QoE={mq:8.1f}  Rebuf={mr:5.2f}%  Interv={mi:5.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved per-episode results -> {out_path}")

    chunk_path = out_path.with_name("online_decisions.csv")
    pd.DataFrame(chunk_rows).to_csv(chunk_path, index=False)
    print(f"Saved per-chunk decisions -> {chunk_path}")

    out_df = pd.DataFrame(rows)
    summary_rows = []
    for cfg_name in out_df["Method"].unique():
        sub = out_df[out_df["Method"] == cfg_name]
        qoe = sub["QoE"].to_numpy(float)
        rb = sub["Rebuffer"].to_numpy(float)
        q_lo, q_hi = _bootstrap_ci(qoe)
        r_lo, r_hi = _bootstrap_ci(rb)
        summary_rows.append({
            "Method": cfg_name, "n": int(len(sub)),
            "QoE_mean": round(float(qoe.mean()), 2), "QoE_ci_lo": round(q_lo, 2), "QoE_ci_hi": round(q_hi, 2),
            "QoE_CVaR10": round(_cvar_low(qoe, 0.1), 2),
            "VMAF_mean": round(float(sub["VMAF"].mean()), 2),
            "Rebuf_pct": round(float(rb.mean()), 3), "Rebuf_ci_lo": round(r_lo, 3), "Rebuf_ci_hi": round(r_hi, 3),
            "Rebuf_p95": round(float(np.percentile(rb, 95)), 3), "Rebuf_p99": round(float(np.percentile(rb, 99)), 3),
            "StallFree_frac": round(float((sub["Any_Stall"] == 0).mean()), 3),
            "Switch_mean": round(float(sub["Switch"].mean()), 2),
            "Interv_rate_pct": round(float(sub["Intervention_Rate"].mean()) * 100.0, 2),
        })
    summary = pd.DataFrame(summary_rows).sort_values("QoE_mean", ascending=False)
    summary_path = out_path.with_name("online_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary             -> {summary_path}")
    print("\n=== V14 SHIELD SWEEP SUMMARY (sorted by mean QoE) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
