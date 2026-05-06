"""
Off-policy shield replay (V12.1) -- VMAF-Aware Shield evaluation.

Goal
----
Evaluate the new VMAF-Aware Soft Projection (V12.1) shield WITHOUT retraining,
by replaying raw policy actions from the existing decision log under different
shield configurations. The Proposed agent (trained without a shield) provides
the raw action sequence, ensuring the policy itself is held constant across
shield variants.

Method
------
For each (video, episode):
  1. Build a fresh environment matching the original evaluation setup.
  2. Reset with seed = episode index (matches evaluate_all_models_v12.py).
  3. Step chunk-by-chunk; at each chunk, read the raw policy action that was
     emitted by the Proposed (no-shield) agent in the original run.
  4. Apply the candidate shield to that raw action.
  5. Step the env with the projected action.
  6. Aggregate QoE/VMAF/Rebuf/Switches/Intervention-rate.

Why this is fair
----------------
Different shields can lead to different buffer trajectories, so absolute
numbers may differ slightly from a full retrain+eval. However, the *relative*
ordering across shield configurations is what we need: this design holds the
policy constant and varies only the shield.

Outputs (default, isolated from v12 baseline artifacts)
-------------------------------------------------------
  results/v121_vmaf_shield/replay_episodes.csv  per-episode metrics
  results/v121_vmaf_shield/replay_summary.csv   aggregated means
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.environment.abr_multi_env_v12 import ABREnv
from src.training.safety_shield_v12 import ShieldConfig, safe_adjust_action

PATHS = get_paths()
EVAL_REBUF_PENALTY = 4.3
EVAL_SMOOTH_PENALTY = 1.0

# Isolated output folder so this experiment never overwrites v12 baseline files.
V121_OUT_DIR = PATHS["results"] / "v121_vmaf_shield"


def _make_env(video: str) -> ABREnv:
    return ABREnv(
        video_names=[video],
        trace_dir=str(PATHS["test_traces"]),
        vmaf_dir=str(PATHS["vmaf_scores"]),
        siti_dir=str(PATHS["content_features"]),
        max_chunks=48,
        random_seed=12345,
        use_future=True,
        use_lyapunov=True,
    )


def replay_episode(env: ABREnv, raw_actions: np.ndarray, shield_cfg: ShieldConfig, seed: int) -> dict:
    obs, info = env.reset(seed=int(seed))
    qoe = 0.0
    total_vmaf = 0.0
    total_rebuf_s = 0.0
    last_vmaf = float(getattr(env, "last_vmaf", 35.0))
    last_action = -1
    switches = 0
    interventions = 0
    chunks = 0

    for k in range(min(len(raw_actions), env.max_chunks)):
        if env.chunk_idx >= env.max_chunks:
            break

        raw_action = int(raw_actions[k])
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

        if done:
            break

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
    cfgs: dict[str, ShieldConfig] = {}

    cfgs["shield_off"] = ShieldConfig(level="off")
    cfgs["shield_legacy"] = ShieldConfig(level="light", vmaf_aware=False)
    cfgs["shield_legacy_riskgate"] = ShieldConfig(
        level="light", vmaf_aware=False, only_when_risky=True, risky_dl_over_buf_ratio=1.10
    )

    for tol in [0.8, 1.0, 1.2, 1.5]:
        for budget in [4.0, 8.0, 12.0]:
            key = f"vmaf_aware_tol{tol:.1f}_bud{int(budget):02d}"
            cfgs[key] = ShieldConfig(
                level="light",
                vmaf_aware=True,
                soft_tolerance=float(tol),
                vmaf_loss_budget=float(budget),
            )

    for tol in [1.0, 1.2]:
        for budget in [8.0, 12.0]:
            key = f"vmaf_riskgate_tol{tol:.1f}_bud{int(budget):02d}"
            cfgs[key] = ShieldConfig(
                level="light",
                vmaf_aware=True,
                soft_tolerance=float(tol),
                vmaf_loss_budget=float(budget),
                only_when_risky=True,
                risky_dl_over_buf_ratio=1.10,
            )

    return cfgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision-log",
        type=str,
        default=str(PATHS["results"] / "decision_log_v12_v12_policy.csv"),
    )
    parser.add_argument(
        "--source-method",
        type=str,
        default="Proposed",
        help="Which method's raw actions to replay (Proposed = no-shield-trained).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(V121_OUT_DIR / "replay_episodes.csv"),
        help="Per-episode CSV path. Default lives under "
             "results/v121_vmaf_shield/ to stay separate from "
             "the v12 baseline artifacts.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default="bigbuckbunny,crowd_run,tearsofsteel_short,sintel",
    )
    args = parser.parse_args()

    decision_log = Path(args.decision_log)
    if not decision_log.exists():
        print(f"[ERROR] decision log not found: {decision_log}")
        sys.exit(1)

    df = pd.read_csv(decision_log)
    raw = df[df["Method"] == args.source_method].copy()
    if raw.empty:
        print(f"[ERROR] no rows for method '{args.source_method}'")
        sys.exit(1)

    test_videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    cfgs = build_shield_grid()

    print(f"Replaying raw actions from method='{args.source_method}'")
    print(f"Videos: {test_videos}")
    print(f"Shield configs: {len(cfgs)}")
    print()

    rows = []
    for cfg_name, cfg in cfgs.items():
        for video in test_videos:
            sub = raw[raw["Video"] == video]
            episodes = sorted(sub["Episode"].unique())
            if not episodes:
                continue

            env = _make_env(video)
            for ep in episodes:
                ep_actions = (
                    sub[sub["Episode"] == ep]
                    .sort_values("Chunk")["Action"]
                    .values
                )
                m = replay_episode(env, ep_actions, cfg, seed=int(ep))
                m.update({"Method": cfg_name, "Video": video, "Episode": int(ep)})
                rows.append(m)

        n_eps = sum(1 for r in rows if r["Method"] == cfg_name)
        sub = [r for r in rows if r["Method"] == cfg_name]
        if sub:
            mq = float(np.mean([r["QoE"] for r in sub]))
            mv = float(np.mean([r["VMAF"] for r in sub]))
            mr = float(np.mean([r["Rebuffer"] for r in sub]))
            mi = float(np.mean([r["Intervention_Rate"] for r in sub])) * 100.0
            print(
                f"  {cfg_name:38s} n={n_eps:3d}  "
                f"QoE={mq:7.1f}  VMAF={mv:5.2f}  Rebuf={mr:5.2f}%  Interv={mi:5.2f}%"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-episode results -> {out_path}")

    summary = (
        out_df.groupby("Method")
        .agg(
            QoE_mean=("QoE", "mean"),
            QoE_ci_lo=("QoE", lambda x: np.percentile(x, 2.5)),
            QoE_ci_hi=("QoE", lambda x: np.percentile(x, 97.5)),
            VMAF_mean=("VMAF", "mean"),
            Rebuf_pct=("Rebuffer", "mean"),
            Switch_mean=("Switch", "mean"),
            Interv_rate_pct=("Intervention_Rate", lambda x: float(np.mean(x)) * 100.0),
            n=("QoE", "count"),
        )
        .round(2)
        .sort_values("QoE_mean", ascending=False)
    )
    # Sibling summary file in the same folder.
    summary_path = out_path.with_name(out_path.stem.replace("episodes", "summary") + ".csv")
    if summary_path == out_path:
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved summary             -> {summary_path}")
    print("\n=== SHIELD REPLAY SUMMARY (sorted by mean QoE) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
