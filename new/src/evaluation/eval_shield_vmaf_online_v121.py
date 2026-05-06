"""
Online evaluation of the V12.1 VMAF-Aware Shield using a trained policy.

Difference vs replay_shield_vmaf_v121.py
----------------------------------------
This script loads a *trained* PPO policy (e.g. proposed_v12) and runs full
on-policy episodes for each shield configuration. Numbers here are
production-faithful; the replay script is a fast offline approximation.

Recommended source policy
-------------------------
Use the policy that was trained WITHOUT a shield in the loop, so the shield
is purely an inference-time intervention and shield variants are comparable:
    --policy proposed_v12

Usage on the server
-------------------
    cd new
    python src/evaluation/eval_shield_vmaf_online_v121.py \
        --policy proposed_v12 \
        --episodes 20

Outputs (default, isolated from v12 baseline artifacts)
-------------------------------------------------------
  results/v121_vmaf_shield/online_episodes.csv  per-episode metrics
  results/v121_vmaf_shield/online_summary.csv   aggregated means + bootstrap CI
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


def _resolve_model_path(folder_name: str) -> Path | None:
    base = PATHS["models"] / "master_v12" / folder_name
    best = base / "best_model" / "best_model"
    if best.with_suffix(".zip").exists():
        return best
    final = base / "final_model"
    if final.with_suffix(".zip").exists():
        return final
    return None


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


def build_shield_grid(quick: bool = False) -> dict[str, ShieldConfig]:
    cfgs: dict[str, ShieldConfig] = {}

    cfgs["shield_off"] = ShieldConfig(level="off")
    cfgs["shield_legacy"] = ShieldConfig(level="light", vmaf_aware=False)
    cfgs["shield_legacy_riskgate"] = ShieldConfig(
        level="light", vmaf_aware=False, only_when_risky=True, risky_dl_over_buf_ratio=1.10
    )

    if quick:
        # Compact grid: representative VMAF-aware and lookahead points
        for tol, budget in [(1.0, 8.0), (1.2, 12.0)]:
            key = f"vmaf_aware_tol{tol:.1f}_bud{int(budget):02d}"
            cfgs[key] = ShieldConfig(
                level="light", vmaf_aware=True,
                soft_tolerance=float(tol), vmaf_loss_budget=float(budget),
            )
        # Lookahead-Rollout gates (V12.2)
        for h, mb in [(2, 0.5), (3, 0.5), (3, 1.0), (5, 1.0)]:
            key = f"lookahead_h{h}_mb{mb:.1f}"
            cfgs[key] = ShieldConfig(
                level="light", lookahead_horizon=int(h),
                lookahead_min_buffer=float(mb),
            )
        # Lookahead + VMAF-aware fallback when rollout fails
        cfgs["lookahead_h3_mb1.0_vmafFB"] = ShieldConfig(
            level="light", lookahead_horizon=3, lookahead_min_buffer=1.0,
            vmaf_aware=True, soft_tolerance=1.0, vmaf_loss_budget=8.0,
        )
        return cfgs

    # Full grid -------------------------------------------------------
    # VMAF-aware sweep
    for tol in [0.8, 1.0, 1.2, 1.5]:
        for budget in [4.0, 8.0, 12.0]:
            key = f"vmaf_aware_tol{tol:.1f}_bud{int(budget):02d}"
            cfgs[key] = ShieldConfig(
                level="light", vmaf_aware=True,
                soft_tolerance=float(tol), vmaf_loss_budget=float(budget),
            )
    for tol in [1.0, 1.2]:
        for budget in [8.0, 12.0]:
            key = f"vmaf_riskgate_tol{tol:.1f}_bud{int(budget):02d}"
            cfgs[key] = ShieldConfig(
                level="light", vmaf_aware=True,
                soft_tolerance=float(tol), vmaf_loss_budget=float(budget),
                only_when_risky=True, risky_dl_over_buf_ratio=1.10,
            )

    # Lookahead-Rollout sweep (V12.2)
    for h in [2, 3, 5]:
        for mb in [0.5, 1.0, 1.5]:
            key = f"lookahead_h{h}_mb{mb:.1f}"
            cfgs[key] = ShieldConfig(
                level="light", lookahead_horizon=int(h),
                lookahead_min_buffer=float(mb),
            )
    # Lookahead with a less-pessimistic rollout throughput
    for h in [3, 5]:
        for ts in [1.0, 0.95]:
            key = f"lookahead_h{h}_mb1.0_ts{ts:.2f}"
            cfgs[key] = ShieldConfig(
                level="light", lookahead_horizon=int(h),
                lookahead_min_buffer=1.0, lookahead_tp_scale=float(ts),
            )
    # Lookahead + VMAF-aware fallback (composite)
    for h in [2, 3, 5]:
        key = f"lookahead_h{h}_mb1.0_vmafFB"
        cfgs[key] = ShieldConfig(
            level="light", lookahead_horizon=int(h), lookahead_min_buffer=1.0,
            vmaf_aware=True, soft_tolerance=1.0, vmaf_loss_budget=8.0,
        )

    # Threshold sweep (V12.3): make the shield more permissive by raising the
    # catastrophic ratio and the risky-gate ratio. Goal: trade a small Rebuf
    # increase for a big QoE recovery, walking the Pareto front.
    for cat in [2.0, 3.0, 4.0, 5.0]:
        key = f"thresh_cat{cat:.1f}"
        cfgs[key] = ShieldConfig(level="light", catastrophic_ratio=float(cat))
    for cat, rg in [(3.0, 1.30), (3.0, 1.50), (4.0, 1.50), (4.0, 2.00),
                    (5.0, 1.50), (5.0, 2.00)]:
        key = f"thresh_cat{cat:.1f}_rg{rg:.2f}"
        cfgs[key] = ShieldConfig(
            level="light", catastrophic_ratio=float(cat),
            only_when_risky=True, risky_dl_over_buf_ratio=float(rg),
        )
    # Permissive shield + VMAF-aware fallback (best of both)
    for cat in [3.0, 4.0]:
        key = f"thresh_cat{cat:.1f}_vmafFB"
        cfgs[key] = ShieldConfig(
            level="light", catastrophic_ratio=float(cat),
            vmaf_aware=True, soft_tolerance=1.2, vmaf_loss_budget=8.0,
        )
    return cfgs


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(values[idx]))
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2)))
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="proposed_v12",
                        help="Folder name under results/models/master_v12/")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--videos", type=str,
                        default="bigbuckbunny,crowd_run,tearsofsteel_short,sintel")
    parser.add_argument("--out", type=str,
                        default=str(V121_OUT_DIR / "online_episodes.csv"),
                        help="Per-episode CSV path. Default lives under "
                             "results/v121_vmaf_shield/ to stay separate from "
                             "the v12 baseline artifacts.")
    parser.add_argument("--quick", action="store_true",
                        help="Use compact shield grid (5 configs vs 19) for fast iteration.")
    args = parser.parse_args()

    model_path = _resolve_model_path(args.policy)
    if model_path is None:
        print(f"[ERROR] No trained model found for policy '{args.policy}'.")
        print(f"        Looked under: {PATHS['models'] / 'master_v12' / args.policy}")
        sys.exit(1)

    print(f"Loading policy: {model_path}")
    model = PPO.load(str(model_path))

    cfgs = build_shield_grid(quick=args.quick)
    test_videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    print(f"Shield configs : {len(cfgs)}")
    print(f"Videos         : {test_videos}")
    print(f"Episodes/video : {args.episodes}")
    print()

    rows = []
    for cfg_name, cfg in cfgs.items():
        for video in test_videos:
            env = _make_env(video)
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
                f"  {cfg_name:38s} n={len(sub):3d}  "
                f"QoE={mq:7.1f}  VMAF={mv:5.2f}  Rebuf={mr:5.2f}%  Interv={mi:5.2f}%"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved per-episode results -> {out_path}")

    summary_rows = []
    for cfg_name in out_df["Method"].unique():
        sub = out_df[out_df["Method"] == cfg_name]
        qoe_lo, qoe_hi = _bootstrap_ci(sub["QoE"].values.astype(float))
        rb_lo, rb_hi = _bootstrap_ci(sub["Rebuffer"].values.astype(float))
        summary_rows.append({
            "Method": cfg_name,
            "n": int(len(sub)),
            "QoE_mean": round(float(sub["QoE"].mean()), 2),
            "QoE_ci_lo": round(qoe_lo, 2),
            "QoE_ci_hi": round(qoe_hi, 2),
            "VMAF_mean": round(float(sub["VMAF"].mean()), 2),
            "Rebuf_pct": round(float(sub["Rebuffer"].mean()), 2),
            "Rebuf_ci_lo": round(rb_lo, 2),
            "Rebuf_ci_hi": round(rb_hi, 2),
            "Switch_mean": round(float(sub["Switch"].mean()), 2),
            "Interv_rate_pct": round(float(sub["Intervention_Rate"].mean()) * 100.0, 2),
        })

    summary = pd.DataFrame(summary_rows).sort_values("QoE_mean", ascending=False)
    # Sibling summary file in the same folder.
    summary_path = out_path.with_name(out_path.stem.replace("episodes", "summary") + ".csv")
    if summary_path == out_path:
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary             -> {summary_path}")
    print("\n=== SHIELD ONLINE SUMMARY (sorted by mean QoE) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
