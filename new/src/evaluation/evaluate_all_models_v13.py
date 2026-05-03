"""
Master evaluation script (v13).

V13 can evaluate freshly trained v13 models. If a v13 checkpoint is missing,
it falls back to the corresponding v12 checkpoint so guard designs can be
tested immediately without retraining.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from stable_baselines3 import PPO

try:
    from scipy import stats as sp_stats
except Exception:
    sp_stats = None

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.baselines.bba import BBA
from src.environment.abr_multi_env_v13 import ABREnv
from src.training.safety_shield_v13 import ShieldConfigV13, SafetyShieldWrapperV13

from evaluate_all_models_v12 import (  # noqa: E402
    ContentBlindWrapper,
    EVAL_REBUF_PENALTY,
    EVAL_SMOOTH_PENALTY,
    VBRAwareFugu,
    VBRAwareGenie,
    VBRAwareRobustMPC,
)

PATHS = get_paths()

MODEL_SOURCES = {
    "Proposed_V13_Base": [("master_v13", "proposed_v13"), ("master_v12", "proposed_v12")],
    "Proposed_V13_SoftGuard": [("master_v13", "proposed_v13"), ("master_v12", "proposed_v12")],
    "Proposed_V13_TightGuard": [("master_v13", "proposed_v13"), ("master_v12", "proposed_v12")],
    "Proposed_V13_TrainGuard": [("master_v13", "proposed_guarded_qoe_v13"), ("master_v12", "proposed_shielded_riskgate_v12")],
    "Pensieve_V13": [("master_v13", "pensieve_v13"), ("master_v12", "pensieve_v12")],
}

MODEL_ENV_CONFIG = {
    "Proposed_V13_Base": {"use_future": True, "use_lyapunov": True},
    "Proposed_V13_SoftGuard": {"use_future": True, "use_lyapunov": True},
    "Proposed_V13_TightGuard": {"use_future": True, "use_lyapunov": True},
    "Proposed_V13_TrainGuard": {"use_future": True, "use_lyapunov": True},
    "Pensieve_V13": {"use_future": False, "use_lyapunov": False},
}


def _resolve_model_path(master: str, folder: str) -> Optional[Path]:
    base = PATHS["models"] / master / folder
    best_path = base / "best_model" / "best_model"
    if best_path.with_suffix(".zip").exists():
        return best_path
    final_path = base / "final_model"
    if final_path.with_suffix(".zip").exists():
        return final_path
    return None


def load_rl_model(display_name: str, sources):
    for master, folder in sources:
        resolved = _resolve_model_path(master, folder)
        if resolved is not None:
            return PPO.load(str(resolved)), resolved
    raise FileNotFoundError(f"No model found for {display_name} in {sources}")


def _normalize_methods(methods: Optional[Iterable[str]]) -> Optional[set[str]]:
    if methods is None:
        return None
    normalized = {m.strip() for m in methods if m.strip()}
    return normalized or None


def build_methods(only: Optional[Iterable[str]] = None):
    selected = _normalize_methods(only)
    methods: Dict[str, object] = {}
    for display_name, sources in MODEL_SOURCES.items():
        if selected and display_name not in selected:
            continue
        try:
            model, path = load_rl_model(display_name, sources)
            methods[display_name] = model
            print(f"Loaded {display_name} from {path}")
        except Exception as exc:
            print(f"[WARN] Missing {display_name}: {exc}")

    builtin = {
        "RobustMPC": "mpc",
        "Genie": "genie",
        "BBA": BBA(ABREnv.BITRATE_LEVELS),
        "Fugu": "fugu",
    }
    for name, impl in builtin.items():
        if not selected or name in selected:
            methods[name] = impl
    return methods


def _make_env(video_name: str, use_future: bool = False, use_lyapunov: bool = False):
    return ABREnv(
        video_names=[video_name],
        trace_dir=str(PATHS["test_traces"]),
        vmaf_dir=str(PATHS["vmaf_scores"]),
        siti_dir=str(PATHS["content_features"]),
        max_chunks=48,
        random_seed=12345,
        use_future=use_future,
        use_lyapunov=use_lyapunov,
    )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _soft_guard_cfg() -> ShieldConfigV13:
    return ShieldConfigV13(
        level="qoe",
        safety_tp_scale=_env_float("ABR_V13_TP_SCALE", 0.97),
        min_guard_action=_env_int("ABR_V13_MIN_GUARD_ACTION", 1),
        risky_dl_over_buf_ratio=_env_float("ABR_V13_RISK_RATIO", 1.35),
        max_predicted_stall_s=_env_float("ABR_V13_ALLOWED_STALL", 0.25),
        max_downgrade_steps=_env_int("ABR_V13_MAX_DOWNGRADE", 2),
        smooth_recovery=_env_bool("ABR_V13_SMOOTH_RECOVERY", True),
        recovery_window=_env_int("ABR_V13_RECOVERY_WINDOW", 3),
        max_recovery_upshift=_env_int("ABR_V13_MAX_RECOVERY_UPSHIFT", 1),
        recovery_buffer_s=_env_float("ABR_V13_RECOVERY_BUFFER", 6.0),
    )


def _tight_guard_cfg() -> ShieldConfigV13:
    return ShieldConfigV13(
        level="qoe",
        safety_tp_scale=_env_float("ABR_V13_TIGHT_TP_SCALE", 0.95),
        min_guard_action=_env_int("ABR_V13_TIGHT_MIN_GUARD_ACTION", 1),
        risky_dl_over_buf_ratio=_env_float("ABR_V13_TIGHT_RISK_RATIO", 1.15),
        max_predicted_stall_s=_env_float("ABR_V13_TIGHT_ALLOWED_STALL", 0.10),
        max_downgrade_steps=_env_int("ABR_V13_TIGHT_MAX_DOWNGRADE", 3),
        smooth_recovery=_env_bool("ABR_V13_TIGHT_SMOOTH_RECOVERY", True),
        recovery_window=_env_int("ABR_V13_TIGHT_RECOVERY_WINDOW", 3),
        max_recovery_upshift=_env_int("ABR_V13_TIGHT_MAX_RECOVERY_UPSHIFT", 1),
        recovery_buffer_s=_env_float("ABR_V13_TIGHT_RECOVERY_BUFFER", 6.0),
    )


def _wrap_method_env(method_name: str, env):
    if method_name == "Proposed_V13_SoftGuard":
        return SafetyShieldWrapperV13(env, cfg=_soft_guard_cfg())
    if method_name == "Proposed_V13_TightGuard":
        return SafetyShieldWrapperV13(env, cfg=_tight_guard_cfg())
    if method_name == "Proposed_V13_TrainGuard":
        return SafetyShieldWrapperV13(env, cfg=_soft_guard_cfg())
    return env


def _applied_action(info: dict, fallback_action: int) -> int:
    return int(info.get("shielded_action", info.get("applied_action", fallback_action)))


def run_eval(episodes_per_video: int = 20, suffix: str = "_v13_policy", methods: Optional[Iterable[str]] = None):
    results = []
    chunk_decisions = []
    test_videos = ["bigbuckbunny", "crowd_run", "tearsofsteel_short", "sintel"]
    active_methods = build_methods(methods)
    if not active_methods:
        print("No methods to evaluate.")
        return None

    for video_name in test_videos:
        for name, model in active_methods.items():
            env_cfg = MODEL_ENV_CONFIG.get(name, {"use_future": False, "use_lyapunov": False})
            env = _make_env(video_name, **env_cfg)
            eval_env = _wrap_method_env(name, env)
            if name == "Pensieve_V13":
                eval_env = ContentBlindWrapper(eval_env)

            for ep in range(episodes_per_video):
                obs, info = eval_env.reset(seed=ep)
                done = False
                last_br = None
                switches = 0
                last_tp = 2000.0
                prev_vmaf = getattr(env, "last_vmaf", 35.0)

                if name == "RobustMPC":
                    active_model = VBRAwareRobustMPC(env)
                elif name == "Genie":
                    active_model = VBRAwareGenie(env)
                elif name == "Fugu":
                    active_model = VBRAwareFugu(env)
                else:
                    active_model = model

                while not done:
                    chunk_idx_before = env.chunk_idx
                    buffer_before = env.buffer_level

                    if name == "RobustMPC":
                        action = active_model.select_bitrate(info["buffer_level"], last_tp, getattr(env, "last_vmaf", 35.0))
                    elif name == "Genie":
                        action = active_model.select_bitrate(env.chunk_idx, env.buffer_level, env.current_trace["throughput_kbps"])
                    elif name == "BBA":
                        action = active_model.select_bitrate(info["buffer_level"])
                    elif name == "Fugu":
                        action = active_model.select_bitrate(info["buffer_level"], last_tp, getattr(env, "last_vmaf", 35.0))
                    else:
                        action, _ = active_model.predict(obs, deterministic=True)

                    action = int(action)
                    obs, _, done, _, info = eval_env.step(action)

                    applied_action = _applied_action(info, action)
                    if last_br is not None and applied_action != last_br:
                        switches += 1
                    last_br = applied_action

                    bitrate_kbps = int(env.BITRATE_LEVELS[applied_action])
                    step_vmaf = env.vmaf_scores.get(bitrate_kbps, 35.0)
                    step_rebuf = info.get("rebuffer", 0.0)
                    step_tp = info.get("throughput", last_tp)
                    smooth_pen = 0.0 if chunk_idx_before == 0 else abs(step_vmaf - prev_vmaf)
                    step_qoe = step_vmaf - EVAL_REBUF_PENALTY * step_rebuf - EVAL_SMOOTH_PENALTY * smooth_pen

                    chunk_decisions.append(
                        {
                            "Method": name,
                            "Video": video_name,
                            "Episode": ep,
                            "Chunk": chunk_idx_before,
                            "Action": action,
                            "Applied_Action": applied_action,
                            "Bitrate_kbps": bitrate_kbps,
                            "Throughput_kbps": round(step_tp, 1),
                            "Buffer_Before": round(buffer_before, 2),
                            "Buffer_After": round(info.get("buffer", 0.0), 2),
                            "Rebuffer_s": round(step_rebuf, 3),
                            "VMAF": round(step_vmaf, 2),
                            "Smooth_Penalty": round(smooth_pen, 2),
                            "Step_QoE": round(step_qoe, 2),
                            "Shield_Intervened": int(info.get("shield_intervened", 0)),
                            "Shield_Rate": float(info.get("shield_intervention_rate", 0.0)),
                            "Shield_Reason": info.get("shield_reason", ""),
                        }
                    )

                    prev_vmaf = step_vmaf
                    last_tp = step_tp

                qoe = info["total_quality"] - EVAL_REBUF_PENALTY * info["total_rebuffer"] - EVAL_SMOOTH_PENALTY * info["total_smoothness"]
                video_duration = env.chunk_idx * 4.0
                rebuf_ratio = (info["total_rebuffer"] / video_duration) * 100 if video_duration > 0 else 0
                results.append(
                    {
                        "Method": name,
                        "Video": video_name,
                        "Episode": ep,
                        "VMAF": info["avg_quality"],
                        "Rebuffer": rebuf_ratio,
                        "QoE": qoe,
                        "Switch": switches,
                    }
                )

    df = pd.DataFrame(results)
    out_csv = PATHS["results"] / f"detailed_stats_master_v13{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved episode results : {out_csv}")

    df_chunks = pd.DataFrame(chunk_decisions)
    decisions_csv = PATHS["results"] / f"decision_log_v13{suffix}.csv"
    df_chunks.to_csv(decisions_csv, index=False)
    print(f"Saved decision log    : {decisions_csv}")

    if sp_stats is not None and "Genie" in df["Method"].values:
        genie = df[df["Method"] == "Genie"][["Video", "Episode", "QoE"]].rename(columns={"QoE": "QoE_genie"})
        for method_name in sorted([x for x in df["Method"].unique() if x != "Genie"]):
            mdf = df[df["Method"] == method_name][["Video", "Episode", "QoE"]].rename(columns={"QoE": "QoE_m"})
            paired = mdf.merge(genie, on=["Video", "Episode"], how="inner")
            if len(paired) < 10:
                continue
            try:
                _, pval = sp_stats.wilcoxon(paired["QoE_m"].values, paired["QoE_genie"].values)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                print(f"  {method_name:26s}: n={len(paired):3d} p={pval:.4f} {sig}")
            except Exception:
                pass

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--suffix", type=str, default="_v13_policy")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method names.")
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    run_eval(episodes_per_video=args.episodes, suffix=args.suffix, methods=methods)


if __name__ == "__main__":
    main()
