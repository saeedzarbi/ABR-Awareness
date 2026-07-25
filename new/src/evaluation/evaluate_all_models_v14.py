"""
Master evaluation script (V14, reviewer-response) for all methods.

Changes vs. evaluate_all_models_v12.py
--------------------------------------
1. Corrected QoE surrogate: ``EVAL_REBUF_PENALTY = 100.0`` (one second of stall
   ~ one chunk at max VMAF), matching the training objective.  [review P0.1]
2. Tail metrics: per-episode ``Max_Stall_s`` and ``Any_Stall`` plus a summary
   CSV with P95/P99 rebuffering, stall-free fraction, and CVaR_0.1 of QoE.
   A safety mechanism must be judged on the harm tail, not just means. [P1.6]
3. Seen/unseen tagging: ``sintel`` is held out from training; the summary reports
   metrics for seen videos, the unseen video, and pooled, so generalisation is
   explicit rather than hidden.  [review P0.6 / F5.1]
4. Seed-scoped checkpoints: ``--seed`` selects which training seed's checkpoint
   to load from ``models/master_v14/<folder>/seed_<s>/``.

Outputs (suffix defaults to ``_v14_seed<seed>``):
  detailed_stats_master_v14<suffix>.csv
  summary_master_v14<suffix>.csv
  decision_log_v14<suffix>.csv
  <method>_vs_genie_v14<suffix>.csv
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO

try:
    from scipy import stats as sp_stats
except Exception:
    sp_stats = None

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.baselines.bba import BBA
from src.environment.abr_multi_env_v14 import ABREnv
from src.training.safety_shield_v14 import SafetyShieldWrapper, ShieldConfig
from src.training.shield_aware_wrappers_v12 import HysteresisActionWrapper, HysteresisConfig

PATHS = get_paths()

# One second of rebuffering ~ losing one chunk at maximum perceptual quality.
EVAL_REBUF_PENALTY = 100.0
EVAL_SMOOTH_PENALTY = 1.0

MODEL_TAG = "master_v14"

# Videos seen during training (see train_all_models_v14.Config.TRAIN_VIDEOS).
SEEN_VIDEOS = {"bigbuckbunny", "crowd_run", "tearsofsteel_short"}
UNSEEN_VIDEOS = {"sintel"}
TEST_VIDEOS = ["bigbuckbunny", "crowd_run", "tearsofsteel_short", "sintel"]

MODEL_DIRS = {
    "Proposed": "proposed_v14",
    "Proposed_Shielded": "proposed_shielded_v14",
    "Proposed_ShieldedQoE": "proposed_shielded_qoe_v14",
    "Proposed_ShieldedRiskGate": "proposed_shielded_riskgate_v14",
    "Ablation_Base": "ablation_base_v14",
    "Ablation_Future": "ablation_future_v14",
    "Ablation_Lyap": "ablation_lyap_v14",
    "Pensieve": "pensieve_v14",
}

MODEL_ENV_CONFIG = {
    "Proposed": {"use_future": True, "use_lyapunov": True},
    "Proposed_Shielded": {"use_future": True, "use_lyapunov": True},
    "Proposed_ShieldedQoE": {"use_future": True, "use_lyapunov": True},
    "Proposed_ShieldedRiskGate": {"use_future": True, "use_lyapunov": True},
    "Ablation_Base": {"use_future": False, "use_lyapunov": False},
    "Ablation_Future": {"use_future": True, "use_lyapunov": False},
    "Ablation_Lyap": {"use_future": False, "use_lyapunov": True},
    "Pensieve": {"use_future": False, "use_lyapunov": False},
}


class ContentBlindWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        modified = obs.copy()
        modified[15:] = 0.0
        return modified


class VBRAwareGenie:
    def __init__(self, env):
        self.env = env
        self.bitrate_levels = env.BITRATE_LEVELS
        self.n_actions = len(self.bitrate_levels)
        self.chunk_duration = env.CHUNK_DURATION
        self.buffer_max = env.BUFFER_MAX
        self.max_chunks = env.max_chunks
        self.smooth_penalty = EVAL_SMOOTH_PENALTY
        self.rebuf_penalty = EVAL_REBUF_PENALTY
        self.min_tp = env.MIN_NETWORK_THROUGHPUT
        self.max_tp = env.MAX_NETWORK_THROUGHPUT
        self.buf_step = 0.25
        self.n_buf = int(self.buffer_max / self.buf_step) + 1
        self._policy = None
        self._last_action = 0
        self._policy_cache = {}

    def _buf_idx(self, buf):
        idx = int(round(buf / self.buf_step))
        return max(0, min(idx, self.n_buf - 1))

    def select_bitrate(self, chunk_idx, buffer_level, trace_throughput):
        if chunk_idx == 0:
            cache_key = (
                getattr(self.env, "current_video_name", "unknown"),
                int(getattr(self.env, "current_trace_idx", -1)),
            )
            if cache_key not in self._policy_cache:
                self._policy_cache[cache_key] = self._solve_dp(trace_throughput)
            self._policy = self._policy_cache[cache_key]
            self._last_action = 0
        bi = self._buf_idx(buffer_level)
        action = int(self._policy[chunk_idx, self._last_action, bi])
        self._last_action = action
        return action

    def _solve_dp(self, trace_tp):
        n_a = self.n_actions
        n_k = self.max_chunks
        vmaf_map = self.env.vmaf_scores
        vmaf_vals = np.array([vmaf_map.get(int(br), 35.0) for br in self.bitrate_levels])
        dl_times = np.zeros((n_k, n_a))
        for k in range(n_k):
            tp_idx = int(k * self.chunk_duration) % len(trace_tp)
            tp = np.clip(trace_tp[tp_idx], self.min_tp, self.max_tp)
            for a in range(n_a):
                chunk_bits = self.env.get_chunk_size_bits(int(self.bitrate_levels[a]), k)
                dl_times[k, a] = min(chunk_bits / (tp * 1000.0), 60.0)

        V = np.zeros((n_a, self.n_buf))
        policy = np.zeros((n_k, n_a, self.n_buf), dtype=np.int32)
        for k in range(n_k - 1, -1, -1):
            V_new = np.full((n_a, self.n_buf), -1e9)
            for last_a in range(n_a):
                for bi in range(self.n_buf):
                    buf = bi * self.buf_step
                    best_val, best_act = -1e9, 0
                    for a in range(n_a):
                        dl = dl_times[k, a]
                        rebuf = max(0.0, dl - buf)
                        new_buf = min(max(0.0, buf - dl) + self.chunk_duration, self.buffer_max)
                        new_bi = self._buf_idx(new_buf)
                        reward = (
                            vmaf_vals[a]
                            - self.rebuf_penalty * rebuf
                            - self.smooth_penalty * abs(vmaf_vals[a] - vmaf_vals[last_a])
                        )
                        val = reward + V[a, new_bi]
                        if val > best_val:
                            best_val, best_act = val, a
                    V_new[last_a, bi] = best_val
                    policy[k, last_a, bi] = best_act
            V = V_new
        return policy


class VBRAwareFugu:
    def __init__(self, env, noise_level=0.10):
        self.env = env
        self.noise_level = noise_level

    def select_bitrate(self, buffer_level, last_throughput, last_vmaf):
        trace_tp = self.env.current_trace["throughput_kbps"]
        true_tp = trace_tp[int(self.env.chunk_idx * self.env.CHUNK_DURATION) % len(trace_tp)]
        noise = random.uniform(1.0 - self.noise_level, 1.0 + self.noise_level)
        predicted_tp = true_tp * noise

        best_action = 0
        max_qoe = -float("inf")
        horizon = 5
        for br_idx in range(len(self.env.BITRATE_LEVELS)):
            tot_qoe = 0.0
            curr_buffer = buffer_level
            sim_last_vmaf = last_vmaf
            for h in range(horizon):
                br = self.env.BITRATE_LEVELS[br_idx]
                chunk_idx = self.env.chunk_idx + h
                size = self.env.get_chunk_size_bits(int(br), chunk_idx)
                dl_time = size / (predicted_tp * 1000.0 + 1e-6)
                rebuf = max(0.0, dl_time - curr_buffer)
                curr_buffer = max(0.0, curr_buffer - dl_time) + self.env.CHUNK_DURATION
                vmaf_est = self.env.vmaf_scores.get(int(br), 35.0)
                qoe = vmaf_est - EVAL_REBUF_PENALTY * rebuf - EVAL_SMOOTH_PENALTY * abs(vmaf_est - sim_last_vmaf)
                tot_qoe += qoe
                sim_last_vmaf = vmaf_est
            if tot_qoe > max_qoe:
                max_qoe = tot_qoe
                best_action = br_idx
        return best_action


class VBRAwareRobustMPC:
    def __init__(self, env, search_horizon=3):
        self.env = env
        self.search_horizon = search_horizon
        self.past_throughput = []
        import itertools

        possible_actions = list(range(len(env.BITRATE_LEVELS)))
        self.trajectories = list(itertools.product(possible_actions, repeat=self.search_horizon))

    def _harmonic_mean_tp(self):
        if not self.past_throughput:
            return 2000.0
        samples = self.past_throughput[-5:]
        return len(samples) / sum(1.0 / (t + 1e-6) for t in samples)

    def select_bitrate(self, buffer_level, last_throughput_kbps, last_vmaf):
        if last_throughput_kbps > 0:
            self.past_throughput.append(last_throughput_kbps)
        predicted_tp = self._harmonic_mean_tp()
        chunk_idx = self.env.chunk_idx
        vmaf_map = self.env.vmaf_scores

        best_action = 0
        max_reward = -float("inf")
        for trajectory in self.trajectories:
            cumulative_reward = 0.0
            sim_buffer = buffer_level
            sim_last_vmaf = last_vmaf
            for step, action in enumerate(trajectory):
                br = self.env.BITRATE_LEVELS[action]
                cidx = min(chunk_idx + step, self.env.max_chunks - 1)
                chunk_bits = self.env.get_chunk_size_bits(int(br), cidx)
                download_time = chunk_bits / (predicted_tp * 1000.0 + 1e-6)
                rebuffer = max(0.0, download_time - sim_buffer)
                sim_buffer = max(0.0, sim_buffer - download_time) + self.env.CHUNK_DURATION
                vmaf = vmaf_map.get(int(br), 35.0)
                smoothness = abs(vmaf - sim_last_vmaf)
                reward = vmaf - EVAL_REBUF_PENALTY * rebuffer - EVAL_SMOOTH_PENALTY * smoothness
                cumulative_reward += reward
                sim_last_vmaf = vmaf
                if sim_buffer < 0:
                    cumulative_reward -= 1000
                    break
            if cumulative_reward > max_reward:
                max_reward = cumulative_reward
                best_action = trajectory[0]
        return best_action


def _resolve_model_path(model_base: Path):
    best_path = model_base / "best_model" / "best_model"
    if best_path.with_suffix(".zip").exists():
        return best_path
    final_path = model_base / "final_model"
    if final_path.with_suffix(".zip").exists():
        return final_path
    return None


def load_rl_model(display_name: str, folder_name: str, seed: int):
    base = PATHS["models"] / MODEL_TAG / folder_name / f"seed_{seed}"
    resolved = _resolve_model_path(base)
    if resolved is None:
        raise FileNotFoundError(f"No v14 model found for {display_name} at {base}")
    return PPO.load(str(resolved)), resolved


def build_methods(seed: int):
    methods: Dict[str, object] = {}
    for display_name, folder in MODEL_DIRS.items():
        try:
            model, path = load_rl_model(display_name, folder, seed)
            methods[display_name] = model
            print(f"Loaded {display_name} from {path}")
        except Exception as exc:
            print(f"[WARN] Missing {display_name}: {exc}")
    methods["RobustMPC"] = "mpc"
    methods["Genie"] = "genie"
    methods["BBA"] = BBA(ABREnv.BITRATE_LEVELS)
    methods["Fugu"] = "fugu"
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


def _shield_cfg_from_env(risk_gate: bool = False) -> ShieldConfig:
    level = os.environ.get("ABR_SHIELD_LEVEL", "light").strip().lower()
    if level not in {"off", "light", "strong"}:
        level = "light"
    try:
        risky_ratio = float(os.environ.get("ABR_V14_RISK_RATIO", "1.10"))
    except Exception:
        risky_ratio = 1.10
    only_when_risky = risk_gate or os.environ.get("ABR_V14_RISK_GATE", "0").strip().lower() in {"1", "true", "yes"}
    return ShieldConfig(level=level, only_when_risky=only_when_risky, risky_dl_over_buf_ratio=risky_ratio)


def _wrap_method_env(method_name: str, env):
    if method_name == "Proposed_ShieldedQoE":
        cfg = HysteresisConfig(
            max_step=int(os.environ.get("ABR_HYST_MAX_STEP", "1")),
            min_buf_for_upswitch=float(os.environ.get("ABR_HYST_MIN_BUF", "1.5")),
        )
        env = HysteresisActionWrapper(env, cfg=cfg)
        env = SafetyShieldWrapper(env, cfg=_shield_cfg_from_env(risk_gate=False))
        return env
    if method_name == "Proposed_ShieldedRiskGate":
        return SafetyShieldWrapper(env, cfg=_shield_cfg_from_env(risk_gate=True))
    if method_name == "Proposed_Shielded":
        return SafetyShieldWrapper(env, cfg=_shield_cfg_from_env(risk_gate=False))
    return env


def _applied_action(info: dict, fallback_action: int) -> int:
    return int(info.get("shielded_action", info.get("applied_action", fallback_action)))


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([values[rng.integers(0, values.size, values.size)].mean() for _ in range(n_boot)])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def _cvar_low(values: np.ndarray, alpha: float = 0.1) -> float:
    """CVaR of the lower (worst) alpha tail; for QoE, higher is better so the
    worst outcomes are the smallest values."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * values.size)))
    return float(values[:k].mean())


def run_eval(episodes_per_video: int = 20, seed: int = 0, suffix: str = ""):
    results = []
    chunk_decisions = []
    methods = build_methods(seed)
    rl_methods = [m for m in methods if m in MODEL_DIRS]
    if not rl_methods:
        print("[WARN] No v14 RL checkpoints found for this seed; only baselines will run.")

    for video_name in TEST_VIDEOS:
        for name, model in methods.items():
            env_cfg = MODEL_ENV_CONFIG.get(name, {"use_future": False, "use_lyapunov": False})
            env = _make_env(video_name, **env_cfg)
            eval_env = _wrap_method_env(name, env)
            if name == "Pensieve":
                eval_env = ContentBlindWrapper(eval_env)

            for ep in range(episodes_per_video):
                obs, info = eval_env.reset(seed=ep)
                done = False
                last_br = None
                switches = 0
                last_tp = 2000.0
                prev_vmaf = getattr(env, "last_vmaf", 35.0)
                max_stall = 0.0

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
                        cur_vmaf = getattr(env, "last_vmaf", 35.0)
                        action = active_model.select_bitrate(info["buffer_level"], last_tp, cur_vmaf)
                    elif name == "Genie":
                        trace_tp = env.current_trace["throughput_kbps"]
                        action = active_model.select_bitrate(env.chunk_idx, env.buffer_level, trace_tp)
                    elif name == "BBA":
                        action = active_model.select_bitrate(info["buffer_level"])
                    elif name == "Fugu":
                        cur_vmaf = getattr(env, "last_vmaf", 35.0)
                        action = active_model.select_bitrate(info["buffer_level"], last_tp, cur_vmaf)
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
                    max_stall = max(max_stall, float(step_rebuf))
                    step_tp = info.get("throughput", last_tp)
                    smooth_pen = 0.0 if chunk_idx_before == 0 else abs(step_vmaf - prev_vmaf)
                    step_qoe = step_vmaf - EVAL_REBUF_PENALTY * step_rebuf - EVAL_SMOOTH_PENALTY * smooth_pen

                    chunk_decisions.append({
                        "Method": name, "Video": video_name, "Episode": ep,
                        "Chunk": chunk_idx_before, "Action": action, "Applied_Action": applied_action,
                        "Bitrate_kbps": bitrate_kbps, "Throughput_kbps": round(step_tp, 1),
                        "Buffer_Before": round(buffer_before, 2), "Buffer_After": round(info.get("buffer", 0.0), 2),
                        "Rebuffer_s": round(step_rebuf, 3), "VMAF": round(step_vmaf, 2),
                        "Smooth_Penalty": round(smooth_pen, 2), "Step_QoE": round(step_qoe, 2),
                        "Shield_Intervened": int(info.get("shield_intervened", 0)),
                        "Shield_Rate": float(info.get("shield_intervention_rate", 0.0)),
                    })

                    prev_vmaf = step_vmaf
                    last_tp = step_tp

                qoe = info["total_quality"] - EVAL_REBUF_PENALTY * info["total_rebuffer"] - EVAL_SMOOTH_PENALTY * info["total_smoothness"]
                video_duration = env.chunk_idx * 4.0
                rebuf_ratio = (info["total_rebuffer"] / video_duration) * 100 if video_duration > 0 else 0

                results.append({
                    "Method": name, "Video": video_name, "Episode": ep,
                    "Seen": int(video_name in SEEN_VIDEOS),
                    "VMAF": info["avg_quality"], "Rebuffer": rebuf_ratio, "QoE": qoe,
                    "Switch": switches, "Max_Stall_s": round(max_stall, 3),
                    "Any_Stall": int(info["total_rebuffer"] > 1e-6),
                })

    df = pd.DataFrame(results)
    out_csv = PATHS["results"] / f"detailed_stats_master_v14{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved episode results : {out_csv}")

    _write_summary(df, suffix=suffix)

    df_chunks = pd.DataFrame(chunk_decisions)
    decisions_csv = PATHS["results"] / f"decision_log_v14{suffix}.csv"
    df_chunks.to_csv(decisions_csv, index=False)
    print(f"Saved decision log    : {decisions_csv}")

    for m in ["Proposed", "Proposed_Shielded", "Proposed_ShieldedQoE", "Proposed_ShieldedRiskGate"]:
        _build_comparison(df_chunks, method_name=m, suffix=suffix)

    return df


def _summary_rows(df: pd.DataFrame, scope_label: str):
    rows = []
    for m in sorted(df["Method"].unique()):
        sub = df[df["Method"] == m]
        qoe = sub["QoE"].to_numpy(float)
        rb = sub["Rebuffer"].to_numpy(float)
        q_lo, q_hi = _bootstrap_ci(qoe)
        r_lo, r_hi = _bootstrap_ci(rb)
        rows.append({
            "Scope": scope_label, "Method": m, "n": int(len(sub)),
            "QoE_mean": round(float(qoe.mean()), 2), "QoE_ci_lo": round(q_lo, 2), "QoE_ci_hi": round(q_hi, 2),
            "QoE_CVaR10": round(_cvar_low(qoe, 0.1), 2),
            "VMAF_mean": round(float(sub["VMAF"].mean()), 2),
            "Rebuf_pct_mean": round(float(rb.mean()), 3), "Rebuf_ci_lo": round(r_lo, 3), "Rebuf_ci_hi": round(r_hi, 3),
            "Rebuf_p95": round(float(np.percentile(rb, 95)), 3), "Rebuf_p99": round(float(np.percentile(rb, 99)), 3),
            "Max_Stall_s_p99": round(float(np.percentile(sub["Max_Stall_s"].to_numpy(float), 99)), 3),
            "StallFree_frac": round(float((sub["Any_Stall"] == 0).mean()), 3),
            "Switch_mean": round(float(sub["Switch"].mean()), 2),
        })
    return rows


def _write_summary(df: pd.DataFrame, suffix: str = ""):
    """Summary with means, CIs, and tail metrics, split by seen/unseen/pooled."""
    rows = []
    rows += _summary_rows(df, "pooled")
    rows += _summary_rows(df[df["Seen"] == 1], "seen")
    unseen = df[df["Seen"] == 0]
    if not unseen.empty:
        rows += _summary_rows(unseen, "unseen")
    summary = pd.DataFrame(rows)
    out = PATHS["results"] / f"summary_master_v14{suffix}.csv"
    summary.to_csv(out, index=False)
    print(f"Saved summary         : {out}")
    print("\n=== V14 POOLED SUMMARY (sorted by QoE) ===")
    print(summary[summary["Scope"] == "pooled"].sort_values("QoE_mean", ascending=False).to_string(index=False))


def _build_comparison(df_chunks: pd.DataFrame, method_name: str, suffix: str = ""):
    if method_name not in df_chunks["Method"].values or "Genie" not in df_chunks["Method"].values:
        return
    proposed = df_chunks[df_chunks["Method"] == method_name].copy()
    genie = df_chunks[df_chunks["Method"] == "Genie"].copy()
    merge_keys = ["Video", "Episode", "Chunk"]
    cmp = proposed.merge(
        genie[merge_keys + ["Action", "Bitrate_kbps", "Rebuffer_s", "VMAF", "Step_QoE"]],
        on=merge_keys, suffixes=("_proposed", "_genie"), how="inner",
    )
    cmp["QoE_diff"] = cmp["Step_QoE_proposed"] - cmp["Step_QoE_genie"]
    cmp_csv = PATHS["results"] / f"{method_name.lower()}_vs_genie_v14{suffix}.csv"
    cmp.to_csv(cmp_csv, index=False)
    print(f"Saved comparison log  : {cmp_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Training-seed checkpoint to evaluate.")
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    suffix = args.suffix if args.suffix is not None else f"_v14_seed{args.seed}"
    run_eval(episodes_per_video=args.episodes, seed=args.seed, suffix=suffix)


if __name__ == "__main__":
    main()
