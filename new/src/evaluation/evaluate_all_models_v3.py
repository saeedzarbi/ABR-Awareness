"""
Master evaluation script (V3) for all methods:
- Proposed, Ablation_Base, Ablation_Future, Ablation_Lyap, Pensieve
- RobustMPC, Genie, BBA, Fugu

Reads trained models from:
  results/models/master_v3/<model_name>/
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.baselines.bba import BBA
from src.baselines.mpc_vmaf import RobustMPC
from src.environment.abr_multi_env_v2 import ABREnv

PATHS = get_paths()


MODEL_DIRS = {
    "Proposed": "proposed",
    "Ablation_Base": "ablation_base",
    "Ablation_Future": "ablation_future",
    "Ablation_Lyap": "ablation_lyap",
    "Pensieve": "pensieve",
}


class VBRAwareGenie:
    def __init__(self, env):
        self.env = env
        self.bitrate_levels = env.BITRATE_LEVELS
        self.n_actions = len(self.bitrate_levels)
        self.chunk_duration = env.CHUNK_DURATION
        self.buffer_max = env.BUFFER_MAX
        self.max_chunks = env.max_chunks
        self.rebuf_penalty = env.REBUF_PENALTY_BASE
        self.smooth_penalty = env.SMOOTH_PENALTY_WEIGHT
        self.min_tp = env.MIN_NETWORK_THROUGHPUT
        self.max_tp = env.MAX_NETWORK_THROUGHPUT
        self.buf_step = 0.25
        self.n_buf = int(self.buffer_max / self.buf_step) + 1
        self._policy = None
        self._last_action = 0

    def _buf_idx(self, buf):
        return int(np.clip(round(buf / self.buf_step), 0, self.n_buf - 1))

    def select_bitrate(self, chunk_idx, buffer_level, trace_throughput):
        if chunk_idx == 0:
            self._solve_dp(trace_throughput)
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
        self._policy = np.zeros((n_k, n_a, self.n_buf), dtype=np.int32)

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
                    self._policy[k, last_a, bi] = best_act
            V = V_new


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
                qoe = (
                    vmaf_est
                    - self.env.REBUF_PENALTY_BASE * rebuf
                    - self.env.SMOOTH_PENALTY_WEIGHT * abs(vmaf_est - sim_last_vmaf)
                )
                tot_qoe += qoe
                sim_last_vmaf = vmaf_est
            if tot_qoe > max_qoe:
                max_qoe = tot_qoe
                best_action = br_idx

        return best_action


def load_rl_model(folder_name: str):
    model_base = PATHS["models"] / "master_v3" / folder_name
    best_path = model_base / "best_model" / "best_model"
    final_path = model_base / "final_model"
    path = best_path if best_path.with_suffix(".zip").exists() else final_path
    return PPO.load(str(path))


def build_methods():
    methods = {}
    for display_name, folder in MODEL_DIRS.items():
        try:
            methods[display_name] = load_rl_model(folder)
            print(f"Loaded {display_name}")
        except Exception as exc:
            print(f"[WARN] Missing {display_name}: {exc}")

    methods["RobustMPC"] = "mpc"
    methods["Genie"] = "genie"
    methods["BBA"] = BBA(ABREnv.BITRATE_LEVELS)
    methods["Fugu"] = "fugu"
    return methods


def run_eval(episodes_per_video: int = 20):
    results = []
    test_videos = ["bigbuckbunny", "crowd_run", "tearsofsteel_short", "sintel"]
    methods = build_methods()
    if not methods:
        print("No methods to evaluate.")
        return None

    for video_name in test_videos:
        print(f"\nEvaluating on {video_name}")
        env = ABREnv(
            video_names=[video_name],
            trace_dir=str(PATHS["test_traces"]),
            vmaf_dir=str(PATHS["vmaf_scores"]),
            siti_dir=str(PATHS["content_features"]),
            max_chunks=48,
            random_seed=12345,
        )

        for name, model in methods.items():
            print(f"  - {name}", end="\r")
            for ep in range(episodes_per_video):
                obs, info = env.reset(seed=ep)
                done = False
                last_br = None
                switches = 0
                last_tp = 2000.0

                # Per-episode fresh instances for stateful baselines.
                if name == "RobustMPC":
                    active_model = RobustMPC(env)
                elif name == "Genie":
                    active_model = VBRAwareGenie(env)
                elif name == "Fugu":
                    active_model = VBRAwareFugu(env)
                else:
                    active_model = model

                while not done:
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
                    if last_br is not None and action != last_br:
                        switches += 1
                    last_br = action

                    obs, _, done, _, info = env.step(action)
                    last_tp = info.get("throughput", last_tp)

                qoe = (
                    info["total_quality"]
                    - env.REBUF_PENALTY_BASE * info["total_rebuffer"]
                    - env.SMOOTH_PENALTY_WEIGHT * info["total_smoothness"]
                )
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
        print(f"  Done: {video_name}")

    df = pd.DataFrame(results)
    out_csv = PATHS["results"] / "detailed_stats_master_v3.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed results: {out_csv}")

    summary = df.groupby("Method").agg(
        {
            "QoE": ["mean", "std"],
            "VMAF": ["mean", "std"],
            "Rebuffer": ["mean", "std"],
            "Switch": ["mean"],
        }
    ).round(2)
    print("\nOVERALL STATISTICAL SUMMARY (V3):")
    print(summary)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    run_eval(episodes_per_video=args.episodes)


if __name__ == "__main__":
    main()
