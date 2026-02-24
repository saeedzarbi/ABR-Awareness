"""
Master Evaluation Script V2 — Fair comparison with VBR environment.

Key fixes from V1:
  1. Uses abr_multi_env_v2 → self.np_random seeding guarantees ALL methods
     see the exact same (video, trace, initial_conditions) per episode.
  2. VBR-aware Genie (oracle knows actual chunk sizes).
  3. VBR-aware Fugu (manifest provides chunk sizes in DASH).
  4. MPC throughput history reset per episode (no cross-episode leakage).
"""

import sys
from pathlib import Path
import random
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_multi_env_v2 import ABREnv
from src.baselines.mpc_vmaf import RobustMPC
from src.baselines.bba import BBA
from configs.paths import get_paths
import numpy as np
import pandas as pd
import argparse

PATHS = get_paths()


# ============================================================================
# VBR-aware baselines (override only what changes)
# ============================================================================
class VBRAwareGenie:
    """Offline-optimal DP that uses actual VBR chunk sizes."""

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
        vmaf_vals = np.array(
            [vmaf_map.get(int(br), 35.0) for br in self.bitrate_levels]
        )

        dl_times = np.zeros((n_k, n_a))
        for k in range(n_k):
            tp_idx = int(k * self.chunk_duration) % len(trace_tp)
            tp = np.clip(trace_tp[tp_idx], self.min_tp, self.max_tp)
            for a in range(n_a):
                chunk_bits = self.env.get_chunk_size_bits(
                    int(self.bitrate_levels[a]), k
                )
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
                        new_buf = min(
                            max(0.0, buf - dl) + self.chunk_duration,
                            self.buffer_max,
                        )
                        new_bi = self._buf_idx(new_buf)
                        reward = (
                            vmaf_vals[a]
                            - self.rebuf_penalty * rebuf
                            - self.smooth_penalty
                            * abs(vmaf_vals[a] - vmaf_vals[last_a])
                        )
                        val = reward + V[a, new_bi]
                        if val > best_val:
                            best_val, best_act = val, a
                    V_new[last_a, bi] = best_val
                    self._policy[k, last_a, bi] = best_act
            V = V_new


class VBRAwareFugu:
    """Fugu-style learned predictor using manifest chunk sizes."""

    def __init__(self, env, noise_level=0.10):
        self.env = env
        self.noise_level = noise_level

    def select_bitrate(self, buffer_level, last_throughput, last_vmaf):
        trace_tp = self.env.current_trace['throughput_kbps']
        true_tp = trace_tp[
            int(self.env.chunk_idx * self.env.CHUNK_DURATION) % len(trace_tp)
        ]
        noise = random.uniform(
            1.0 - self.noise_level, 1.0 + self.noise_level
        )
        predicted_tp = true_tp * noise

        best_action = 0
        max_qoe = -float('inf')
        horizon = 5

        for br_idx in range(len(self.env.BITRATE_LEVELS)):
            tot_qoe = 0
            curr_buffer = buffer_level
            sim_last_vmaf = last_vmaf

            for h in range(horizon):
                br = self.env.BITRATE_LEVELS[br_idx]
                chunk_idx = self.env.chunk_idx + h
                size = self.env.get_chunk_size_bits(int(br), chunk_idx)
                dl_time = size / (predicted_tp * 1000.0 + 1e-6)
                rebuf = max(0, dl_time - curr_buffer)
                curr_buffer = max(0, curr_buffer - dl_time) \
                              + self.env.CHUNK_DURATION

                vmaf_est = self.env.vmaf_scores.get(int(br), 35.0)
                qoe = (vmaf_est
                       - self.env.REBUF_PENALTY_BASE * rebuf
                       - self.env.SMOOTH_PENALTY_WEIGHT
                       * abs(vmaf_est - sim_last_vmaf))
                tot_qoe += qoe
                sim_last_vmaf = vmaf_est

            if tot_qoe > max_qoe:
                max_qoe = tot_qoe
                best_action = br_idx

        return best_action


# ============================================================================
# Evaluation Logger
# ============================================================================
class EvaluationLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_logs = []
        self.episode_logs = []

    def log_chunk(self, env, action, info, episode_num):
        self.chunk_logs.append({
            'episode': episode_num,
            'chunk': env.chunk_idx - 1,
            'video': env.current_video_name,
            'action': int(action),
            'bitrate': int(env.BITRATE_LEVELS[action]),
            'buffer': float(info['buffer_level']),
            'throughput': float(info['throughput']),
            'rebuffer': float(info.get('rebuffer', 0)),
            'vmaf': float(env.last_vmaf),
            'reward': float(info.get('reward', 0)),
        })

    def log_episode(self, env, total_reward, switches, episode_num):
        reb_rate = (
            (env.total_rebuffer / (env.chunk_idx * 4.0)) * 100
            if env.chunk_idx > 0 else 0
        )
        self.episode_logs.append({
            'episode': episode_num,
            'video': env.current_video_name,
            'trace_idx': env.current_trace_idx,
            'total_reward': float(total_reward),
            'avg_vmaf': float(env.total_quality / env.chunk_idx),
            'total_rebuffer': float(env.total_rebuffer),
            'rebuffer_rate': float(reb_rate),
            'total_smooth': float(env.total_smooth),
            'switches': int(switches),
        })

    def save_logs(self, method_name='method'):
        if self.chunk_logs:
            pd.DataFrame(self.chunk_logs).to_csv(
                self.log_dir / f'{method_name}_chunks.csv', index=False
            )
        if self.episode_logs:
            pd.DataFrame(self.episode_logs).to_csv(
                self.log_dir / f'{method_name}_episodes.csv', index=False
            )


# ============================================================================
# Main Evaluator
# ============================================================================
class TCSVT_Evaluator:
    def __init__(self):
        self.test_trace_dir = PATHS['test_traces']
        self.results_detailed = []
        self.test_videos = [
            'bigbuckbunny', 'crowd_run', 'tearsofsteel_short', 'sintel',
        ]

    def load_methods(self):
        methods = {}

        model_specs = {
            'Proposed': 'ppo_proposed_v4',
            'Ablation_Base': 'ablation_base_ppo_v2',
            'Ablation_Future': 'ablation_ppo_future_v2',
            'Ablation_Lyap': 'ablation_ppo_lyapunov_v2',
        }
        for name, folder in model_specs.items():
            try:
                path = PATHS['models'] / folder / 'best_model' / 'best_model'
                if not path.with_suffix('.zip').exists():
                    path = PATHS['models'] / folder / 'final_model'
                methods[name] = PPO.load(str(path))
                print(f"✅ Loaded {name} from: {path}")
            except Exception as e:
                print(f"⚠️  {name} missing: {e}")

        methods['RobustMPC'] = 'mpc'
        methods['Genie'] = 'genie'
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
        methods['Fugu'] = 'fugu'

        return methods

    def evaluate(self, methods, episodes_per_video=20, enable_logging=True):
        print(f"\n🔬 Running Master Evaluation V2 (N={episodes_per_video})...")
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No traces found in test_traces folder!")
            return

        for video_idx, video_name in enumerate(self.test_videos, 1):
            print(f"\n📹 Video: {video_name} "
                  f"({video_idx}/{len(self.test_videos)})")

            env = ABREnv(
                video_names=[video_name],
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345,
            )

            for name, model in methods.items():
                print(f"   > {name}...", end='\r')
                if enable_logging:
                    logger = EvaluationLogger(
                        PATHS['logs'] / 'evaluation_master_v2'
                    )

                for ep in range(episodes_per_video):
                    # Same seed → same (trace, initial_conditions) for every
                    # method, thanks to np_random-based seeding in env_v2.
                    obs, info = env.reset(seed=ep)
                    done = False
                    last_br, switches, last_tp = 0, 0, 2000.0
                    total_reward = 0

                    # Per-episode baseline instances (clean state)
                    if name == 'RobustMPC':
                        active_model = RobustMPC(env)
                    elif name == 'Genie':
                        active_model = VBRAwareGenie(env)
                    elif name == 'Fugu':
                        active_model = VBRAwareFugu(env)
                    elif name == 'BBA':
                        active_model = model
                    else:
                        active_model = model

                    while not done:
                        if name == 'RobustMPC':
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(
                                info['buffer_level'], last_tp, cur_vmaf
                            )
                        elif name == 'Genie':
                            trace_tp = env.current_trace['throughput_kbps']
                            action = active_model.select_bitrate(
                                env.chunk_idx, env.buffer_level, trace_tp
                            )
                        elif name == 'BBA':
                            action = active_model.select_bitrate(
                                info['buffer_level']
                            )
                        elif name == 'Fugu':
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(
                                info['buffer_level'], last_tp, cur_vmaf
                            )
                        else:
                            try:
                                expected_dim = \
                                    active_model.observation_space.shape[0]
                            except Exception:
                                expected_dim = len(obs)

                            curr_obs = obs[:expected_dim].copy()

                            if (expected_dim == 29
                                    and name in ['Ablation_Base',
                                                 'Ablation_Lyap']):
                                curr_obs[23:] = 0.0

                            action, _ = active_model.predict(
                                curr_obs, deterministic=True
                            )

                        if action != last_br:
                            switches += 1
                        last_br = action

                        obs, reward, done, _, info = env.step(action)
                        total_reward += reward
                        last_tp = info.get('throughput', last_tp)

                        if enable_logging and name == 'Proposed':
                            logger.log_chunk(env, action, info, ep)

                    qoe = (
                        info['total_quality']
                        - env.REBUF_PENALTY_BASE * info['total_rebuffer']
                        - env.SMOOTH_PENALTY_WEIGHT * info['total_smoothness']
                    )
                    video_duration = env.chunk_idx * 4.0
                    rebuf_ratio = (
                        (info['total_rebuffer'] / video_duration) * 100
                        if video_duration > 0 else 0
                    )

                    self.results_detailed.append({
                        'Method': name,
                        'Video': video_name,
                        'Episode': ep,
                        'VMAF': info['avg_quality'],
                        'Rebuffer': rebuf_ratio,
                        'QoE': qoe,
                        'Switch': switches,
                    })

                    if enable_logging and name == 'Proposed':
                        logger.log_episode(env, total_reward, switches, ep)

                if enable_logging and name == 'Proposed':
                    logger.save_logs(f'{name}_{video_name}')

            print(f"   ✅ {video_name} Done.        ")

    def save_statistics(self):
        if not self.results_detailed:
            return
        df = pd.DataFrame(self.results_detailed)
        path = PATHS['results'] / 'detailed_stats_v2.csv'
        df.to_csv(path, index=False)
        print(f"\n✅ Saved results to: {path}")
        self.print_summary(df)
        return df

    def print_summary(self, df):
        summary = df.groupby('Method').agg({
            'QoE': ['mean', 'std'],
            'VMAF': ['mean', 'std'],
            'Rebuffer': ['mean', 'std'],
            'Switch': ['mean'],
        }).round(2)
        print("\n🏆 OVERALL STATISTICAL SUMMARY (V2):")
        print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--no-logging', action='store_true')
    args = parser.parse_args()

    evaluator = TCSVT_Evaluator()
    methods = evaluator.load_methods()
    if methods:
        evaluator.evaluate(
            methods,
            episodes_per_video=args.episodes,
            enable_logging=not args.no_logging,
        )
        evaluator.save_statistics()
