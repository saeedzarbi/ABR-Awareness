import sys
from pathlib import Path
import os
import json
import random
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_multi_env_l import ABREnv
from src.baselines.mpc_vmaf import RobustMPC
from src.baselines.genie import Genie
from src.baselines.bba import BBA
from configs.paths import get_paths
import numpy as np
import pandas as pd
import argparse

PATHS = get_paths()


class SimulatedFugu:
    """
    Simulates a modern 'Learned Throughput Predictor' (Fugu-style).
    Peeks at the real future throughput with realistic noise (~10% error).
    Uses the same QoE weights as evaluation for internal MPC planning.
    """
    def __init__(self, env, noise_level=0.10):
        self.env = env
        self.noise_level = noise_level

    def select_bitrate(self, buffer_level, last_throughput, last_vmaf):
        true_tp = self.env.current_trace['throughput_kbps'][
            int(self.env.chunk_idx * self.env.CHUNK_DURATION) % len(self.env.current_trace['throughput_kbps'])
        ]
        noise = random.uniform(1.0 - self.noise_level, 1.0 + self.noise_level)
        predicted_tp = true_tp * noise

        best_action = 0
        max_qoe = -float('inf')
        horizon = 5

        for br_idx in range(len(self.env.BITRATE_LEVELS)):
            tot_qoe = 0
            curr_buffer = buffer_level
            sim_last_vmaf = last_vmaf

            for _ in range(horizon):
                br = self.env.BITRATE_LEVELS[br_idx]
                size = br * 1000 * self.env.CHUNK_DURATION
                dl_time = size / (predicted_tp * 1000.0 + 1e-6)
                rebuf = max(0, dl_time - curr_buffer)
                curr_buffer = max(0, curr_buffer - dl_time) + self.env.CHUNK_DURATION

                vmaf_est = self.env.vmaf_scores.get(int(br), 35.0)
                qoe = vmaf_est \
                      - self.env.REBUF_PENALTY_BASE * rebuf \
                      - self.env.SMOOTH_PENALTY_WEIGHT * abs(vmaf_est - sim_last_vmaf)
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
        chunk_data = {
            'episode': episode_num, 'chunk': env.chunk_idx - 1, 'video': env.current_video_name,
            'action': int(action), 'bitrate': int(env.BITRATE_LEVELS[action]),
            'buffer': float(info['buffer_level']), 'throughput': float(info['throughput']),
            'rebuffer': float(info.get('rebuffer', 0)), 'vmaf': float(env.last_vmaf),
            'reward': float(info.get('reward', 0))
        }
        self.chunk_logs.append(chunk_data)
    
    def log_episode(self, env, total_reward, switches, episode_num):
        rebuffer_rate = (env.total_rebuffer / (env.chunk_idx * 4.0)) * 100 if env.chunk_idx > 0 else 0
        episode_data = {
            'episode': episode_num, 'video': env.current_video_name, 'trace_idx': env.current_trace_idx,
            'total_reward': float(total_reward), 'avg_vmaf': float(env.total_quality / env.chunk_idx),
            'total_rebuffer': float(env.total_rebuffer), 'rebuffer_rate': float(rebuffer_rate),
            'total_smooth': float(env.total_smooth), 'switches': int(switches)
        }
        self.episode_logs.append(episode_data)
    
    def save_logs(self, method_name='method'):
        if self.chunk_logs:
            pd.DataFrame(self.chunk_logs).to_csv(self.log_dir / f'{method_name}_chunks5g.csv', index=False)
        if self.episode_logs:
            pd.DataFrame(self.episode_logs).to_csv(self.log_dir / f'{method_name}_episodes5g.csv', index=False)

def send_slack_message(status, step, message):
    pass # (Slack code kept clean for brevity, add back if you actively use it)

# ============================================================================
# Main Evaluator
# ============================================================================
class TCSVT_Evaluator:
    def __init__(self):
        self.test_trace_dir = PATHS['test_traces']
        self.results_detailed = [] 
        self.test_videos = ['bigbuckbunny', 'crowd_run', 'tearsofsteel_short', 'sintel']

    def load_methods(self):
        methods = {}
        
        # 1. Full Proposed Model (V22)
        try:
            path = PATHS['models'] / 'ppo_proposed_v3_lyapunov' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists(): path = PATHS['models'] / 'ppo_proposed_v3_lyapunov' / 'final_model'
            methods['Proposed'] = PPO.load(str(path))
            print(f"✅ Loaded Proposed (Full) from: {path}")
        except Exception as e: print(f"⚠️ Proposed missing: {e}")
        
        # 2. Ablation Models
        ablation_dict = {
            'Ablation_Base': 'ablation_base_ppo',
            'Ablation_Future': 'ablation_ppo_future',
            'Ablation_Lyap': 'ablation_ppo_lyapunov'
        }
        for name, folder in ablation_dict.items():
            try:
                path = PATHS['models'] / folder / 'best_model' / 'best_model'
                if not path.with_suffix('.zip').exists(): path = PATHS['models'] / folder / 'final_model'
                methods[name] = PPO.load(str(path))
                print(f"✅ Loaded {name} from: {path}")
            except Exception as e: print(f"⚠️ {name} missing: {e}")

        # 3. Pensieve
        try:
            path = PATHS['models'] / 'pensieve_multi_vmaf_new_14' / 'best_model' / 'best_model'
            methods['Pensieve'] = PPO.load(str(path))
            print(f"✅ Loaded Pensieve from: {path}")
        except Exception as e: print(f"⚠️ Pensieve missing: {e}")

        # 4. Baselines (instantiated per-video in evaluate())
        methods['RobustMPC'] = 'mpc'
        methods['Genie'] = 'genie'
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
        methods['Fugu'] = 'fugu'

        return methods

    def evaluate(self, methods, episodes_per_video=20, enable_logging=True):
        print(f"\n🔬 Running Master Evaluation (N={episodes_per_video})...")
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No traces found in test_traces folder!")
            return

        for video_idx, video_name in enumerate(self.test_videos, 1):
            print(f"\n📹 Testing on Video: {video_name} ({video_idx}/{len(self.test_videos)})")
            
            env = ABREnv(
                video_names=[video_name],
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48, random_seed=12345
            )

            for name, model in methods.items():
                print(f"   > Running {name}...", end='\r')
                if enable_logging: logger = EvaluationLogger(PATHS['logs'] / 'evaluation_master')
                
                active_model = model
                if name == 'RobustMPC': active_model = RobustMPC(env)
                elif name == 'Genie': active_model = Genie(env)
                elif name == 'Fugu': active_model = SimulatedFugu(env)
                
                for ep in range(episodes_per_video):
                    obs, info = env.reset(seed=ep)
                    done = False
                    last_br, switches, last_tp = 0, 0, 2000.0
                    total_reward = 0
                    
                    while not done:
                        trace_tp = env.current_trace['throughput_kbps']
                        
                        if name == 'RobustMPC':
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(info['buffer_level'], last_tp, cur_vmaf)
                        elif name == 'Genie':
                            action = active_model.select_bitrate(env.chunk_idx, env.buffer_level, trace_tp)
                        elif name == 'BBA':
                            action = active_model.select_bitrate(info['buffer_level'])
                        elif name == 'Fugu':
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(info['buffer_level'], last_tp, cur_vmaf)
                        else:
                            # --- DYNAMIC OBSERVATION SLICING ---
                            try:
                                expected_dim = active_model.observation_space.shape[0]
                            except:
                                expected_dim = len(obs)

                            # بُرش دادن آرایه دقیقاً به اندازه‌ای که مدل در زمان آموزش دیده است
                            curr_obs = obs[:expected_dim].copy()
                            
                            # ماسک کردن اطلاعات آینده برای مدل‌های Ablation (اگر 29 تایی هستند)
                            if expected_dim == 29 and name in ['Ablation_Base', 'Ablation_Lyap']:
                                curr_obs[23:] = 0.0 
                            
                            action, _ = active_model.predict(curr_obs, deterministic=True)
                        
                        if action != last_br: switches += 1
                        last_br = action
                        
                        obs, reward, done, _, info = env.step(action)
                        total_reward += reward
                        last_tp = info.get('throughput', last_tp)
                        
                        if enable_logging and name == 'Proposed':
                            logger.log_chunk(env, action, info, ep)
                    
                    # Calculate QoE
                    qoe = info['total_quality'] - (env.REBUF_PENALTY_BASE * info['total_rebuffer']) - (env.SMOOTH_PENALTY_WEIGHT * info['total_smoothness'])
                    video_duration = env.chunk_idx * 4.0
                    rebuf_ratio = (info['total_rebuffer'] / video_duration) * 100 if video_duration > 0 else 0

                    self.results_detailed.append({
                        'Method': name, 'Video': video_name, 'Episode': ep,
                        'VMAF': info['avg_quality'], 'Rebuffer': rebuf_ratio,
                        'QoE': qoe, 'Switch': switches
                    })
                    
                    if enable_logging and name == 'Proposed':
                        logger.log_episode(env, total_reward, switches, ep)
                
                if enable_logging and name == 'Proposed':
                    logger.save_logs(f'{name}_{video_name}')
                
            print(f"   ✅ {video_name} Done.        ")

    def save_statistics(self):
        if not self.results_detailed: return
        df = pd.DataFrame(self.results_detailed)
        path = PATHS['results'] / 'detailed_stats_proposed_v3_lyapunov.csv'
        df.to_csv(path, index=False)
        print(f"\n✅ Saved comprehensive results to: {path}")
        self.print_summary(df)
        return df

    def print_summary(self, df):
        summary = df.groupby('Method').agg({
            'QoE': ['mean', 'std'], 'VMAF': ['mean', 'std'], 'Rebuffer': ['mean', 'std'], 'Switch': ['mean']
        }).round(2)
        print("\n🏆 OVERALL STATISTICAL SUMMARY (Master Evaluation):")
        print(summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--no-logging', action='store_true')
    args = parser.parse_args()
    
    evaluator = TCSVT_Evaluator()
    methods = evaluator.load_methods()
    if methods:
        evaluator.evaluate(methods, episodes_per_video=args.episodes, enable_logging=not args.no_logging)
        evaluator.save_statistics()