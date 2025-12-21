import sys
from pathlib import Path
import os
import json
import urllib.request
import random
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_multi_env import ABREnv
from src.baselines.mpc_vmaf import RobustMPC 
from src.baselines.genie import Genie
from src.baselines.bba import BBA
from configs.paths import get_paths
import numpy as np
import pandas as pd
import argparse

PATHS = get_paths()

# Slack webhook URL
SLACK_WEBHOOK = os.getenv('SLACK_WEBHOOK_URL', '')

# ============================================================================
# New Baseline: Simulated Fugu (Modern Prediction-Based MPC)
# ============================================================================
class SimulatedFugu(RobustMPC):
    """
    Simulates a modern 'Learned Throughput Predictor' (like Fugu or Oboe).
    Instead of using Harmonic Mean (like RobustMPC), it 'peeks' at the real future throughput
    but adds some realistic noise/error (e.g., +/- 10% error).
    This represents the upper-bound performance of predictor-based methods.
    """
    def __init__(self, env, noise_level=0.10):
        super().__init__(env)
        self.noise_level = noise_level # 10% prediction error
    
    def select_bitrate(self, buffer_level, last_throughput, vmaf):
        # Override the throughput prediction logic
        # 1. Get True Future Throughput (Oracle)
        # We estimate it by looking at the current trace pointer in env
        true_tp = self.env.current_trace['throughput_kbps'][self.env.current_trace_idx]
        
        # 2. Add realistic noise (Simulating ML prediction error)
        noise = random.uniform(1.0 - self.noise_level, 1.0 + self.noise_level)
        predicted_tp = true_tp * noise
        
        # 3. Use RobustMPC solver with this "Smart Prediction"
        # We temporarily hijack the harmonic mean logic by passing this predicted_tp
        # implicitly, but since RobustMPC.select_bitrate calculates its own harmonic mean,
        # we need to re-implement the core MPC loop here or hack it.
        # EASIER WAY: Let's re-implement the MPC planning loop with our predicted_tp.
        
        best_bitrate = 0
        max_qoe = -float('inf')
        
        # Simple MPC Horizon (5 steps)
        HORIZON = 5
        
        # Check all possible next bitrates
        for br_idx in range(len(self.env.BITRATE_LEVELS)):
            # Simulate trajectory
            tot_qoe = 0
            curr_buffer = buffer_level
            
            # For the first step, use the chosen bitrate
            br = self.env.BITRATE_LEVELS[br_idx]
            size = br * 1000 * 4.0 # 4s chunk
            dl_time = size / (predicted_tp * 1000.0)
            rebuf = max(0, dl_time - curr_buffer)
            curr_buffer = max(0, curr_buffer - dl_time) + 4.0
            
            # VMAF approximation (linear map for speed)
            # This is a simplified MPC for Fugu simulation
            vmaf_est = self.env.vmaf_scores.get(br, 50)
            
            qoe = vmaf_est - 4.3 * rebuf
            tot_qoe += qoe
            
            # For future steps in horizon, assume we pick the same bitrate (simplified)
            # or use a greedy approach. Let's assume we stick to predicted capacity.
            for _ in range(HORIZON - 1):
                # Assume throughput stays similar (Fugu logic)
                dl_time_fut = size / (predicted_tp * 1000.0)
                rebuf_fut = max(0, dl_time_fut - curr_buffer)
                curr_buffer = max(0, curr_buffer - dl_time_fut) + 4.0
                tot_qoe += (vmaf_est - 4.3 * rebuf_fut)
            
            if tot_qoe > max_qoe:
                max_qoe = tot_qoe
                best_bitrate = br_idx
                
        return best_bitrate

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
        }
        self.chunk_logs.append(chunk_data)
    
    def log_episode(self, env, total_reward, switches, episode_num):
        rebuffer_rate = (env.total_rebuffer / (env.chunk_idx * 4.0)) * 100 if env.chunk_idx > 0 else 0
        episode_data = {
            'episode': episode_num,
            'video': env.current_video_name,
            'trace_idx': env.current_trace_idx,
            'total_reward': float(total_reward),
            'avg_vmaf': float(env.total_quality / env.chunk_idx),
            'total_rebuffer': float(env.total_rebuffer),
            'rebuffer_rate': float(rebuffer_rate),
            'total_smooth': float(env.total_smooth),
            'switches': int(switches),
        }
        self.episode_logs.append(episode_data)
    
    def save_logs(self, method_name='method'):
        if self.chunk_logs:
            pd.DataFrame(self.chunk_logs).to_csv(self.log_dir / f'{method_name}_chunks.csv', index=False)
        if self.episode_logs:
            pd.DataFrame(self.episode_logs).to_csv(self.log_dir / f'{method_name}_episodes.csv', index=False)
    
    def analyze_logs(self, method_name='method'):
        if not self.episode_logs: return
        df = pd.DataFrame(self.episode_logs)
        print(f"\n📊 {method_name} - Performance Analysis:")
        print("="*60)
        print(f"   Avg VMAF:      {df['avg_vmaf'].mean():.2f}")
        print(f"   Avg Rebuffer:  {df['rebuffer_rate'].mean():.2f}%")
        print(f"   Avg QoE:       {df['total_reward'].mean():.1f}")
        print(f"   Avg Switches:  {df['switches'].mean():.1f}")
        print("="*60)

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
        
        # 1. Proposed (V22)
        try:
            path = PATHS['models'] / 'ppo_abr_multi_dynamic_22' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_multi_dynamic_22' / 'final_model'
            methods['Proposed'] = PPO.load(str(path))
            print(f"✅ Loaded Proposed from: {path}")
        except: print("⚠️ Proposed missing")
        
        # 2. Pensieve
        try:
            path = PATHS['models'] / 'pensieve_multi_vmaf_new_14' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_multi_vmaf_new_16' / 'final_model'
            methods['Pensieve'] = PPO.load(str(path))
            print(f"✅ Loaded Pensieve from: {path}")
        except: print("⚠️ Pensieve missing")

        # 3. Modern Baselines
        methods['RobustMPC'] = 'mpc_placeholder' 
        methods['Genie'] = 'genie_placeholder'
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
        methods['Fugu (Sim)'] = 'fugu_placeholder' # <--- NEW
            
        return methods

    def evaluate(self, methods, episodes_per_video=20, enable_logging=True):
        print(f"\n🔬 Running Final Evaluation (N={episodes_per_video})...")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No traces found.")
            return

        for video_idx, video_name in enumerate(self.test_videos, 1):
            print(f"\n📹 Testing on Video: {video_name} ({video_idx}/{len(self.test_videos)})")
            
            if not (PATHS['content_features'] / f"{video_name}_siti.json").exists(): continue

            env = ABREnv(
                video_names=[video_name],
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345
            )

            for name, model in methods.items():
                print(f"   > Running {name}...", end='\r')
                
                if enable_logging: logger = EvaluationLogger(PATHS['logs'] / 'evaluation_final')
                
                active_model = model
                if name == 'RobustMPC': active_model = RobustMPC(env)
                elif name == 'Genie': active_model = Genie(env)
                elif name == 'Fugu (Sim)': active_model = SimulatedFugu(env, noise_level=0.1) # New Fugu
                
                for ep in range(episodes_per_video):
                    obs, info = env.reset(seed=ep)
                    done = False
                    last_br = 0
                    switches = 0
                    last_tp = 2000.0
                    trace_tp = env.current_trace['throughput_kbps']
                    total_reward = 0
                    
                    while not done:
                        if name == 'RobustMPC':
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(info['buffer_level'], last_tp, cur_vmaf)
                        elif name == 'Genie':
                            action = active_model.select_bitrate(env.chunk_idx, env.buffer_level, trace_tp)
                        elif name == 'Fugu (Sim)':
                            # Fugu Logic
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(info['buffer_level'], last_tp, cur_vmaf)
                        elif name == 'BBA':
                            action = active_model.select_bitrate(info['buffer_level'])
                        else:
                            # RL Models
                            try: expected_dim = active_model.observation_space.shape[0]
                            except: expected_dim = 23
                            
                            curr_obs = obs[:expected_dim].copy() if obs.shape[0] > expected_dim else obs.copy()
                            if name == 'Pensieve' and expected_dim >= 23: curr_obs[10:] = 0.0 
                                
                            action, _ = active_model.predict(curr_obs, deterministic=True)
                        
                        if action != last_br: switches += 1
                        last_br = action
                        
                        obs, reward, done, _, info = env.step(action)
                        total_reward += reward
                        last_tp = info.get('throughput', last_tp)
                        
                        if enable_logging and name == 'Proposed':
                            logger.log_chunk(env, action, info, ep)
                    
                    # QoE Calc
                    qoe = info['total_quality'] - (env.REBUF_PENALTY_BASE * info['total_rebuffer']) - (env.SMOOTH_PENALTY_WEIGHT * info['total_smoothness'])
                    video_duration = env.chunk_idx * 4.0
                    rebuf_ratio = (info['total_rebuffer'] / video_duration) * 100 if video_duration > 0 else 0

                    self.results_detailed.append({
                        'Method': name,
                        'Video': video_name, 
                        'Episode': ep,
                        'VMAF': info['avg_quality'],
                        'Rebuffer': rebuf_ratio,
                        'QoE': qoe,
                        'Switch': switches
                    })
                    
                    if enable_logging and name == 'Proposed':
                        logger.log_episode(env, total_reward, switches, ep)
                
                if enable_logging and name == 'Proposed':
                    logger.save_logs(f'{name}_{video_name}')
                    logger.analyze_logs(f'{name}_{video_name}')
                
            print(f"   ✅ {video_name} Done.        ")

    def save_statistics(self):
        if not self.results_detailed: return
        df = pd.DataFrame(self.results_detailed)
        path = PATHS['results'] / 'detailed_stats_multi_video_final.csv'
        df.to_csv(path, index=False)
        print(f"\n✅ Saved results to: {path}")
        self.print_summary(df)
        return df
    
    def print_summary(self, df):
        summary = df.groupby('Method').agg({
            'QoE': ['mean', 'std'], 'VMAF': ['mean', 'std'], 'Rebuffer': ['mean', 'std']
        }).round(2)
        print("\n🏆 Overall Statistical Summary:")
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