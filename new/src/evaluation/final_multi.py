import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from src.baselines.mpc_vmaf import RobustMPC 
from src.baselines.genie import Genie
from src.baselines.bba import BBA
from configs.paths import get_paths
import numpy as np
import pandas as pd
import argparse

PATHS = get_paths()

class TCSVT_Evaluator:
    def __init__(self):
        self.test_trace_dir = PATHS['test_traces']
        self.results_detailed = [] 

        self.test_videos = [
            'bigbuckbunny',    
            'crowd_run',    
            'parkjoy',       
            'tearsofsteel_short' 
        ]

    def load_methods(self):
        methods = {}
        
        # 1. Proposed
        try:
            path = PATHS['models'] / 'ppo_abr_multi_dynamic_3' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_multi_dynamic_3' / 'final_model'
            methods['Proposed'] = PPO.load(str(path))
            print(f"✓ Loaded Proposed from: {path}")
        except: print("⚠ Proposed missing.")
        
        # 2. Pensieve
        try:
            path = PATHS['models'] / 'pensieve_multi_vmaf_new' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_multi_vmaf_new' / 'final_model'
            methods['Pensieve'] = PPO.load(str(path))
            print(f"✓ Loaded Pensieve from: {path}")
        except: print("⚠ Pensieve missing.")

        # Baselines
        methods['RobustMPC'] = 'mpc_placeholder' 
        methods['Genie'] = 'genie_placeholder'
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
            
        return methods

    def evaluate(self, methods, episodes_per_video=20):
        print(f"\n🔬 Running Statistical Evaluation (N={episodes_per_video} per video)...")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No traces found.")
            return

        for video_name in self.test_videos:
            print(f"\n📹 Testing on Video: {video_name}")
            
            if not (PATHS['content_features'] / f"{video_name}_siti.json").exists():
                print(f"   ⚠ Skipping {video_name}: Data not found.")
                continue

            env = ABREnv(
                video_name=video_name,
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345
            )

            for name, model in methods.items():
                print(f"   > Running {name}...", end='\r')
                
                active_model = model
                if name == 'RobustMPC': active_model = RobustMPC(env)
                elif name == 'Genie': active_model = Genie(env)
                
                for ep in range(episodes_per_video):
                    obs, info = env.reset()
                    done = False
                    last_br = 0
                    switches = 0
                    last_tp = 2000.0
                    trace_tp = env.current_trace['throughput_kbps']
                    
                    while not done:
                        if name == 'RobustMPC':
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(info['buffer_level'], last_tp, cur_vmaf)
                        elif name == 'Genie':
                            action = active_model.select_bitrate(env.chunk_idx, env.buffer_level, trace_tp)
                        elif name == 'BBA':
                            action = active_model.select_bitrate(info['buffer_level'])
                        else:
                            if name == 'Pensieve': obs[10:] = 0.0
                            action, _ = active_model.predict(obs, deterministic=True)
                        
                        if action != last_br: switches += 1
                        last_br = action
                        
                        obs, _, done, _, info = env.step(action)
                        last_tp = info.get('throughput', last_tp)
                    
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
            print(f"   ✓ {video_name} Done.        ")

    def save_statistics(self):
        if not self.results_detailed: return
        
        df = pd.DataFrame(self.results_detailed)
        path = PATHS['results'] / 'detailed_stats_multi_video_3.csv'
        df.to_csv(path, index=False)
        print(f"\n✓ Saved results to: {path}")
        
        self.print_summary(df)
        return df
    
    def load_and_calculate_statistics(self, csv_file):
        path = PATHS['results'] / csv_file
        if not path.exists():
            print(f"❌ File not found: {path}")
            return
        
        print(f"\n📊 Loading from: {csv_file}")
        df = pd.read_csv(path)
        self.print_summary(df)
        return df

    def print_summary(self, df):
        summary = df.groupby('Method').agg({
            'QoE': ['mean', 'std'],
            'VMAF': ['mean', 'std'],
            'Rebuffer': ['mean', 'std']
        }).round(2)
        print("\n🏆 Overall Statistical Summary:")
        print(summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Load existing CSV')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per video')
    args = parser.parse_args()
    
    evaluator = TCSVT_Evaluator()
    
    if args.load:
        evaluator.load_and_calculate_statistics(args.load)
    else:
        methods = evaluator.load_methods()
        if methods:
            evaluator.evaluate(methods, episodes_per_video=args.episodes)
            evaluator.save_statistics()