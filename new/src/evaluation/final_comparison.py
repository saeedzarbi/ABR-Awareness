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
import matplotlib.pyplot as plt
import seaborn as sns

PATHS = get_paths()

class TCSVT_Evaluator:
    def __init__(self):
        self.test_trace_dir = PATHS['test_traces']
        self.results_detailed = [] # Store every episode
        
    def load_methods(self):
        methods = {}
        
        # 1. Proposed
        try:
            path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'final_model'
            methods['Proposed'] = PPO.load(str(path))
        except: print("⚠ Proposed missing.")
        
        # 2. Baseline: Pensieve*
        try:
            path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'final_model'
            methods['Pensieve'] = PPO.load(str(path))
        except: print("⚠ Pensieve missing.")

        # 3. RobustMPC
        methods['RobustMPC'] = 'mpc_placeholder' 
        
        # 4. Genie
        methods['Genie'] = 'genie_placeholder'
        
        # 5. BBA
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
            
        return methods

    def evaluate(self, methods, video_name='bigbuckbunny', episodes=50):
        print(f"\n🔬 Running Statistical Evaluation (N={episodes})...")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No traces found.")
            return

        for name, model in methods.items():
            print(f"   Running {name}...", end='', flush=True)
            
            env = ABREnv(
                video_name=video_name,
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345
            )
            
            # Init specific models
            active_model = model
            if name == 'RobustMPC': active_model = RobustMPC(env)
            elif name == 'Genie': active_model = Genie(env)
            
            for ep in range(episodes):
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
                    
                    obs, reward, done, _, info = env.step(action)
                    last_tp = info.get('throughput', last_tp)
                
                # Save PER EPISODE metrics
                qoe = info['total_quality'] - (env.REBUF_PENALTY_BASE * info['total_rebuffer']) - (env.SMOOTH_PENALTY_WEIGHT * info['total_smoothness'])
                
                self.results_detailed.append({
                    'Method': name,
                    'Episode': ep,
                    'VMAF': info['avg_quality'],
                    'Rebuffer': (info['total_rebuffer'] / (48*4)) * 100,
                    'QoE': qoe,
                    'Switch': switches
                })
            print(" Done.")

    def save_statistics(self):
        if not self.results_detailed: return
        
        df = pd.DataFrame(self.results_detailed)
        df.to_csv(PATHS['results'] / 'detailed_stats.csv', index=False)
        print(f"\n✓ Detailed statistics saved to: detailed_stats.csv")
        
        # Calculate Mean +/- Std
        summary = df.groupby('Method').agg(['mean', 'std']).round(2)
        print("\n🏆 Statistical Summary:")
        print(summary[['QoE', 'VMAF', 'Rebuffer']])
        
        return df

if __name__ == '__main__':
    evaluator = TCSVT_Evaluator()
    methods = evaluator.load_methods()
    if methods:
        evaluator.evaluate(methods, episodes=50) # Run 50 episodes for stats
        evaluator.save_statistics()