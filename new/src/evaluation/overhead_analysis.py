"""
Evaluation Script for IEEE TCSVT Submission.
Tests Generalization Capability on Unseen Network Traces (Norway Dataset).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from src.baselines.bba import BBA
# from src.baselines.mpc import MPC  # Uncomment if MPC is available
from configs.paths import get_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATHS = get_paths()

class TCSVT_Evaluator:
    def __init__(self):
        # CRITICAL: Point to TEST traces (not used in training)
        self.test_trace_dir = 'data/network_traces/cooked_test_traces'
        self.results = []
        
    def load_methods(self):
        methods = {}
        
        # 1. Proposed Lyapunov-PPO
        try:
            path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'best_model' / 'best_model'
            methods['Proposed (Lyapunov)'] = PPO.load(str(path))
        except:
            print("⚠ Warning: Proposed model not found. Train it first.")

        # 2. Baseline: BBA
        methods['BBA (Robust)'] = BBA(ABREnv.BITRATE_LEVELS)
        
        # 3. Baseline: Pensieve (Load your trained Pensieve model here)
        try:
            path = PATHS['models'] / 'pensieve' / 'best_model' / 'best_model'
            methods['Pensieve'] = PPO.load(str(path))
        except:
            print("⚠ Pensieve model not found.")
            
        return methods

    def evaluate(self, methods, video_name='sample1', episodes=50):
        print(f"\n🔬 Evaluating on Test Set (Generalization Check) - Video: {video_name}")
        
        for name, model in methods.items():
            print(f"   Running {name}...", end='', flush=True)
            
            # Create Fresh Environment with Test Traces
            env = ABREnv(
                video_name=video_name,
                trace_dir=self.test_trace_dir,  # KEY: Testing on Norway traces
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345  # Fixed seed for fair comparison
            )
            
            ep_rewards, ep_vmafs, ep_rebufs, ep_switches = [], [], [], []
            
            for _ in range(episodes):
                obs, info = env.reset()
                done = False
                last_br = 0
                switches = 0
                
                while not done:
                    if 'BBA' in name:
                        action = model.select_bitrate(info['buffer_level'])
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    
                    if action != last_br: switches += 1
                    last_br = action
                    
                    obs, reward, done, _, info = env.step(action)
                
                ep_rewards.append(info['total_quality'] - 4.3 * info['total_rebuffer']) # Standard QoE
                ep_vmafs.append(info['avg_quality'] * 100) # Scale back to 0-100
                ep_rebufs.append(info['total_rebuffer'])
                ep_switches.append(switches)
                
            self.results.append({
                'Method': name,
                'Avg VMAF': np.mean(ep_vmafs),
                'Rebuffering Ratio (%)': np.mean(ep_rebufs) / (48*4) * 100, # approx duration
                'Switch Freq': np.mean(ep_switches),
                'Standard QoE': np.mean(ep_rewards)
            })
            print(" Done.")

    def save_and_plot(self):
        df = pd.DataFrame(self.results)
        print("\n🏆 Final Generalization Results:")
        print(df.groupby('Method').mean())
        
        # Save for LaTeX table
        df.to_csv('results/tcsvt_generalization_results.csv')
        
        # Generate CDF Plot (Mandatory for TCSVT)
        self.plot_cdf(df, 'Avg VMAF', 'VMAF Score', 'vmaf_cdf.png')
        self.plot_cdf(df, 'Standard QoE', 'QoE Score', 'qoe_cdf.png')

    def plot_cdf(self, df, metric, xlabel, filename):
        plt.figure(figsize=(8, 5))
        for method in df['Method'].unique():
            data = np.sort(df[df['Method'] == method][metric])
            yvals = np.arange(len(data)) / float(len(data) - 1)
            plt.plot(data, yvals, label=method, linewidth=2)
            
        plt.xlabel(xlabel)
        plt.ylabel('CDF')
        plt.title(f'CDF of {xlabel} on Unseen Networks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/{filename}', dpi=300)
        print(f"✓ Plot saved: results/{filename}")

if __name__ == '__main__':
    evaluator = TCSVT_Evaluator()
    methods = evaluator.load_methods()
    evaluator.evaluate(methods)
    evaluator.save_and_plot()