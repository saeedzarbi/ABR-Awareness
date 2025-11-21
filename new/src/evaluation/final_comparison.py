import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
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
        print(f"📂 Loading Test Traces from: {self.test_trace_dir}")
        self.results = []
        
    def load_methods(self):
        methods = {}
        
        # 1. Proposed (Lyapunov)
        try:
            path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'final_model'
            methods['Proposed (Lyapunov)'] = PPO.load(str(path))
            print("✓ Proposed model loaded.")
        except:
            print("⚠ Proposed model not found.")

        # 2. Baseline: BBA
        methods['BBA (Robust)'] = BBA(ABREnv.BITRATE_LEVELS)
        print("✓ BBA baseline loaded.")
        
        # 3. Baseline: Pensieve (Retrained)
        try:
            path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'final_model'
            methods['Pensieve*'] = PPO.load(str(path)) # Asterisk denotes retrained
            print("✓ Pensieve* model loaded.")
        except:
            print("⚠ Pensieve model not found.")
            
        return methods

    # UPDATE: Default video name changed to 'bigbuckbunny'
    def evaluate(self, methods, video_name='bigbuckbunny', episodes=50):
        print(f"\n🔬 Evaluating on Unseen Test Set - Video: {video_name}")
        
        traces = list(self.test_trace_dir.glob("*.json"))
        if not traces:
            print("❌ CRITICAL ERROR: No test traces found!")
            print(f"   Please run 'python organize_data.py' first.")
            return

        print(f"   Found {len(traces)} test traces.")

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
            
            ep_rewards, ep_vmafs, ep_rebufs, ep_switches = [], [], [], []
            
            # Run evaluation loop
            for i in range(episodes):
                obs, info = env.reset()
                done = False
                last_br = 0
                switches = 0
                
                while not done:
                    if 'BBA' in name:
                        action = model.select_bitrate(info['buffer_level'])
                    elif 'Random' in name:
                        action = env.action_space.sample()
                    else:
                        # For Pensieve, we must mask the observation manually here too
                        # to be consistent with training wrapper
                        if 'Pensieve' in name:
                            obs[10:] = 0.0
                            
                        action, _ = model.predict(obs, deterministic=True)
                    
                    if action != last_br: switches += 1
                    last_br = action
                    
                    obs, reward, done, _, info = env.step(action)
                
                # Metrics calculation
                qoe = (info['total_quality'] * 100) - (4.3 * info['total_rebuffer'])
                ep_rewards.append(qoe)
                ep_vmafs.append(info['avg_quality'] * 100)
                ep_rebufs.append(info['total_rebuffer'])
                ep_switches.append(switches)
                
            self.results.append({
                'Method': name,
                'Avg VMAF': np.mean(ep_vmafs),
                'Rebuffering Ratio (%)': (np.mean(ep_rebufs) / (48*4)) * 100,
                'Switch Freq': np.mean(ep_switches),
                'Standard QoE': np.mean(ep_rewards)
            })
            print(" Done.")

    def save_and_plot(self):
        if not self.results:
            print("No results to save.")
            return
            
        df = pd.DataFrame(self.results)
        print("\n🏆 Final Results (Averaged over episodes):")
        print(df.groupby('Method').mean())
        
        # Save CSV
        df.to_csv(PATHS['results'] / 'tcsvt_generalization_results.csv', index=False)
        
        # Generate plots
        self.plot_metric(df, 'Standard QoE', 'QoE Score', 'qoe_comparison.png')
        self.plot_metric(df, 'Avg VMAF', 'Average VMAF', 'vmaf_comparison.png')
        self.plot_metric(df, 'Rebuffering Ratio (%)', 'Rebuffering Ratio (%)', 'rebuffer_comparison.png')

    def plot_metric(self, df, metric, title, filename):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Method', y=metric, data=df, palette='viridis')
        plt.title(f'{title} on Unseen Test Networks')
        plt.ylabel(metric)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(PATHS['results'] / filename, dpi=300)
        print(f"✓ Plot saved: results/{filename}")

if __name__ == '__main__':
    evaluator = TCSVT_Evaluator()
    methods = evaluator.load_methods()
    if methods:
        evaluator.evaluate(methods, episodes=50)
        evaluator.save_and_plot()
    else:
        print("❌ No models loaded. Please train models first.")