import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from configs.paths import get_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATHS = get_paths()

class TCSVT_Evaluator:
    def __init__(self):
        self.test_trace_dir = PATHS['test_traces']
        self.results = []
        
    def load_methods(self):
        methods = {}
        # Load models (same as before)...
        try:
            path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'final_model'
            methods['Proposed (Lyapunov)'] = PPO.load(str(path))
            print("✓ Proposed model loaded.")
        except: print("⚠ Proposed missing.")

        try:
            path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'final_model'
            methods['Pensieve*'] = PPO.load(str(path))
            print("✓ Pensieve* model loaded.")
        except: print("⚠ Pensieve missing.")
            
        return methods

    def evaluate(self, methods, video_name='bigbuckbunny', episodes=50):
        print(f"\n🔬 Evaluating on Unseen Test Set - Video: {video_name}")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No test traces found!")
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
            
            ep_rewards, ep_vmafs, ep_rebufs, ep_switches = [], [], [], []
            
            for _ in range(episodes):
                obs, info = env.reset()
                done = False
                last_br = 0
                switches = 0
                
                while not done:
                    if 'Random' in name:
                        action = env.action_space.sample()
                    else:
                        if 'Pensieve' in name:
                            obs[10:] = 0.0
                        action, _ = model.predict(obs, deterministic=True)
                    
                    if action != last_br: switches += 1
                    last_br = action
                    
                    obs, reward, done, _, info = env.step(action)
                
                # Metrics Calculation (UPDATED for 0-100 scale)
                # total_quality is now sum of 0-100 scores.
                # QoE = VMAF_SUM - 85 * Rebuf_Time (Using our Env's logic)
                # But for standard reporting, we can use the env's internal reward or recalculate.
                
                # Let's use the sum of rewards for RL comparison
                ep_rewards.append(info['total_quality'] - 85.0 * info['total_rebuffer'])
                
                # VMAF is already 0-100 in avg_quality
                ep_vmafs.append(info['avg_quality']) 
                
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
        if not self.results: return
        df = pd.DataFrame(self.results)
        print("\n🏆 Final Results:")
        print(df.groupby('Method').mean())
        
        df.to_csv(PATHS['results'] / 'tcsvt_generalization_results.csv', index=False)
        self.plot_metric(df, 'Avg VMAF', 'Average VMAF (0-100)', 'vmaf_comparison.png')
        self.plot_metric(df, 'Rebuffering Ratio (%)', 'Rebuffering Ratio (%)', 'rebuffer_comparison.png')

    def plot_metric(self, df, metric, title, filename):
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Method', y=metric, data=df, palette='viridis')
        plt.title(title)
        plt.ylabel(metric)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(PATHS['results'] / filename, dpi=300)

if __name__ == '__main__':
    evaluator = TCSVT_Evaluator()
    methods = evaluator.load_methods()
    if methods:
        evaluator.evaluate(methods, episodes=50)
        evaluator.save_and_plot()