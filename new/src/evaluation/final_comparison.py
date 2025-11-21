import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from src.baselines.mpc_vmaf import RobustMPC  # <--- NEW IMPORT
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
        
        # 1. Proposed (Lyapunov)
        try:
            path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'final_model'
            methods['Proposed (Lyapunov)'] = PPO.load(str(path))
            print("✓ Proposed model loaded.")
        except: print("⚠ Proposed missing.")
        
        # 2. Baseline: Pensieve*
        try:
            path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_retrained_vmaf' / 'final_model'
            methods['Pensieve*'] = PPO.load(str(path))
            print("✓ Pensieve* model loaded.")
        except: print("⚠ Pensieve missing.")

        # 3. Baseline: RobustMPC (VMAF-Aware)
        # MPC needs the environment instance to initialize, we'll init it inside the loop
        methods['RobustMPC'] = 'mpc_placeholder' 
        print("✓ RobustMPC baseline ready.")
            
        return methods

    def evaluate(self, methods, video_name='bigbuckbunny', episodes=50):
        print(f"\n🔬 Evaluating on Unseen Test Set - Video: {video_name}")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No test traces found!")
            return

        # Create a shared environment for initialization if needed
        temp_env = ABREnv(video_name=video_name, trace_dir=str(self.test_trace_dir))

        for name, model in methods.items():
            print(f"   Running {name}...", end='', flush=True)
            
            # Fresh env for each method
            env = ABREnv(
                video_name=video_name,
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345
            )
            
            # Initialize MPC if needed
            active_model = model
            if name == 'RobustMPC':
                active_model = RobustMPC(env)
            
            ep_rewards, ep_vmafs, ep_rebufs, ep_switches = [], [], [], []
            
            for _ in range(episodes):
                obs, info = env.reset()
                done = False
                last_br = 0
                switches = 0
                last_throughput = 2000.0 # Initial guess for MPC
                
                while not done:
                    if name == 'RobustMPC':
                        # MPC logic
                        action = active_model.select_bitrate(
                            info['buffer_level'], 
                            last_throughput,
                            env.last_quality_metric # Passing current VMAF knowledge
                        )
                    elif 'Random' in name:
                        action = env.action_space.sample()
                    else:
                        # RL Agents
                        if 'Pensieve' in name:
                            obs[10:] = 0.0 # Mask content features
                        action, _ = active_model.predict(obs, deterministic=True)
                    
                    if action != last_br: switches += 1
                    last_br = action
                    
                    obs, reward, done, _, info = env.step(action)
                    
                    # Update throughput for MPC
                    last_throughput = info.get('throughput', last_throughput)
                
                # Calculate metrics
                # Standard QoE = VMAF - 50*Rebuf - 0.2*Smooth (Using our updated weights)
                qoe = info['total_quality'] \
                      - (env.REBUF_PENALTY_BASE * info['total_rebuffer']) \
                      - (env.SMOOTH_PENALTY_WEIGHT * info['total_smoothness'])
                
                ep_rewards.append(qoe)
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
        
        # Colors for plots
        colors = sns.color_palette("viridis", n_colors=len(df['Method'].unique()))
        
        self.plot_metric(df, 'Standard QoE', 'QoE Score', 'qoe_comparison.png', colors)
        self.plot_metric(df, 'Avg VMAF', 'Average VMAF (0-100)', 'vmaf_comparison.png', colors)
        self.plot_metric(df, 'Rebuffering Ratio (%)', 'Rebuffering Ratio (%)', 'rebuffer_comparison.png', colors)

    def plot_metric(self, df, metric, title, filename, colors):
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Method', y=metric, data=df, palette=colors)
        plt.title(title, fontweight='bold')
        plt.ylabel(metric)
        plt.xticks(rotation=15)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(PATHS['results'] / filename, dpi=300)

if __name__ == '__main__':
    evaluator = TCSVT_Evaluator()
    methods = evaluator.load_methods()
    if methods:
        evaluator.evaluate(methods, episodes=50)
        evaluator.save_and_plot()