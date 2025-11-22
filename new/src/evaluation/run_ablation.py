"""
Ablation Study for IEEE TCSVT.
Evaluates the contribution of each component by disabling it at inference time.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent.parent))
from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

def run_ablation():
    print("\n🔍 Running Ablation Study...")
    
    # Load Best Model
    try:
        path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'best_model' / 'best_model'
        if not path.with_suffix('.zip').exists():
             path = PATHS['models'] / 'ppo_abr_v4_lyapunov' / 'final_model'
        model = PPO.load(str(path))
    except:
        print("❌ Model not found.")
        return

    scenarios = ['Full Model (Proposed)', 'w/o Content Awareness', 'w/o VMAF Lookahead']
    results = []
    
    # Use Test Traces
    trace_dir = str(PATHS['test_traces'])
    
    for scenario in scenarios:
        print(f"   Testing: {scenario}...", end='', flush=True)
        
        env = ABREnv(
            video_name='bigbuckbunny',
            trace_dir=trace_dir,
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=48,
            random_seed=12345
        )
        
        ep_rewards = []
        
        for _ in range(30): # 30 episodes
            obs, info = env.reset()
            done = False
            
            while not done:
                # --- ABLATION LOGIC ---
                modified_obs = obs.copy()
                
                if scenario == 'w/o Content Awareness':
                    # Mask SI/TI (indices 10, 11)
                    modified_obs[10:12] = 0.0
                
                elif scenario == 'w/o VMAF Lookahead':
                    # Mask VMAF Predictions (indices 12 to 17)
                    modified_obs[12:] = 0.0
                
                # Predict with modified observation
                action, _ = model.predict(modified_obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
            
            # Standard QoE
            qoe = info['total_quality'] - (65.0 * info['total_rebuffer']) - (0.1 * info['total_smoothness'])
            ep_rewards.append(qoe)
            
        results.append({
            'Variant': scenario,
            'Avg QoE': np.mean(ep_rewards)
        })
        print(f" QoE: {np.mean(ep_rewards):.2f}")

    # Plotting
    df = pd.DataFrame(results)
    plt.figure(figsize=(7, 5))
    
    # Calculate drop percentage
    full_score = df[df['Variant']=='Full Model (Proposed)']['Avg QoE'].values[0]
    df['Drop'] = full_score - df['Avg QoE']
    
    sns.barplot(x='Variant', y='Avg QoE', data=df, palette='magma')
    plt.title("Ablation Study: Contribution of Components")
    plt.ylabel("Average QoE")
    plt.ylim(1000, full_score * 1.1)
    plt.tight_layout()
    
    plt.savefig(PATHS['results'] / 'ablation_study_plot.png', dpi=300)
    print(f"\n✓ Ablation plot saved to: results/ablation_study_plot.png")

if __name__ == '__main__':
    run_ablation()