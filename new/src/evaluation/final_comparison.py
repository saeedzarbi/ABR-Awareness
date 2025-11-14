"""
Final comparison of all methods: PPO, BBA, Random.
"""

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


def evaluate_method(method_name: str, env: ABREnv, model=None, num_episodes: int = 20):
    """
    Evaluate a method (PPO/BBA/Random).
    
    Args:
        method_name: Name of method
        env: ABR environment
        model: Trained model (for PPO) or BBA instance
        num_episodes: Number of episodes
        
    Returns:
        Dictionary with results
    """
    print(f"\n📊 Evaluating {method_name}...")
    
    rewards = []
    rebuffers = []
    qualities = []
    bitrate_switches = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        last_action = 0
        switches = 0
        
        while not done:
            if method_name == 'PPO':
                action, _ = model.predict(obs, deterministic=True)
            elif method_name == 'BBA':
                action = model.select_bitrate(info['buffer_level'])
            else:  # Random
                action = env.action_space.sample()
            
            # Count bitrate switches
            if action != last_action:
                switches += 1
            last_action = action
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        rebuffers.append(info['total_rebuffer'])
        qualities.append(info['avg_quality'])
        bitrate_switches.append(switches)
    
    results = {
        'method': method_name,
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'rebuffer_mean': np.mean(rebuffers),
        'rebuffer_std': np.std(rebuffers),
        'quality_mean': np.mean(qualities),
        'quality_std': np.std(qualities),
        'switches_mean': np.mean(bitrate_switches),
        'switches_std': np.std(bitrate_switches)
    }
    
    print(f"  ✓ {method_name}: Reward={results['reward_mean']:.2f}, "
          f"Rebuffer={results['rebuffer_mean']:.2f}s, "
          f"Quality={results['quality_mean']:.3f}")
    
    return results


def create_comparison_table(results: list):
    """Create and display comparison table."""
    df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("Final Comparison Table")
    print(f"{'='*80}")
    print(f"{'Method':<10} | {'Reward':<15} | {'Rebuffer (s)':<15} | {'Quality':<15} | {'Switches':<10}")
    print(f"{'-'*80}")
    
    for _, row in df.iterrows():
        print(f"{row['method']:<10} | "
              f"{row['reward_mean']:6.2f} ± {row['reward_std']:5.2f} | "
              f"{row['rebuffer_mean']:6.2f} ± {row['rebuffer_std']:5.2f} | "
              f"{row['quality_mean']:5.3f} ± {row['quality_std']:5.3f} | "
              f"{row['switches_mean']:5.1f} ± {row['switches_std']:4.1f}")
    
    print(f"{'='*80}\n")
    
    return df


def plot_comparison(df: pd.DataFrame):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ('reward_mean', 'Average Reward', 'Reward'),
        ('rebuffer_mean', 'Average Rebuffering Time (s)', 'Rebuffering'),
        ('quality_mean', 'Average Video Quality', 'Quality'),
        ('switches_mean', 'Average Bitrate Switches', 'Switches')
    ]
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        x = range(len(df))
        y = df[metric]
        yerr = df[metric.replace('_mean', '_std')]
        
        bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(df['method'])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        # Color bars
        colors = ['green', 'blue', 'red']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    
    # Save figure
    save_path = PATHS['results'] / 'method_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")
    
    return fig


def main():
    """Main comparison script."""
    print("\n" + "="*80)
    print("Final Method Comparison: PPO vs BBA vs Random")
    print("="*80)
    
    num_episodes = 20
    
    # Create environment
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48,
        random_seed=42
    )
    
    all_results = []
    
    # 1. Evaluate PPO
    try:
        model_path = PATHS['models'] / 'ppo_abr' / 'best_model' / 'best_model'
        ppo_model = PPO.load(str(model_path))
        ppo_results = evaluate_method('PPO', env, ppo_model, num_episodes)
        all_results.append(ppo_results)
    except Exception as e:
        print(f"⚠ Could not load PPO model: {e}")
    
    # 2. Evaluate BBA
    bba = BBA(env.BITRATE_LEVELS)
    bba_results = evaluate_method('BBA', env, bba, num_episodes)
    all_results.append(bba_results)
    
    # 3. Evaluate Random
    random_results = evaluate_method('Random', env, None, num_episodes)
    all_results.append(random_results)
    
    # Create comparison table
    df = create_comparison_table(all_results)
    
    # Save to CSV
    csv_path = PATHS['results'] / 'final_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    # Create plots
    try:
        plot_comparison(df)
    except Exception as e:
        print(f"⚠ Could not create plots: {e}")
    
    # Calculate improvements
    if len(all_results) >= 2:
        print("\n" + "="*80)
        print("PPO Improvements over BBA:")
        print("="*80)
        
        ppo = all_results[0]
        bba = all_results[1]
        
        reward_imp = ((ppo['reward_mean'] - bba['reward_mean']) / abs(bba['reward_mean'])) * 100
        rebuffer_imp = ((bba['rebuffer_mean'] - ppo['rebuffer_mean']) / bba['rebuffer_mean']) * 100
        quality_imp = ((ppo['quality_mean'] - bba['quality_mean']) / bba['quality_mean']) * 100
        
        print(f"  Reward improvement:     {reward_imp:+.1f}%")
        print(f"  Rebuffering reduction:  {rebuffer_imp:+.1f}%")
        print(f"  Quality improvement:    {quality_imp:+.1f}%")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()