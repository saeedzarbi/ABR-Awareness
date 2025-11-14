"""
Compare PPO V1 vs V2 vs BBA vs Random.
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

PATHS = get_paths()


def evaluate_model(name, env, model, num_episodes=20):
    """Evaluate a model."""
    print(f"\n📊 Evaluating {name}...")
    
    rewards, rebuffers, qualities, switches = [], [], [], []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        last_action = 0
        ep_switches = 0
        
        while not done:
            if name == 'BBA':
                action = model.select_bitrate(info['buffer_level'])
            elif name == 'Random':
                action = env.action_space.sample()
            else:  # PPO
                action, _ = model.predict(obs, deterministic=True)
            
            if action != last_action:
                ep_switches += 1
            last_action = action
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        rebuffers.append(info['total_rebuffer'])
        qualities.append(info['avg_quality'])
        switches.append(ep_switches)
        
        if (ep + 1) % 5 == 0:
            print(f"  Progress: {ep+1}/{num_episodes} episodes")
    
    results = {
        'name': name,
        'reward': np.mean(rewards),
        'reward_std': np.std(rewards),
        'rebuffer': np.mean(rebuffers),
        'rebuffer_std': np.std(rebuffers),
        'quality': np.mean(qualities),
        'quality_std': np.std(qualities),
        'switches': np.mean(switches),
        'switches_std': np.std(switches)
    }
    
    print(f"  ✓ Reward: {results['reward']:.2f} ± {results['reward_std']:.2f}")
    print(f"    Rebuffer: {results['rebuffer']:.2f}s ± {results['rebuffer_std']:.2f}s")
    print(f"    Quality: {results['quality']:.3f} ± {results['quality_std']:.3f}")
    print(f"    Switches: {results['switches']:.1f} ± {results['switches_std']:.1f}")
    
    return results


def main():
    print("\n" + "="*70)
    print("🔬 Final Comparison: PPO V1 vs V2 vs BBA vs Random")
    print("="*70)
    
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
    
    # PPO V2 (latest)
    print("\n" + "-"*70)
    try:
        v2_path = PATHS['models'] / 'ppo_abr_v2' / 'best_model' / 'best_model'
        if not v2_path.with_suffix('.zip').exists():
            v2_path = PATHS['models'] / 'ppo_abr_v2' / 'final_model'
        
        print(f"Loading V2 from: {v2_path}")
        v2_model = PPO.load(str(v2_path))
        all_results.append(evaluate_model('PPO_V2', env, v2_model, 20))
    except Exception as e:
        print(f"⚠ PPO V2 not found: {e}")
    
    # PPO V1 (original)
    print("\n" + "-"*70)
    try:
        v1_path = PATHS['models'] / 'ppo_abr' / 'best_model' / 'best_model'
        if not v1_path.with_suffix('.zip').exists():
            v1_path = PATHS['models'] / 'ppo_abr' / 'final_model'
        
        print(f"Loading V1 from: {v1_path}")
        v1_model = PPO.load(str(v1_path))
        all_results.append(evaluate_model('PPO_V1', env, v1_model, 20))
    except Exception as e:
        print(f"⚠ PPO V1 not found: {e}")
    
    # BBA baseline
    print("\n" + "-"*70)
    bba = BBA(env.BITRATE_LEVELS)
    all_results.append(evaluate_model('BBA', env, bba, 20))
    
    # Random baseline
    print("\n" + "-"*70)
    all_results.append(evaluate_model('Random', env, None, 20))
    
    # Create comparison table
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"{'Method':<10} | {'Reward':<18} | {'Rebuffer (s)':<18} | {'Quality':<15} | {'Switches':<12}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['name']:<10} | "
              f"{row['reward']:7.2f} ± {row['reward_std']:6.2f} | "
              f"{row['rebuffer']:7.2f} ± {row['rebuffer_std']:6.2f} | "
              f"{row['quality']:6.3f} ± {row['quality_std']:5.3f} | "
              f"{row['switches']:5.1f} ± {row['switches_std']:4.1f}")
    
    print("="*80 + "\n")
    
    # Analysis
    if len(all_results) >= 3:
        # Find indices
        v2_idx = next((i for i, r in enumerate(all_results) if r['name'] == 'PPO_V2'), None)
        v1_idx = next((i for i, r in enumerate(all_results) if r['name'] == 'PPO_V1'), None)
        bba_idx = next((i for i, r in enumerate(all_results) if r['name'] == 'BBA'), None)
        
        if v2_idx is not None and bba_idx is not None:
            v2 = all_results[v2_idx]
            bba = all_results[bba_idx]
            
            print("="*70)
            print("PPO V2 vs BBA:")
            print("="*70)
            reward_diff = v2['reward'] - bba['reward']
            rebuffer_diff = v2['rebuffer'] - bba['rebuffer']
            quality_diff = v2['quality'] - bba['quality']
            
            print(f"  Reward:    {reward_diff:+7.2f}  {'✓ Better' if reward_diff > 0 else '✗ Worse'}")
            print(f"  Rebuffer:  {rebuffer_diff:+7.2f}s {'✗ Worse' if rebuffer_diff > 0 else '✓ Better'}")
            print(f"  Quality:   {quality_diff:+7.3f}  {'✓ Better' if quality_diff > 0 else '✗ Worse'}")
            print(f"  Switches:  {v2['switches'] - bba['switches']:+7.1f}")
            print()
        
        if v2_idx is not None and v1_idx is not None:
            v2 = all_results[v2_idx]
            v1 = all_results[v1_idx]
            
            print("="*70)
            print("PPO V2 vs V1 (Improvement):")
            print("="*70)
            reward_imp = v2['reward'] - v1['reward']
            rebuffer_imp = v1['rebuffer'] - v2['rebuffer']
            quality_imp = v2['quality'] - v1['quality']
            
            print(f"  Reward:    {reward_imp:+7.2f}  {'✓ Improved' if reward_imp > 0 else '✗ Degraded'}")
            print(f"  Rebuffer:  {rebuffer_imp:+7.2f}s {'✓ Reduced' if rebuffer_imp > 0 else '✗ Increased'}")
            print(f"  Quality:   {quality_imp:+7.3f}  {'✓ Improved' if quality_imp > 0 else '✗ Degraded'}")
            print(f"  Switches:  {v2['switches'] - v1['switches']:+7.1f}")
            print()
    
    # Save results
    csv_path = PATHS['results'] / 'version_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}\n")


if __name__ == '__main__':
    main()