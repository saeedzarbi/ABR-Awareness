"""
Compare PPO V3 vs V4 to see improvements.
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


def evaluate_model(model, env, num_episodes=30):
    """Evaluate a model."""
    rewards = []
    rebuffers = []
    qualities = []
    switches = []
    bitrates = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        last_action = 0
        ep_switches = 0
        ep_bitrates = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            if action != last_action:
                ep_switches += 1
            last_action = action
            
            ep_bitrates.append(env.BITRATE_LEVELS[action])
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        rebuffers.append(info['total_rebuffer'])
        qualities.append(info['avg_quality'])
        switches.append(ep_switches)
        bitrates.append(np.mean(ep_bitrates))
    
    return {
        'reward': np.mean(rewards),
        'reward_std': np.std(rewards),
        'rebuffer': np.mean(rebuffers),
        'rebuffer_std': np.std(rebuffers),
        'quality': np.mean(qualities),
        'quality_std': np.std(qualities),
        'switches': np.mean(switches),
        'switches_std': np.std(switches),
        'bitrate': np.mean(bitrates),
        'bitrate_std': np.std(bitrates)
    }


def main():
    print("\n" + "="*70)
    print("🔬 V3 vs V4 Comparison")
    print("="*70 + "\n")
    
    # Create environment
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48,
        random_seed=42
    )
    
    # Load models
    print("Loading models...")
    v3_model = PPO.load(str(PATHS['models'] / 'ppo_abr_v3' / 'best_model' / 'best_model'))
    v4_model = PPO.load(str(PATHS['models'] / 'ppo_abr_v4' / 'best_model' / 'best_model'))
    bba = BBA(env.BITRATE_LEVELS)
    print("✓ Models loaded\n")
    
    # Evaluate
    print("Evaluating V3...")
    v3_results = evaluate_model(v3_model, env, 30)
    
    print("Evaluating V4...")
    v4_results = evaluate_model(v4_model, env, 30)
    
    print("Evaluating BBA...")
    bba_results = {
        'reward': 0, 'rebuffer': 0, 'quality': 0,
        'switches': 0, 'bitrate': 0
    }
    for ep in range(30):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        last_action = 0
        ep_switches = 0
        ep_bitrates = []
        
        while not done:
            action = bba.select_bitrate(info['buffer_level'])
            if action != last_action:
                ep_switches += 1
            last_action = action
            ep_bitrates.append(env.BITRATE_LEVELS[action])
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        bba_results['reward'] += ep_reward
        bba_results['rebuffer'] += info['total_rebuffer']
        bba_results['quality'] += info['avg_quality']
        bba_results['switches'] += ep_switches
        bba_results['bitrate'] += np.mean(ep_bitrates)
    
    for key in bba_results:
        bba_results[key] /= 30
    
    # Display results
    print("\n" + "="*70)
    print("Results Comparison")
    print("="*70 + "\n")
    
    print(f"{'Metric':<20} | {'V3':<20} | {'V4':<20} | {'BBA':<20} | {'V4 vs V3':<15}")
    print("-"*105)
    
    metrics = [
        ('Reward', 'reward'),
        ('Rebuffering (s)', 'rebuffer'),
        ('Quality', 'quality'),
        ('Switches', 'switches'),
        ('Avg Bitrate (Kbps)', 'bitrate')
    ]
    
    for label, key in metrics:
        v3_val = v3_results[key]
        v4_val = v4_results[key]
        bba_val = bba_results[key]
        
        if key in ['reward', 'quality', 'bitrate']:
            improvement = ((v4_val - v3_val) / abs(v3_val)) * 100 if v3_val != 0 else 0
            symbol = "↑" if improvement > 0 else "↓"
        else:
            improvement = ((v3_val - v4_val) / abs(v3_val)) * 100 if v3_val != 0 else 0
            symbol = "↑" if improvement > 0 else "↓"
        
        print(f"{label:<20} | {v3_val:^20.2f} | {v4_val:^20.2f} | {bba_val:^20.2f} | {symbol} {abs(improvement):>5.1f}%")
    
    print("\n" + "="*70)
    print("V4 Improvements:")
    print("="*70)
    
    reward_imp = v4_results['reward'] - v3_results['reward']
    quality_imp = v4_results['quality'] - v3_results['quality']
    bitrate_imp = v4_results['bitrate'] - v3_results['bitrate']
    
    print(f"  Reward:    {reward_imp:+.2f}")
    print(f"  Quality:   {quality_imp:+.3f} ({quality_imp/v3_results['quality']*100:+.1f}%)")
    print(f"  Bitrate:   {bitrate_imp:+.0f} Kbps ({bitrate_imp/v3_results['bitrate']*100:+.1f}%)")
    
    if quality_imp > 0.05:
        print("\n  ✓ Significant quality improvement!")
    if bitrate_imp > 200:
        print("  ✓ More aggressive bitrate selection!")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()