# فایل جدید: src/evaluation/action_analysis.py

"""Analyze agent's bitrate selection behavior."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from configs.paths import get_paths
import numpy as np
import matplotlib.pyplot as plt

PATHS = get_paths()


def analyze_actions(model_path, num_episodes=10):
    """Analyze what bitrates the agent selects."""
    
    model = PPO.load(model_path)
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48,
        random_seed=42
    )
    
    all_actions = []
    all_buffers = []
    all_throughputs = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action)
            all_buffers.append(info.get('buffer_level', 0))
            
            obs, reward, terminated, truncated, info = env.step(action)
            all_throughputs.append(info.get('throughput', 0))
            done = terminated or truncated
    
    # Analysis
    actions_array = np.array(all_actions)
    bitrates = env.BITRATE_LEVELS
    
    print("\n" + "="*60)
    print("Action Distribution:")
    print("="*60)
    
    for i, br in enumerate(bitrates):
        count = np.sum(actions_array == i)
        pct = count / len(actions_array) * 100
        print(f"  {br:4d} Kbps: {count:4d} times ({pct:5.1f}%)")
    
    print(f"\n  Most selected: {bitrates[np.argmax(np.bincount(actions_array))]} Kbps")
    print(f"  Average bitrate: {np.mean([bitrates[a] for a in actions_array]):.0f} Kbps")
    print("="*60 + "\n")


if __name__ == '__main__':
    v2_path = PATHS['models'] / 'ppo_abr_v2' / 'best_model' / 'best_model'
    analyze_actions(str(v2_path), num_episodes=20)