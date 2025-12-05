"""Quick V4 action distribution analysis."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from configs.paths import get_paths
import numpy as np

PATHS = get_paths()

# Load V4 and V3 for comparison
print("\n" + "="*70)
print("V3 vs V4 Action Distribution Comparison")
print("="*70 + "\n")

for version in ['v3', 'v4']:
    print(f"\n{'='*70}")
    print(f"PPO V{version.upper()}")
    print(f"{'='*70}\n")
    
    model = PPO.load(str(PATHS['models'] / f'ppo_abr_{version}' / 'best_model' / 'best_model'))
    
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48
    )
    
    all_actions = []
    all_buffers = []
    
    for ep in range(100):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action)
            all_buffers.append(info.get('buffer_level', 0))
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
    actions = np.array(all_actions)
    buffers = np.array(all_buffers)
    
    print("Action Distribution:")
    print("-" * 70)
    for i, br in enumerate(env.BITRATE_LEVELS):
        count = np.sum(actions == i)
        pct = (count / len(actions)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {br:4d} Kbps: {count:5d} ({pct:5.1f}%) {bar}")
    
    avg_bitrate = np.mean([env.BITRATE_LEVELS[a] for a in actions])
    print(f"\n  Average bitrate: {avg_bitrate:.0f} Kbps")
    
    # By buffer
    print("\n" + "-"*70)
    print("By Buffer Level:")
    print("-"*70)
    for low, high, label in [(0,5,"Low"), (5,10,"Med"), (10,15,"Good"), (15,30,"High")]:
        mask = (buffers >= low) & (buffers < high)
        if np.sum(mask) > 0:
            avg_br = np.mean([env.BITRATE_LEVELS[a] for a in actions[mask]])
            print(f"  {label:6s} ({low:2d}-{high:2d}s): {np.sum(mask):5d} → {avg_br:6.0f} Kbps")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("V4 should have higher bitrate and more aggressive in high buffer")
print("="*70 + "\n")
