"""
Test baseline policies on same traces
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_env_v2 import ContentAwareEnvV2
import numpy as np


def test_policy(env, policy_name, policy_fn, num_episodes=10):
    """Test a policy"""
    
    episode_rewards = []
    episode_rebuffers = []
    
    for ep in range(num_episodes):
        state = env.reset(video_id=(ep % 6) + 1, split='test')
        episode_reward = 0
        episode_rebuffer = 0
        
        done = False
        while not done:
            action = policy_fn(state, env)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_rebuffer += info['rebuffer_time']
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(episode_rebuffer)
    
    print(f"\n{policy_name}:")
    print(f"  Avg Reward:     {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg Rebuffering: {np.mean(episode_rebuffers):6.2f}s")
    
    return np.mean(episode_rewards)


def fixed_low(state, env):
    return 0  # Always lowest bitrate


def fixed_mid(state, env):
    return 2  # Always 1850 kbps


def buffer_based(state, env):
    """Simple buffer-based policy"""
    buffer = state['network'][2, -1] * 60.0
    
    if buffer < 5:
        return 0
    elif buffer < 10:
        return 1
    elif buffer < 20:
        return 2
    elif buffer < 30:
        return 3
    else:
        return 4


def throughput_based(state, env):
    """Simple throughput-based policy"""
    past_tp = state['network'][0, :] * 6000.0
    recent_tp = past_tp[past_tp > 0]
    
    if len(recent_tp) > 0:
        avg_tp = np.mean(recent_tp[-3:])
    else:
        avg_tp = 1000
    
    # Use 70% of throughput
    target = avg_tp * 0.7
    
    for i, br in enumerate(env.bitrate_levels):
        if br > target:
            return max(0, i - 1)
    
    return len(env.bitrate_levels) - 1


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Baseline Policies")
    print("=" * 70)
    
    env = ContentAwareEnvV2(use_real_traces=True)
    
    baselines = [
        ("Fixed Low (300 kbps)", fixed_low),
        ("Fixed Mid (1850 kbps)", fixed_mid),
        ("Buffer-Based", buffer_based),
        ("Throughput-Based", throughput_based),
    ]
    
    results = {}
    for name, policy_fn in baselines:
        reward = test_policy(env, name, policy_fn, num_episodes=10)
        results[name] = reward
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    for name, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<25}: {reward:7.2f}")
    
    print("\n" + "=" * 70)
    print("Our trained model achieved: -723.46")
    print("Compare with baselines above")
    print("=" * 70)
