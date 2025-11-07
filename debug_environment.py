"""
Debug script to understand the abnormal results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from models.content_aware_env_v2 import ContentAwareEnvV2

print("=" * 80)
print("🔍 Environment Debug")
print("=" * 80)

# Create environment
env = ContentAwareEnvV2(use_real_traces=True)

# Test 10 episodes with different strategies
strategies = [
    ("Always Lowest (300)", [0] * 48),
    ("Always Medium (1850)", [2] * 48),
    ("Always Highest (6000)", [5] * 48),
    ("Random", None),
    ("Conservative", [0, 0, 1, 1, 2, 2] * 8),
    ("Aggressive", [3, 4, 5, 4, 3, 4] * 8)
]

for strategy_name, actions in strategies:
    print(f"\n📊 Testing: {strategy_name}")
    print("-" * 40)
    
    total_rewards = []
    total_rebuffers = []
    total_bitrates = []
    total_vmafs = []
    
    for episode in range(5):
        state = env.reset(split='test')
        
        episode_reward = 0
        episode_rebuffer = 0
        episode_bitrates = []
        episode_vmafs = []
        
        for chunk_idx in range(48):
            # Get action
            if actions is None:  # Random
                action = np.random.randint(0, 6)
            else:
                action = actions[chunk_idx]
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Track metrics
            episode_reward += reward
            episode_rebuffer += info['rebuffer_time']
            episode_bitrates.append(info['bitrate'])
            episode_vmafs.append(info.get('vmaf', 0))
            
            # Debug first chunk of first episode
            if episode == 0 and chunk_idx == 0:
                print(f"  First chunk details:")
                print(f"    Action: {action} ({env.bitrate_levels[action]} kbps)")
                print(f"    Reward: {reward:.2f}")
                print(f"    Rebuffer: {info['rebuffer_time']:.2f}s")
                print(f"    Buffer: {info['buffer']:.2f}s")
                print(f"    VMAF: {info.get('vmaf', 0):.1f}")
                print(f"    Throughput: {info['throughput']:.0f} kbps")
            
            if done:
                break
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_rebuffers.append(episode_rebuffer)
        total_bitrates.append(np.mean(episode_bitrates))
        total_vmafs.append(np.mean(episode_vmafs))
    
    # Summary
    print(f"\n  Results over 5 episodes:")
    print(f"    Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"    Rebuffer: {np.mean(total_rebuffers):.2f}s ± {np.std(total_rebuffers):.2f}")
    print(f"    Bitrate: {np.mean(total_bitrates):.0f} ± {np.std(total_bitrates):.0f}")
    print(f"    VMAF: {np.mean(total_vmafs):.1f} ± {np.std(total_vmafs):.1f}")

print("\n" + "=" * 80)
print("🔍 Checking Network Traces")
print("=" * 80)

# Check trace statistics
from models.trace_loader import TraceLoader
trace_loader = TraceLoader()

print(f"\nTotal traces: {len(trace_loader.all_traces)}")

# Sample some traces
for i in range(3):
    trace = trace_loader.sample_trace('test')
    throughputs = []
    for t in range(0, 20, 1):
        tp = trace.get_throughput(t)
        throughputs.append(tp)
    
    print(f"\nTrace {i+1} sample throughputs (first 20s):")
    print(f"  Mean: {np.mean(throughputs):.0f} kbps")
    print(f"  Std:  {np.std(throughputs):.0f} kbps")
    print(f"  Min:  {np.min(throughputs):.0f} kbps")
    print(f"  Max:  {np.max(throughputs):.0f} kbps")

print("\n" + "=" * 80)
print("🔍 Checking Reward Calculation")
print("=" * 80)

# Test reward function directly
from training_balanced import BalancedRewardFunction, BalancedConfig

config = BalancedConfig()
reward_func = BalancedRewardFunction(config)

test_cases = [
    (80, 0.0, 1850, 1850),   # Good: high VMAF, no rebuffer
    (80, 1.0, 1850, 1850),   # Bad: rebuffer
    (80, 0.0, 300, 6000),    # Bad: huge jump
    (40, 0.0, 300, 300),     # Low quality, no rebuffer
    (90, 0.0, 4300, 4300),   # Excellent quality
]

print("\nReward function tests:")
for vmaf, rebuf, last_br, curr_br in test_cases:
    reward = reward_func.compute_reward(vmaf, rebuf, last_br, curr_br)
    print(f"  VMAF={vmaf}, Rebuf={rebuf:.1f}s, BR={last_br}->{curr_br}: Reward={reward:.2f}")

print("\n" + "=" * 80)
print("✅ Debug Complete")
print("=" * 80)