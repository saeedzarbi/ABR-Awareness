"""
Test if environment is solvable with simple strategies
"""
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
import numpy as np

print("="*80)
print("TESTING SIMPLE STRATEGIES")
print("="*80)

# Create environment
fcc_loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env = ContentAwareEnvFCC(
    fcc_trace_loader=fcc_loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='val'
)

def test_strategy(name, action_fn, n_episodes=5):
    """Test a strategy"""
    print(f"\n{name}:")
    print("-" * 80)
    
    all_rewards = []
    all_rebuffers = []
    all_vmafs = []
    all_bitrates = []
    
    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        ep_rebuffer = 0
        ep_vmafs = []
        ep_bitrates = []
        done = False
        step = 0
        
        while not done:
            action = action_fn(state, step, env)
            state, reward, done, info = env.step(action)
            
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_vmafs.append(info['vmaf'])
            ep_bitrates.append(info['bitrate'])
            step += 1
        
        all_rewards.append(ep_reward)
        all_rebuffers.append(ep_rebuffer)
        all_vmafs.append(np.mean(ep_vmafs))
        all_bitrates.append(np.mean(ep_bitrates))
    
    print(f"  Reward:    {np.mean(all_rewards):+7.2f} ± {np.std(all_rewards):5.2f}")
    print(f"  Rebuffer:  {np.mean(all_rebuffers):7.2f}s ± {np.std(all_rebuffers):5.2f}s")
    print(f"  VMAF:      {np.mean(all_vmafs):7.1f} ± {np.std(all_vmafs):5.1f}")
    print(f"  Bitrate:   {np.mean(all_bitrates):7.0f} ± {np.std(all_bitrates):5.0f} kbps")
    
    return {
        'reward': np.mean(all_rewards),
        'rebuffer': np.mean(all_rebuffers),
        'vmaf': np.mean(all_vmafs),
        'bitrate': np.mean(all_bitrates)
    }

# Strategy 1: Always lowest
def always_low(state, step, env):
    return 0

# Strategy 2: Always medium
def always_medium(state, step, env):
    return 2

# Strategy 3: Always high
def always_high(state, step, env):
    return 4

# Strategy 4: Buffer-based (BBA-like)
def buffer_based(state, step, env):
    buffer = env.buffer
    if buffer < 5:
        return 0  # Low
    elif buffer < 15:
        return 1  # Medium-low
    elif buffer < 25:
        return 2  # Medium
    elif buffer < 35:
        return 3  # Medium-high
    else:
        return 4  # High

# Strategy 5: Throughput-based
def throughput_based(state, step, env):
    if len(env.past_throughput) == 0:
        return 1
    
    recent_tp = np.mean(env.past_throughput[-3:]) if len(env.past_throughput) >= 3 else env.past_throughput[-1]
    
    # Map throughput to bitrate
    if recent_tp < 500:
        return 0
    elif recent_tp < 1000:
        return 1
    elif recent_tp < 2000:
        return 2
    elif recent_tp < 3500:
        return 3
    else:
        return 4

# Strategy 6: Hybrid (buffer + throughput)
def hybrid(state, step, env):
    buffer = env.buffer
    
    if len(env.past_throughput) == 0:
        return 1
    
    recent_tp = np.mean(env.past_throughput[-3:]) if len(env.past_throughput) >= 3 else env.past_throughput[-1]
    
    # Conservative if buffer low
    if buffer < 8:
        return min(1, int(recent_tp / 1000))
    
    # Use throughput
    if recent_tp < 600:
        return 0
    elif recent_tp < 1200:
        return 1
    elif recent_tp < 2200:
        return 2
    elif recent_tp < 3500:
        return 3
    else:
        return 4

print("\nTesting strategies on validation set:")
print("="*80)

results = {}
results['always_low'] = test_strategy("1. Always Lowest (300 kbps)", always_low)
results['always_medium'] = test_strategy("2. Always Medium (1850 kbps)", always_medium)
results['always_high'] = test_strategy("3. Always High (4300 kbps)", always_high)
results['buffer_based'] = test_strategy("4. Buffer-Based (BBA-like)", buffer_based)
results['throughput_based'] = test_strategy("5. Throughput-Based", throughput_based)
results['hybrid'] = test_strategy("6. Hybrid (Buffer + Throughput)", hybrid)

print("\n" + "="*80)
print("SUMMARY - Best Strategies:")
print("="*80)

# Find best by reward
best_reward = max(results.items(), key=lambda x: x[1]['reward'])
print(f"\nBest Reward: {best_reward[0]}")
print(f"  → {best_reward[1]['reward']:+.2f} reward, "
      f"{best_reward[1]['rebuffer']:.2f}s rebuffer, "
      f"VMAF {best_reward[1]['vmaf']:.1f}")

# Find best balanced
best_balanced = max(results.items(), key=lambda x: x[1]['reward'] + x[1]['vmaf']*0.5 - x[1]['rebuffer']*2)
print(f"\nBest Balanced: {best_balanced[0]}")
print(f"  → {best_balanced[1]['reward']:+.2f} reward, "
      f"{best_balanced[1]['rebuffer']:.2f}s rebuffer, "
      f"VMAF {best_balanced[1]['vmaf']:.1f}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if best_reward[1]['reward'] < 50:
    print("❌ Even simple strategies get low rewards (<50)")
    print("   → Problem: Reward function or environment is broken")
elif best_reward[1]['vmaf'] < 50:
    print("❌ All strategies have low VMAF (<50)")
    print("   → Problem: VMAF predictions or quality calculation")
elif best_reward[1]['rebuffer'] > 10:
    print("❌ All strategies have high rebuffering (>10s)")
    print("   → Problem: Network traces too difficult or download simulation")
else:
    print("✅ Simple strategies work - PPO should be able to learn")
    print(f"   → Target to beat: {best_reward[1]['reward']:+.2f} reward")

print("="*80)