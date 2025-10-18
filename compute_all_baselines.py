"""
محاسبه تمام baseline ها
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("📊 Computing ALL Baselines")
print("=" * 80)
print()

# Load environment
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

mode = 'val'  # همون set که مدل رو تست کردیم

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode=mode
)

print(f"Testing on: {mode} set ({len(loader.val_traces)} traces)")
print()

n_episodes = 50
bitrates = [300, 750, 1850, 2850, 4300, 6000]

# ══════════════════════════════════════════════════════════
# Baseline 1: Fixed Bitrate (Low)
# ══════════════════════════════════════════════════════════

print("🔵 Baseline 1: Fixed Low (300 kbps)")
print("-" * 80)

rewards = []
for ep in range(n_episodes):
    state = env.reset()
    ep_reward = 0
    done = False
    
    while not done:
        action = 0  # همیشه 300 kbps
        state, reward, done, info = env.step(action)
        ep_reward += reward
    
    rewards.append(ep_reward)

print(f"Result: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
fixed_low = np.mean(rewards)
print()

# ══════════════════════════════════════════════════════════
# Baseline 2: Fixed Bitrate (Medium)
# ══════════════════════════════════════════════════════════

print("🟡 Baseline 2: Fixed Medium (1850 kbps)")
print("-" * 80)

rewards = []
for ep in range(n_episodes):
    state = env.reset()
    ep_reward = 0
    done = False
    
    while not done:
        action = 2  # همیشه 1850 kbps
        state, reward, done, info = env.step(action)
        ep_reward += reward
    
    rewards.append(ep_reward)

print(f"Result: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
fixed_medium = np.mean(rewards)
print()

# ══════════════════════════════════════════════════════════
# Baseline 3: Fixed Bitrate (High)
# ══════════════════════════════════════════════════════════

print("🔴 Baseline 3: Fixed High (6000 kbps)")
print("-" * 80)

rewards = []
for ep in range(n_episodes):
    state = env.reset()
    ep_reward = 0
    done = False
    
    while not done:
        action = 5  # همیشه 6000 kbps
        state, reward, done, info = env.step(action)
        ep_reward += reward
    
    rewards.append(ep_reward)

print(f"Result: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
fixed_high = np.mean(rewards)
print()

# ══════════════════════════════════════════════════════════
# Baseline 4: Throughput-Based
# ══════════════════════════════════════════════════════════

print("🟢 Baseline 4: Throughput-Based (85% of avg throughput)")
print("-" * 80)

rewards = []
rebuffers = []

for ep in range(n_episodes):
    state = env.reset()
    ep_reward = 0
    ep_rebuffer = 0
    done = False
    
    throughput_history = []
    
    while not done:
        # محاسبه average throughput
        if len(throughput_history) > 0:
            avg_throughput = np.mean(throughput_history[-5:])  # آخرین 5 تا
            target_bitrate = avg_throughput * 0.85 * 1000  # Mbps → kbps
            
            # انتخاب نزدیک‌ترین bitrate
            action = 0
            for i, br in enumerate(bitrates):
                if target_bitrate >= br:
                    action = i
        else:
            action = 0  # شروع با پایین‌ترین
        
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        
        # ذخیره throughput برای آینده
        # از state می‌گیریم (آخرین مقدار past_throughput)
        if hasattr(env, 'past_throughput') and env.past_throughput:
            throughput_history.append(env.past_throughput[-1])
    
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)

print(f"Result: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
print(f"Rebuffering: {np.mean(rebuffers):.2f}s")
throughput_based = np.mean(rewards)
print()

# ══════════════════════════════════════════════════════════
# Baseline 5: Buffer-Based (BBA)
# ══════════════════════════════════════════════════════════

print("🟣 Baseline 5: Buffer-Based (BBA)")
print("-" * 80)

rewards = []
rebuffers = []

for ep in range(n_episodes):
    state = env.reset()
    ep_reward = 0
    ep_rebuffer = 0
    done = False
    
    while not done:
        # BBA algorithm
        buffer = env.buffer
        
        if buffer < 5:
            action = 0      # 300 kbps
        elif buffer < 10:
            action = 1      # 750 kbps
        elif buffer < 20:
            action = 2      # 1850 kbps
        elif buffer < 30:
            action = 3      # 2850 kbps
        elif buffer < 40:
            action = 4      # 4300 kbps
        else:
            action = 5      # 6000 kbps
        
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
    
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)

print(f"Result: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
print(f"Rebuffering: {np.mean(rebuffers):.2f}s")
buffer_based = np.mean(rewards)
print()

# ══════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════

print("=" * 80)
print("📊 BASELINE COMPARISON")
print("=" * 80)
print()

our_model = 75.33  # نتیجه مدل ما

results = [
    ("Fixed Low (300 kbps)", fixed_low),
    ("Fixed Medium (1850 kbps)", fixed_medium),
    ("Fixed High (6000 kbps)", fixed_high),
    ("Throughput-Based", throughput_based),
    ("Buffer-Based (BBA)", buffer_based),
    ("Our Model", our_model),
]

# مرتب کردن
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print(f"{'Method':<30} {'Reward':>12} {'Rank':>8}")
print("-" * 80)

for rank, (method, reward) in enumerate(results_sorted, 1):
    marker = "🏆" if rank == 1 else f"{rank}."
    highlight = " ← OUR MODEL" if method == "Our Model" else ""
    print(f"{marker} {method:<27} {reward:>+12.2f} {highlight}")

print()
print("=" * 80)
print("📈 Performance Analysis:")
print("=" * 80)

best_baseline = max([r[1] for r in results if r[0] != "Our Model"])
best_baseline_name = [r[0] for r in results if r[1] == best_baseline][0]

improvement = ((our_model - best_baseline) / best_baseline) * 100

print(f"Best Baseline: {best_baseline_name}")
print(f"  Reward: {best_baseline:+.2f}")
print()
print(f"Our Model:")
print(f"  Reward: {our_model:+.2f}")
print(f"  vs Best Baseline: {improvement:+.1f}%")
print()

if improvement > 0:
    print(f"✅ Our model is {improvement:+.1f}% BETTER than best baseline!")
elif improvement > -10:
    print(f"⚠️  Our model is {abs(improvement):.1f}% below best baseline")
else:
    print(f"❌ Our model is {abs(improvement):.1f}% below best baseline")

print("=" * 80)