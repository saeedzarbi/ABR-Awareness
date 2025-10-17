"""
تست کامل از صفر - بهترین حالت
checkpoint_400.pth + Safety Wrapper
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
from datetime import datetime
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🧪 FINAL TEST: Best Model Configuration")
print("=" * 80)
print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ═══════════════════════════════════════════════════════════
# 1. بارگذاری مدل
# ═══════════════════════════════════════════════════════════

print("📦 Loading Model...")
print("-" * 80)

model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)

try:
    checkpoint = torch.load('results/fcc_training/checkpoint_400.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded: checkpoint_400.pth")
    print(f"   Training update: {checkpoint['update']}")
    train_reward = checkpoint.get('train_info', {}).get('mean_reward', 'N/A')
    print(f"   Training reward: {train_reward}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

model.eval()
print()

# ═══════════════════════════════════════════════════════════
# 2. تعریف Safety Wrapper
# ═══════════════════════════════════════════════════════════

print("🛡️  Safety Wrapper Configuration...")
print("-" * 80)

class SafetyWrapper:
    """
    Safety wrapper برای جلوگیری از aggressive decisions
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Safety thresholds
        self.thresholds = [
            (5.0, 1),   # buffer < 5s  → max action = 1 (750 kbps)
            (10.0, 2),  # buffer < 10s → max action = 2 (1850 kbps)
            (20.0, 3),  # buffer < 20s → max action = 3 (2850 kbps)
        ]
    
    def select_action(self, state, buffer):
        """
        انتخاب action با safety rules
        """
        # دریافت action از مدل
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = self.model(net, cont, vmaf)
            action = probs.argmax(dim=1).item()
        
        # اعمال safety rules
        original_action = action
        for threshold, max_action in self.thresholds:
            if buffer < threshold:
                action = min(action, max_action)
                break
        
        return action, original_action

policy = SafetyWrapper(model)

print("Safety Rules:")
print("  - Buffer < 5s  → Max bitrate: 750 kbps")
print("  - Buffer < 10s → Max bitrate: 1850 kbps")
print("  - Buffer < 20s → Max bitrate: 2850 kbps")
print("  - Buffer ≥ 20s → No limit")
print()

# ═══════════════════════════════════════════════════════════
# 3. بارگذاری Test Environment
# ═══════════════════════════════════════════════════════════

print("🌐 Loading Test Environment...")
print("-" * 80)

loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'
)

print(f"✅ Test set loaded:")
print(f"   Number of traces: {len(loader.test_traces)}")
print(f"   Bitrate levels: {env.bitrates} kbps")
print()

# ═══════════════════════════════════════════════════════════
# 4. اجرای تست
# ═══════════════════════════════════════════════════════════

print("🧪 Running Test Episodes...")
print("-" * 80)
print()

n_episodes = 50
episode_results = []

for ep in range(n_episodes):
    state = env.reset()
    
    episode_data = {
        'reward': 0,
        'rebuffer_time': 0,
        'bitrates': [],
        'vmaf_scores': [],
        'buffer_history': [],
        'safety_interventions': 0
    }
    
    done = False
    step = 0
    
    while not done:
        # انتخاب action
        action, original_action = policy.select_action(state, env.buffer)
        
        # بررسی safety intervention
        if action != original_action:
            episode_data['safety_interventions'] += 1
        
        # اجرای action
        next_state, reward, done, info = env.step(action)
        
        # ذخیره اطلاعات
        episode_data['reward'] += reward
        episode_data['rebuffer_time'] += info['rebuffer_time']
        episode_data['bitrates'].append(info['bitrate'])
        episode_data['buffer_history'].append(env.buffer)
        
        state = next_state
        step += 1
    
    episode_results.append(episode_data)
    
    # نمایش پیشرفت
    if (ep + 1) % 10 == 0 or ep == 0:
        print(f"Episode {ep+1:3d}/{n_episodes}: "
              f"Reward = {episode_data['reward']:+8.2f}, "
              f"Rebuffer = {episode_data['rebuffer_time']:5.2f}s, "
              f"Avg Bitrate = {np.mean(episode_data['bitrates']):6.0f} kbps, "
              f"Safety = {episode_data['safety_interventions']:3d}×")

print()

# ═══════════════════════════════════════════════════════════
# 5. تحلیل نتایج
# ═══════════════════════════════════════════════════════════

print("=" * 80)
print("📊 TEST RESULTS")
print("=" * 80)
print()

# محاسبه آمارها
rewards = [ep['reward'] for ep in episode_results]
rebuffers = [ep['rebuffer_time'] for ep in episode_results]
avg_bitrates = [np.mean(ep['bitrates']) for ep in episode_results]
safety_counts = [ep['safety_interventions'] for ep in episode_results]

# نمایش نتایج
print("🎯 Performance Metrics:")
print("-" * 80)
print(f"Reward (QoE):")
print(f"  Mean:        {np.mean(rewards):+8.2f}")
print(f"  Std:         {np.std(rewards):8.2f}")
print(f"  Min:         {np.min(rewards):+8.2f}")
print(f"  Max:         {np.max(rewards):+8.2f}")
print(f"  Median:      {np.median(rewards):+8.2f}")
print()

print(f"Rebuffering:")
print(f"  Mean:        {np.mean(rebuffers):8.2f}s")
print(f"  Std:         {np.std(rebuffers):8.2f}s")
print(f"  Total:       {np.sum(rebuffers):8.2f}s")
print(f"  Max:         {np.max(rebuffers):8.2f}s")
print()

print(f"Bitrate:")
print(f"  Mean:        {np.mean(avg_bitrates):8.0f} kbps")
print(f"  Std:         {np.std(avg_bitrates):8.0f} kbps")
print(f"  Min:         {np.min(avg_bitrates):8.0f} kbps")
print(f"  Max:         {np.max(avg_bitrates):8.0f} kbps")
print()

print(f"Safety Interventions:")
print(f"  Mean:        {np.mean(safety_counts):8.1f} per episode")
print(f"  Total:       {np.sum(safety_counts):8.0f}")
print()

# مقایسه با baseline
print("=" * 80)
print("📈 Comparison with Baselines")
print("=" * 80)
print()

baseline_bba = 102.16
improvement = ((np.mean(rewards) - baseline_bba) / baseline_bba) * 100

print(f"{'Method':<30} {'Reward':>12} {'vs BBA':>12}")
print("-" * 80)
print(f"{'Buffer-Based (BBA)':<30} {baseline_bba:>+12.2f} {'100.0%':>12}")
print(f"{'Our Model + Safety':<30} {np.mean(rewards):>+12.2f} {f'{100+improvement:.1f}%':>12}")
print()

if improvement > 0:
    print(f"✅ Our model is {improvement:+.1f}% better than BBA baseline!")
else:
    print(f"⚠️  Our model is {improvement:.1f}% worse than BBA baseline")

print()

# ═══════════════════════════════════════════════════════════
# 6. تحلیل توزیع
# ═══════════════════════════════════════════════════════════

print("=" * 80)
print("📉 Distribution Analysis")
print("=" * 80)
print()

print("Reward Distribution:")
bins = [(-float('inf'), -100), (-100, 0), (0, 50), (50, 100), (100, 150), (150, float('inf'))]
for low, high in bins:
    count = sum(1 for r in rewards if low < r <= high)
    pct = count / len(rewards) * 100
    bar = '█' * int(pct / 2)
    if high != float('inf'):
        label = f"{low:>6} to {high:<6}"
    else:
        label = f"  >{low:>6}      "
    print(f"  {label}: {bar:<25} {count:3d} ({pct:5.1f}%)")

print()

print("Rebuffering Distribution:")
bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, float('inf'))]
for low, high in bins:
    count = sum(1 for r in rebuffers if low < r <= high)
    pct = count / len(rebuffers) * 100
    bar = '█' * int(pct / 2)
    if high != float('inf'):
        label = f"{low:>3}s to {high:<3}s"
    else:
        label = f" >{low:>3}s     "
    print(f"  {label}: {bar:<25} {count:3d} ({pct:5.1f}%)")

print()

# ═══════════════════════════════════════════════════════════
# 7. ذخیره نتایج
# ═══════════════════════════════════════════════════════════

results_summary = {
    'timestamp': datetime.now().isoformat(),
    'model': 'checkpoint_400.pth + Safety Wrapper',
    'n_episodes': n_episodes,
    'metrics': {
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards))
        },
        'rebuffering': {
            'mean': float(np.mean(rebuffers)),
            'std': float(np.std(rebuffers)),
            'total': float(np.sum(rebuffers)),
            'max': float(np.max(rebuffers))
        },
        'bitrate': {
            'mean': float(np.mean(avg_bitrates)),
            'std': float(np.std(avg_bitrates))
        },
        'safety_interventions': {
            'mean': float(np.mean(safety_counts)),
            'total': int(np.sum(safety_counts))
        }
    },
    'comparison': {
        'baseline_bba': baseline_bba,
        'improvement_percent': float(improvement)
    },
    'episodes': episode_results
}

output_file = f'results/test_best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print("=" * 80)
print("💾 Results Saved")
print("=" * 80)
print(f"Output file: {output_file}")
print()

print("=" * 80)
print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()
print("✅ Test Complete!")