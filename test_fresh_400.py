"""
تست checkpoint جدید (400 fresh)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🧪 Testing Fresh checkpoint_400")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════
# بارگذاری مدل جدید
# ═══════════════════════════════════════════════════════════

model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)

checkpoint = torch.load('results/fcc_training_fresh/checkpoint_400.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded fresh checkpoint_400")
print(f"   Update: {checkpoint['update']}")
print(f"   Train info: {checkpoint.get('train_info', {})}")
print()

model.eval()

# ═══════════════════════════════════════════════════════════
# Safety Wrapper
# ═══════════════════════════════════════════════════════════

class SafetyWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def select_action(self, state, buffer):
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = self.model(net, cont, vmaf)
            action = probs.argmax(dim=1).item()
        
        original = action
        
        if buffer < 5.0:
            action = min(action, 1)
        elif buffer < 10.0:
            action = min(action, 2)
        elif buffer < 20.0:
            action = min(action, 3)
        
        return action, original

policy = SafetyWrapper(model)
print("🛡️  Safety wrapper enabled")
print()

# ═══════════════════════════════════════════════════════════
# Load test environment
# ═══════════════════════════════════════════════════════════

loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

print(f"📊 Traces:")
print(f"   Test: {len(loader.test_traces)}")
print()

# استفاده از val اگر test کم باشه
if len(loader.test_traces) < 20:
    print("⚠️  Using validation (test < 20)")
    mode = 'val'
else:
    mode = 'test'

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode=mode
)

# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

print(f"🧪 Running 30 episodes on {mode} set...")
print("-" * 80)

rewards = []
rebuffers = []
bitrates_list = []
safety_counts = []

for ep in range(30):
    state = env.reset()
    
    if state is None:
        continue
    
    ep_reward = 0
    ep_rebuffer = 0
    ep_bitrates = []
    ep_safety = 0
    done = False
    
    while not done:
        action, original = policy.select_action(state, env.buffer)
        
        if action != original:
            ep_safety += 1
        
        state, reward, done, info = env.step(action)
        
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
    
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates))
    safety_counts.append(ep_safety)
    
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1:2d}/30: Reward={ep_reward:+8.2f}, "
              f"Rebuffer={ep_rebuffer:5.2f}s, "
              f"Safety={ep_safety:2d}×")

print()
print("=" * 80)
print("📊 RESULTS")
print("=" * 80)
print()

print(f"Reward:")
print(f"  Mean:        {np.mean(rewards):+8.2f}")
print(f"  Std:         {np.std(rewards):8.2f}")
print(f"  Min:         {np.min(rewards):+8.2f}")
print(f"  Max:         {np.max(rewards):+8.2f}")
print()

print(f"Rebuffering:")
print(f"  Mean:        {np.mean(rebuffers):8.2f}s")
print(f"  Total:       {np.sum(rebuffers):8.2f}s")
print()

print(f"Bitrate:")
print(f"  Mean:        {np.mean(bitrates_list):8.0f} kbps")
print()

print(f"Safety:")
print(f"  Mean:        {np.mean(safety_counts):8.1f} per episode")
print(f"  Total:       {sum(safety_counts):8d}")
print()

baseline = 102.16
improvement = ((np.mean(rewards) - baseline) / baseline) * 100

print("=" * 80)
print("vs BBA Baseline:")
print(f"  Baseline:    {baseline:+8.2f}")
print(f"  Our Model:   {np.mean(rewards):+8.2f}")
print(f"  Improvement: {improvement:+.1f}%")

if improvement > 10:
    print(f"  Status:      🏆 Excellent!")
elif improvement > 0:
    print(f"  Status:      ✅ Better!")
else:
    print(f"  Status:      ⚠️  Needs work")

print("=" * 80)
