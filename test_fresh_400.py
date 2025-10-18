"""
تست checkpoint_300 با Safety Wrapper
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
print("🏆 FINAL TEST: checkpoint_300 + Safety Wrapper")
print("=" * 80)
print()

# بارگذاری checkpoint_300 (بهترین!)
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)

checkpoint_path = 'results/fcc_training/checkpoint_300.pth'

try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded: {checkpoint_path}")
    print(f"   Update: {checkpoint.get('update', 'N/A')}")
    print(f"   Previous test: +98.21 reward")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)

model.eval()
print()

# Safety Wrapper
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
        
        # Safety rules
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

# Load data
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

mode = 'test' if len(loader.test_traces) >= 20 else 'val'

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode=mode
)

print(f"🧪 Testing on {mode} set")
print()

# Test با 50 episodes
print(f"Running 50 episodes with Safety Wrapper...")
print("-" * 80)

rewards = []
rebuffers = []
bitrates_list = []
safety_counts = []

for ep in range(50):
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
        print(f"Episode {ep+1:2d}/50: Reward={ep_reward:+8.2f}, "
              f"Rebuffer={ep_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(ep_bitrates):6.0f}kbps, "
              f"Safety={ep_safety:2d}×")

print()
print("=" * 80)
print("📊 FINAL RESULTS - checkpoint_300 + Safety Wrapper")
print("=" * 80)
print()

print(f"Reward:")
print(f"  Mean:        {np.mean(rewards):+8.2f}")
print(f"  Std:         {np.std(rewards):8.2f}")
print(f"  Median:      {np.median(rewards):+8.2f}")
print(f"  Min:         {np.min(rewards):+8.2f}")
print(f"  Max:         {np.max(rewards):+8.2f}")
print()

print(f"Rebuffering:")
print(f"  Mean:        {np.mean(rebuffers):8.2f}s")
print(f"  Total:       {np.sum(rebuffers):8.2f}s")
print(f"  Max:         {np.max(rebuffers):8.2f}s")
print()

print(f"Bitrate:")
print(f"  Mean:        {np.mean(bitrates_list):8.0f} kbps")
print(f"  Std:         {np.std(bitrates_list):8.0f} kbps")
print()

print(f"Safety Interventions:")
print(f"  Mean:        {np.mean(safety_counts):8.1f} per episode")
print(f"  Total:       {sum(safety_counts):8d}")
print()

baseline = 102.16
improvement = ((np.mean(rewards) - baseline) / baseline) * 100

print("=" * 80)
print("🎯 Comparison with Baseline")
print("=" * 80)
print(f"Buffer-Based (BBA):     {baseline:+8.2f}  (100.0%)")
print(f"Our Model (no safety):   +98.21        (96.1%)")
print(f"Our Model + Safety:     {np.mean(rewards):+8.2f}  ({100+improvement:.1f}%)")
print()

if improvement > 10:
    print(f"🏆🏆🏆 EXCELLENT! {improvement:+.1f}% better than baseline!")
    print("     ✅ This is our FINAL result for the report!")
elif improvement > 5:
    print(f"🏆 GREAT! {improvement:+.1f}% better than baseline!")
elif improvement > 0:
    print(f"✅ SUCCESS! {improvement:+.1f}% better than baseline!")
elif improvement > -5:
    print(f"⚠️  Close to baseline ({improvement:+.1f}%)")
else:
    print(f"❌ Below baseline ({improvement:+.1f}%)")

print()
print("=" * 80)
print("📝 Summary for Report:")
print("=" * 80)
print(f"✅ Model: checkpoint_300 (best from training)")
print(f"✅ Safety Wrapper: Buffer-based thresholds")
print(f"✅ Test episodes: 50")
print(f"✅ Mean reward: {np.mean(rewards):+.2f}")
print(f"✅ Mean rebuffering: {np.mean(rebuffers):.2f}s")
print(f"✅ Improvement: {improvement:+.1f}% vs BBA")
print("=" * 80)
