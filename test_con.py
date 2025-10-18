"""
تست checkpoint بهترین از continued training
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
print("🏆 Testing Best Checkpoint from Continued Training")
print("=" * 80)
print()

# Load model
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)

checkpoint_path = 'results/fcc_training_continued/checkpoint_best.pth'

try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded: {checkpoint_path}")
    print(f"   Update: {checkpoint.get('update', 'N/A')}")
    print(f"   Best reward (during training): {checkpoint.get('best_reward', 'N/A'):+.2f}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)

model.eval()
print()

# Load environment
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

mode = 'val'

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode=mode
)

print(f"Testing on: {mode} set")
print()

# Test
print("🧪 Running 50 episodes...")
print("-" * 80)

rewards = []
rebuffers = []
bitrates_list = []

for ep in range(50):
    state = env.reset()
    if state is None:
        continue
    
    ep_reward = 0
    ep_rebuffer = 0
    ep_bitrates = []
    done = False
    
    while not done:
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = model(net, cont, vmaf)
            action = probs.argmax(dim=1).item()
        
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
    
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates))
    
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1:2d}/50: Reward={ep_reward:+8.2f}, "
              f"Rebuffer={ep_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(ep_bitrates):6.0f}kbps")

print()
print("=" * 80)
print("📊 FINAL RESULTS - Continued Training")
print("=" * 80)
print()

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
mean_rebuffer = np.mean(rebuffers)
mean_bitrate = np.mean(bitrates_list)

print(f"Reward:")
print(f"  Mean:        {mean_reward:+8.2f}")
print(f"  Std:         {std_reward:8.2f}")
print(f"  Median:      {np.median(rewards):+8.2f}")
print(f"  Min:         {np.min(rewards):+8.2f}")
print(f"  Max:         {np.max(rewards):+8.2f}")
print()

print(f"Rebuffering:")
print(f"  Mean:        {mean_rebuffer:8.2f}s")
print(f"  Total:       {np.sum(rebuffers):8.2f}s")
print(f"  Max:         {np.max(rebuffers):8.2f}s")
print()

print(f"Bitrate:")
print(f"  Mean:        {mean_bitrate:8.0f} kbps")
print(f"  Std:         {np.std(bitrates_list):8.0f} kbps")
print()

print("=" * 80)
print("🎯 COMPARISON")
print("=" * 80)
print()

# مقایسه با نتایج قبلی
old_model = 75.33
bba = 68.24

print(f"{'Method':<30} {'Reward':>12} {'vs BBA':>12}")
print("-" * 80)
print(f"{'BBA Baseline':<30} {bba:>+12.2f} {100.0:>11.1f}%")
print(f"{'Old Model (ckpt_300)':<30} {old_model:>+12.2f} {(old_model/bba*100):>11.1f}%")
print(f"{'New Model (continued)':<30} {mean_reward:>+12.2f} {(mean_reward/bba*100):>11.1f}%")
print()

improvement_vs_bba = ((mean_reward - bba) / bba) * 100
improvement_vs_old = ((mean_reward - old_model) / old_model) * 100

print("=" * 80)
print("📈 Improvements")
print("=" * 80)
print()
print(f"vs BBA:       {improvement_vs_bba:+.1f}%")
print(f"vs Old Model: {improvement_vs_old:+.1f}%")
print()

if mean_reward > bba * 1.15:
    print("🏆🏆🏆 EXCELLENT! Significantly better than BBA!")
elif mean_reward > bba * 1.05:
    print("🏆 GREAT! Much better than BBA!")
elif mean_reward > bba:
    print("✅ SUCCESS! Better than BBA!")
elif mean_reward > bba * 0.95:
    print("✅ Very close to BBA!")
else:
    print("⚠️  Below BBA")

print()
print("=" * 80)
print("📝 FINAL RESULT FOR REPORT:")
print("=" * 80)
print()
print(f"✅ Model: Content-Aware ABR (continued training)")
print(f"✅ Dataset: Validation set (19 FCC traces)")
print(f"✅ Episodes: 50")
print(f"✅ Mean Reward: {mean_reward:+.2f}")
print(f"✅ Improvement vs BBA: {improvement_vs_bba:+.1f}%")
print(f"✅ Mean Rebuffering: {mean_rebuffer:.2f}s")
print(f"✅ Mean Bitrate: {mean_bitrate:.0f} kbps")
print()
print("=" * 80)