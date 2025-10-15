import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 70)
print("🧪 Evaluating FCC Model on TEST Set")
print("=" * 70)

# Load model
print("\n📦 Loading model...")
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)

try:
    checkpoint = torch.load('results/fcc_training/checkpoint_400.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✅ Loaded checkpoint from update {checkpoint['update']}")
except:
    try:
        checkpoint = torch.load('results/fcc_training/checkpoint_100.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ Loaded checkpoint from update {checkpoint['update']}")
    except:
        print("   ❌ No checkpoint found!")
        sys.exit(1)

model.eval()

# Load environment
print("\n📦 Loading test environment...")
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
    mode='test'  # Test set!
)

print("   ✅ Environment loaded")

# Evaluate
print("\n🧪 Running evaluation on 50 test episodes...")
print("-" * 70)

episode_rewards = []
episode_rebuffers = []
episode_bitrates = []

for ep in range(50):
    state = env.reset()
    episode_reward = 0
    episode_rebuffer = 0
    bitrates = []
    done = False
    
    while not done:
        network_state = torch.FloatTensor(state['network']).unsqueeze(0)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf_features = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = model(network_state, content_features, vmaf_features)
            action = action_probs.argmax(dim=1).item()
        
        state, reward, done, info = env.step(action)
        episode_reward += reward
        episode_rebuffer += info['rebuffer_time']
        bitrates.append(info['bitrate'])
    
    episode_rewards.append(episode_reward)
    episode_rebuffers.append(episode_rebuffer)
    episode_bitrates.append(np.mean(bitrates))
    
    if (ep + 1) % 10 == 0:
        print(f"  Episode {ep+1:2d}/50: Reward={episode_reward:+7.2f}, "
              f"Rebuffer={episode_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(bitrates):6.0f}kbps")

print("\n" + "=" * 70)
print("📊 TEST SET RESULTS")
print("=" * 70)
print(f"Reward:")
print(f"  Mean:  {np.mean(episode_rewards):+7.2f}")
print(f"  Std:   {np.std(episode_rewards):7.2f}")
print(f"  Min:   {np.min(episode_rewards):+7.2f}")
print(f"  Max:   {np.max(episode_rewards):+7.2f}")
print()
print(f"Rebuffering:")
print(f"  Mean:  {np.mean(episode_rebuffers):7.2f}s")
print(f"  Total: {np.sum(episode_rebuffers):7.2f}s")
print()
print(f"Bitrate:")
print(f"  Mean:  {np.mean(episode_bitrates):7.0f} kbps")
print(f"  Std:   {np.std(episode_bitrates):7.0f} kbps")
print()
print("=" * 70)
print("Baseline Comparison:")
print(f"  Buffer-Based:  +102.16")
print(f"  Your Model:    {np.mean(episode_rewards):+7.2f}  ({np.mean(episode_rewards)/102.16*100:.1f}% of baseline)")
print("=" * 70)