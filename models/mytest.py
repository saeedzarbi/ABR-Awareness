"""
Training سریع از صفر
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer

print("=" * 80)
print("🚀 Quick Training: Building checkpoint_400 from Scratch")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════

print("📦 Loading Data...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env_train = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='train'
)

print(f"✅ Train traces: {len(loader.train_traces)}")
print()

# Model
print("🧠 Creating Model...")
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

print(f"✅ Model created")
print()

# Training
print("🎯 Starting Training...")
print("   Target: 400 updates (~800K steps)")
print()

update = 0
rollout_steps = 2048
target_updates = 400

while update < target_updates:
    update += 1
    
    # Collect rollout (simplified)
    states = []
    actions = []
    rewards = []
    
    for _ in range(rollout_steps):
        state = env_train.reset() if not states else state
        
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = model(net, cont, vmaf)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
        next_state, reward, done, info = env_train.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Simple update (بدون PPO کامل برای سرعت)
    mean_reward = np.mean(rewards)
    
    # لاگ
    if update % 50 == 0:
        print(f"Update {update:3d}/{target_updates}: Reward = {mean_reward:+8.2f}")
    
    # Save checkpoint
    if update % 100 == 0 or update == target_updates:
        checkpoint = {
            'update': update,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_info': {'mean_reward': mean_reward}
        }
        
        path = f'results/fcc_training/checkpoint_{update}_new.pth'
        torch.save(checkpoint, path)
        print(f"💾 Saved: {path}")
        print()

print()
print("=" * 80)
print("✅ Training Complete!")
print("=" * 80)