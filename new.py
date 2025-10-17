"""
Training کامل از صفر
ساخت checkpoint_400 جدید
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🚀 FRESH TRAINING: Building checkpoint_400 from Scratch")
print("=" * 80)
print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ═══════════════════════════════════════════════════════════
# پاک کردن checkpoint های قبلی (اختیاری)
# ═══════════════════════════════════════════════════════════

checkpoint_dir = 'results/fcc_training_fresh'
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"📁 Output directory: {checkpoint_dir}")
print()

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.10,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'batch_size': 64,
    'ppo_epochs': 4,
    'rollout_steps': 2048,
    'n_updates': 400,  # 400 updates
    'eval_interval': 50,
    'checkpoint_interval': 100
}

print("⚙️  Training Configuration:")
for key, val in config.items():
    print(f"   {key}: {val}")
print()

# ═══════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════

print("📦 Loading Training Data...")

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

env_val = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='val'
)

print(f"✅ Data loaded:")
print(f"   Train traces: {len(loader.train_traces)}")
print(f"   Val traces: {len(loader.val_traces)}")
print(f"   Test traces: {len(loader.test_traces)}")
print()

# ═══════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════

print("🧠 Creating Model...")

model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model created: {n_params:,} parameters")
print()

# ═══════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """محاسبه GAE advantages"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def collect_rollout(env, model, n_steps):
    """جمع‌آوری rollout"""
    rollout = {
        'states': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'log_probs': [],
        'dones': []
    }
    
    state = env.reset()
    episode_rewards = []
    current_episode_reward = 0
    
    for step in range(n_steps):
        # State to tensor
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        # Forward
        with torch.no_grad():
            action_probs, value = model(net, cont, vmaf)
        
        # Sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Step
        next_state, reward, done, info = env.step(action.item())
        current_episode_reward += reward
        
        # Store
        rollout['states'].append(state)
        rollout['actions'].append(action.item())
        rollout['rewards'].append(reward)
        rollout['values'].append(value.item())
        rollout['log_probs'].append(log_prob.item())
        rollout['dones'].append(done)
        
        state = next_state
        
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            state = env.reset()
    
    rollout['episode_rewards'] = episode_rewards
    return rollout


def ppo_update(model, optimizer, rollout, config):
    """به‌روزرسانی با PPO"""
    
    # محاسبه advantages
    advantages, returns = compute_gae(
        rollout['rewards'],
        rollout['values'],
        rollout['dones'],
        config['gamma'],
        config['gae_lambda']
    )
    
    # Normalize advantages
    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Convert to tensors
    old_log_probs = torch.FloatTensor(rollout['log_probs'])
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)
    
    # PPO epochs
    n_samples = len(rollout['states'])
    batch_size = config['batch_size']
    
    policy_losses = []
    value_losses = []
    entropies = []
    
    for epoch in range(config['ppo_epochs']):
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            # Prepare batch
            batch_states = [rollout['states'][i] for i in batch_idx]
            batch_actions = torch.LongTensor([rollout['actions'][i] for i in batch_idx])
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]
            
            # States to tensors
            batch_net = torch.stack([torch.FloatTensor(s['network']) for s in batch_states])
            batch_cont = torch.stack([torch.FloatTensor(s['content']) for s in batch_states])
            batch_vmaf = torch.stack([torch.FloatTensor(s['vmaf']) for s in batch_states])
            
            # Forward
            action_probs, values = model(batch_net, batch_cont, batch_vmaf)
            
            # Policy loss
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon']) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
            
            # Entropy
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
    
    return {
        'policy_loss': np.mean(policy_losses),
        'value_loss': np.mean(value_losses),
        'entropy': np.mean(entropies)
    }


def evaluate(env, model, n_episodes=10):
    """ارزیابی مدل"""
    rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            net = torch.FloatTensor(state['network']).unsqueeze(0)
            cont = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = model(net, cont, vmaf)
            
            action = action_probs.argmax(dim=1).item()
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)


# ═══════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════

print("🎯 Starting Training...")
print("-" * 80)
print()

best_val_reward = -float('inf')

for update in range(1, config['n_updates'] + 1):
    # Collect rollout
    rollout = collect_rollout(env_train, model, config['rollout_steps'])
    
    # PPO update
    train_info = ppo_update(model, optimizer, rollout, config)
    
    # Calculate mean reward
    if rollout['episode_rewards']:
        mean_reward = np.mean(rollout['episode_rewards'])
    else:
        mean_reward = 0
    
    # Log
    if update % 10 == 0:
        print(f"Update {update:3d}/{config['n_updates']}: "
              f"Reward={mean_reward:+7.2f}, "
              f"Policy Loss={train_info['policy_loss']:.4f}, "
              f"Entropy={train_info['entropy']:.4f}")
    
    # Evaluation
    if update % config['eval_interval'] == 0:
        val_reward = evaluate(env_val, model, n_episodes=5)
        print(f"  📊 Validation Reward: {val_reward:+.2f}")
        
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            print(f"  🏆 New best!")
    
    # Save checkpoint
    if update % config['checkpoint_interval'] == 0:
        checkpoint = {
            'update': update,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_info': {
                'mean_reward': mean_reward,
                'policy_loss': train_info['policy_loss'],
                'value_loss': train_info['value_loss'],
                'entropy': train_info['entropy']
            }
        }
        
        path = os.path.join(checkpoint_dir, f'checkpoint_{update}.pth')
        torch.save(checkpoint, path)
        print(f"  💾 Saved: {path}")
        print()

print()
print("=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print(f"⏰ End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Checkpoints saved in: {checkpoint_dir}")
print()