
# نسخه بهبود یافته برای کاهش Overfitting
from models.content_aware_model import create_content_aware_model
"""
Training بهبود یافته با Early Stopping و Monitoring
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import time

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
import random
import numpy as np
import torch

SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

checkpoint_dir = 'results/fcc_training_improved_s'
log_file = os.path.join(checkpoint_dir, 'training_log.json')
os.makedirs(checkpoint_dir, exist_ok=True)

config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.05,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'batch_size': 64,
    'ppo_epochs': 4,
    'rollout_steps': 2048,
    'n_updates': 400,
    'eval_interval': 10,
    'checkpoint_interval': 25,
    'log_interval': 5,
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 0.5
}

loader = FCCTraceLoader(
    fcc_trace_dir='data/network_traces/fcc'
    train_file='data/network_traces/fcc/splits/fcc_train.txt'
    val_file='data/network_traces/fcc/splits/fcc_val.txt'
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

model = create_content_aware_model()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

print(f"\n🚀 Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def collect_rollout(env, model, n_steps):
    rollout = {
        'states': [], 'actions': [], 'rewards': [],
        'values': [], 'log_probs': [], 'dones': []
    }
    
    state = env.reset()
    episode_rewards = []
    current_episode_reward = 0
    
    for step in range(n_steps):
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = model(net, cont, vmaf)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_state, reward, done, info = env.step(action.item())
        current_episode_reward += reward
        
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
    advantages, returns = compute_gae(
        rollout['rewards'], rollout['values'], rollout['dones'],
        config['gamma'], config['gae_lambda']
    )
    
    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    old_log_probs = torch.FloatTensor(rollout['log_probs'])
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)
    
    n_samples = len(rollout['states'])
    batch_size = config['batch_size']
    
    policy_losses, value_losses, entropies = [], [], []
    
    for epoch in range(config['ppo_epochs']):
        indices = np.random.permutation(n_samples)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            batch_states = [rollout['states'][i] for i in batch_idx]
            batch_actions = torch.LongTensor([rollout['actions'][i] for i in batch_idx])
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]
            
            batch_net = torch.stack([torch.FloatTensor(s['network']) for s in batch_states])
            batch_cont = torch.stack([torch.FloatTensor(s['content']) for s in batch_states])
            batch_vmaf = torch.stack([torch.FloatTensor(s['vmaf']) for s in batch_states])
            
            action_probs, values = model(batch_net, batch_cont, batch_vmaf)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)
            
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon']) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
            entropy = dist.entropy().mean()
            
            loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy
            
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
    
    return np.mean(rewards), np.std(rewards)

training_log = []
best_val_reward = -float('inf')
no_improvement_count = 0
start_time = time.time()

for update in range(1, config['n_updates'] + 1):
    update_start = time.time()
    
    # Collect rollout
    rollout = collect_rollout(env_train, model, config['rollout_steps'])
    
    # PPO update
    train_info = ppo_update(model, optimizer, rollout, config)
    
    # Calculate metrics
    mean_reward = np.mean(rollout['episode_rewards']) if rollout['episode_rewards'] else 0
    update_time = time.time() - update_start
    elapsed_time = time.time() - start_time
    
    # Log entry
    log_entry = {
        'update': update,
        'mean_reward': float(mean_reward),
        'policy_loss': float(train_info['policy_loss']),
        'value_loss': float(train_info['value_loss']),
        'entropy': float(train_info['entropy']),
        'n_episodes': len(rollout['episode_rewards']),
        'update_time': float(update_time),
        'elapsed_time': float(elapsed_time)
    }
    
    # Evaluation
    if update % config['eval_interval'] == 0:
        val_mean, val_std = evaluate(env_val, model, n_episodes=10)
        log_entry['val_reward_mean'] = float(val_mean)
        log_entry['val_reward_std'] = float(val_std)
        
        # ✅ Early Stopping Logic
        improvement = val_mean - best_val_reward
        
        if improvement > config['early_stopping_min_delta']:
            best_val_reward = val_mean
            no_improvement_count = 0
            log_entry['new_best'] = True
            
            # Save best model
            best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
            torch.save({
                'update': update,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_reward': val_mean
            }, best_path)
        else:
            no_improvement_count += 1
            
        # Check early stopping
        if no_improvement_count >= config['early_stopping_patience']:
            print(f"\n⏸️  Early stopping at update {update} (Best: {best_val_reward:+.2f})")
            break
    
    training_log.append(log_entry)
    
    # Console output
    if update % config['log_interval'] == 0:
        progress = (update / config['n_updates']) * 100
        marker = "🏆" if log_entry.get('new_best', False) else "  "
        
        if 'val_reward_mean' in log_entry:
            print(f"[{progress:5.1f}%] Reward: {mean_reward:+7.2f}  {marker} Val: {log_entry['val_reward_mean']:+.2f}")
        else:
            print(f"[{progress:5.1f}%] Reward: {mean_reward:+7.2f}")
    
    # Save checkpoint
    if update % config['checkpoint_interval'] == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{update}.pth')
        torch.save({
            'update': update,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_info': train_info
        }, checkpoint_path)
    
    # Save log
    if update % 10 == 0:
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)

# Final save
with open(log_file, 'w') as f:
    json.dump(training_log, f, indent=2)

print(f"\n✅ Training Complete! Best Val Reward: {best_val_reward:+.2f}")
print(f"Time: {(time.time() - start_time) / 60:.1f}min")
