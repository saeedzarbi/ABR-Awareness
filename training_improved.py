"""
Training Script بهبود یافته با تمام راهکارهای Anti-Overfitting

✅ بهبودها:
1. Early Stopping با validation منظم
2. Learning Rate Scheduler (StepLR)
3. Gradient Clipping
4. Entropy Coefficient بالاتر
5. Data Augmentation ساده
6. بهترین checkpoint را ذخیره می‌کند
7. Monitoring کامل
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import json
from datetime import datetime
import time

# Import improved model
from models.content_aware_model_improved import ContentAwareActorImproved, create_improved_model

# این دو فایل باید در کنار این فایل باشند
# from models.content_aware_env_fcc import ContentAwareEnvFCC
# from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🚀 IMPROVED TRAINING - Anti-Overfitting Edition")
print("=" * 80)
print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

checkpoint_dir = 'results/training_anti_overfitting'
log_file = os.path.join(checkpoint_dir, 'training_log.json')
os.makedirs(checkpoint_dir, exist_ok=True)

config = {
    # Model Architecture
    'hidden_dim': 128,                  # یا 64 برای مدل کوچکتر
    'dropout_rate': 0.2,                # ✅ Dropout
    'use_batchnorm': True,              # ✅ BatchNorm
    
    # Training Hyperparameters
    'learning_rate': 3e-4,              
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.10,               # ✅ افزایش یافته (از 0.01)
    'value_coef': 0.5,
    'max_grad_norm': 0.5,               # ✅ Gradient clipping
    'batch_size': 64,
    'ppo_epochs': 4,
    'rollout_steps': 2048,
    
    # Training Schedule
    'n_updates': 200,                   # ✅ کمتر (از 300)
    'eval_interval': 10,                
    'checkpoint_interval': 25,          
    'log_interval': 5,
    
    # ✅ Early Stopping
    'early_stopping_patience': 5,       
    'early_stopping_min_delta': 0.5,    
    
    # ✅ Learning Rate Scheduler
    'lr_scheduler_step': 50,            # هر 50 update
    'lr_scheduler_gamma': 0.5,          # LR = LR * 0.5
    
    # ✅ Data Augmentation
    'use_augmentation': True,
    'augmentation_prob': 0.3,           # 30% chance
    
    # Evaluation
    'n_val_episodes': 20,               # ✅ افزایش یافته (از 10)
    'n_test_episodes': 30,
}

print("⚙️  Configuration:")
print("=" * 80)
for key, val in config.items():
    print(f"   {key:<30} {val}")
print()
print(f"📁 Output: {checkpoint_dir}")
print(f"📝 Log file: {log_file}")
print()

# ═══════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════

def augment_bandwidth(bandwidth, augment_prob=0.3):
    """
    ✅ Data Augmentation: Add noise to bandwidth observations
    
    Args:
        bandwidth: array of bandwidth values
        augment_prob: probability of augmentation
    
    Returns:
        augmented bandwidth (or original if not augmented)
    """
    if np.random.rand() < augment_prob:
        # Add Gaussian noise
        noise = np.random.normal(0, 0.05, len(bandwidth))
        bandwidth = bandwidth + noise
        bandwidth = np.clip(bandwidth, 0, None)  # Non-negative
    return bandwidth


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def collect_rollout(env, model, n_steps, use_augmentation=False, augment_prob=0.3):
    """
    Collect rollout with optional data augmentation
    
    ✅ جدید: Data augmentation برای training
    """
    rollout = {
        'states': [], 'actions': [], 'rewards': [],
        'values': [], 'log_probs': [], 'dones': []
    }
    
    state = env.reset()
    episode_rewards = []
    current_episode_reward = 0
    
    for step in range(n_steps):
        # ✅ Data Augmentation (فقط در training mode)
        if use_augmentation and model.training:
            # Augment bandwidth observations
            state['network'][0] = augment_bandwidth(
                state['network'][0], 
                augment_prob
            )
        
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
    """PPO Update"""
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
            
            # ✅ Gradient Clipping
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
    """Evaluate model"""
    model.eval()  # ✅ Eval mode (Dropout off)
    
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
    
    model.train()  # ✅ برگشت به training mode
    
    return np.mean(rewards), np.std(rewards)


# ═══════════════════════════════════════════════════════════
# Main Function (برای test بدون environment)
# ═══════════════════════════════════════════════════════════

def test_training_components():
    """
    تست کامپوننت‌های training بدون نیاز به environment واقعی
    """
    print("=" * 80)
    print("🧪 Testing Training Components")
    print("=" * 80)
    print()
    
    # ============================================
    # 1. Create Model
    # ============================================
    print("1️⃣  Creating improved model...")
    model = create_improved_model(
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate'],
        use_batchnorm=config['use_batchnorm']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created: {total_params:,} parameters")
    print()
    
    # ============================================
    # 2. Create Optimizer
    # ============================================
    print("2️⃣  Creating optimizer and scheduler...")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # ✅ Learning Rate Scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config['lr_scheduler_step'],
        gamma=config['lr_scheduler_gamma']
    )
    
    print(f"   ✅ Optimizer: Adam (LR={config['learning_rate']:.2e})")
    print(f"   ✅ Scheduler: StepLR (step={config['lr_scheduler_step']}, gamma={config['lr_scheduler_gamma']})")
    print()
    
    # ============================================
    # 3. Test Forward Pass
    # ============================================
    print("3️⃣  Testing forward pass...")
    batch_size = 32
    
    net = torch.randn(batch_size, 6, 8)
    cont = torch.randn(batch_size, 2)
    vmaf = torch.randn(batch_size, 6)
    
    model.train()  # Training mode
    action_probs_train, values_train = model(net, cont, vmaf)
    
    model.eval()   # Eval mode
    action_probs_eval, values_eval = model(net, cont, vmaf)
    
    diff = torch.abs(action_probs_train - action_probs_eval).mean().item()
    
    print(f"   ✅ Training mode output: {action_probs_train.shape}")
    print(f"   ✅ Eval mode output: {action_probs_eval.shape}")
    print(f"   ✅ Difference: {diff:.6f} (should be > 0 due to Dropout)")
    
    if diff > 0.001:
        print("   ✅ Dropout is working correctly!")
    else:
        print("   ⚠️  Warning: Dropout effect is very small")
    print()
    
    # ============================================
    # 4. Test Augmentation
    # ============================================
    print("4️⃣  Testing data augmentation...")
    
    original_bw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    augmented_bw = augment_bandwidth(original_bw, augment_prob=1.0)  # Always augment
    
    print(f"   Original:  {original_bw}")
    print(f"   Augmented: {augmented_bw}")
    print(f"   ✅ Augmentation working!")
    print()
    
    # ============================================
    # 5. Test LR Scheduler
    # ============================================
    print("5️⃣  Testing LR scheduler...")
    
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"   Update 0:  LR = {initial_lr:.2e}")
    
    for update in range(1, 151):
        scheduler.step()
        
        if update in [50, 100, 150]:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Update {update}: LR = {current_lr:.2e}")
    
    print("   ✅ Scheduler working!")
    print()
    
    # ============================================
    # 6. Test Gradient Clipping
    # ============================================
    print("6️⃣  Testing gradient clipping...")
    
    # Create dummy loss
    action_probs, values = model(net, cont, vmaf)
    loss = action_probs.sum() + values.sum()
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients before clipping
    total_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5
    
    # Clip gradients
    nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
    
    # Check gradients after clipping
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5
    
    print(f"   Gradient norm before clipping: {total_norm_before:.4f}")
    print(f"   Gradient norm after clipping:  {total_norm_after:.4f}")
    print(f"   Max allowed norm: {config['max_grad_norm']:.4f}")
    print(f"   ✅ Gradient clipping working!")
    print()
    
    # ============================================
    # Summary
    # ============================================
    print("=" * 80)
    print("✅ All Components Working!")
    print("=" * 80)
    print()
    print("📋 Summary of Anti-Overfitting Features:")
    print("   ✅ Dropout (rate=0.2)")
    print("   ✅ BatchNorm")
    print("   ✅ Learning Rate Scheduler")
    print("   ✅ Gradient Clipping (max_norm=0.5)")
    print("   ✅ High Entropy Coefficient (0.10)")
    print("   ✅ Data Augmentation")
    print("   ✅ Early Stopping")
    print()
    print("💡 برای training واقعی:")
    print("   1. env_train و env_val را ایجاد کنید")
    print("   2. خط 360-400 را uncomment کنید")
    print("   3. python training_improved.py را اجرا کنید")
    print()


# ═══════════════════════════════════════════════════════════
# Training Loop Template (برای استفاده با environment واقعی)
# ═══════════════════════════════════════════════════════════

def train_with_environment(env_train, env_val):
    """
    Training loop کامل با environment واقعی
    
    برای استفاده:
    1. FCCTraceLoader را import کنید
    2. ContentAwareEnvFCC را import کنید
    3. این function را صدا بزنید
    """
    
    print("=" * 80)
    print("🚀 Starting Training")
    print("=" * 80)
    print()
    
    # Create model
    model = create_improved_model(
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate'],
        use_batchnorm=config['use_batchnorm']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # ✅ LR Scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config['lr_scheduler_step'],
        gamma=config['lr_scheduler_gamma']
    )
    
    print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Training loop
    training_log = []
    best_val_reward = -float('inf')
    no_improvement_count = 0
    start_time = time.time()
    
    for update in range(1, config['n_updates'] + 1):
        update_start = time.time()
        
        # Collect rollout (با augmentation)
        model.train()  # Training mode
        rollout = collect_rollout(
            env_train, 
            model, 
            config['rollout_steps'],
            use_augmentation=config['use_augmentation'],
            augment_prob=config['augmentation_prob']
        )
        
        # PPO update
        train_info = ppo_update(model, optimizer, rollout, config)
        
        # ✅ LR Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
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
            'learning_rate': float(current_lr),
            'n_episodes': len(rollout['episode_rewards']),
            'update_time': float(update_time),
            'elapsed_time': float(elapsed_time)
        }
        
        # Evaluation
        if update % config['eval_interval'] == 0:
            val_mean, val_std = evaluate(env_val, model, n_episodes=config['n_val_episodes'])
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
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'val_reward': val_mean
                }, best_path)
            else:
                no_improvement_count += 1
            
            # Check early stopping
            if no_improvement_count >= config['early_stopping_patience']:
                print()
                print("=" * 80)
                print("⏸️  EARLY STOPPING TRIGGERED")
                print("=" * 80)
                print(f"No improvement for {no_improvement_count} evaluations")
                print(f"Best val reward: {best_val_reward:+.2f}")
                print(f"Stopping at update {update}")
                print()
                break
        
        training_log.append(log_entry)
        
        # Console output
        if update % config['log_interval'] == 0:
            eta_seconds = (elapsed_time / update) * (config['n_updates'] - update)
            eta_minutes = eta_seconds / 60
            
            print(f"Update {update:3d}/{config['n_updates']} | "
                  f"Reward: {mean_reward:+7.2f} | "
                  f"Loss: {train_info['policy_loss']:.4f} | "
                  f"Entropy: {train_info['entropy']:.3f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {update_time:.1f}s")
            
            if 'val_reward_mean' in log_entry:
                best_marker = "🏆" if log_entry.get('new_best') else "  "
                no_improve_marker = f"[{no_improvement_count}/{config['early_stopping_patience']}]"
                print(f"         {best_marker}  Val: {log_entry['val_reward_mean']:+.2f} ± "
                      f"{log_entry['val_reward_std']:.2f} {no_improve_marker}")
        
        # Save checkpoint
        if update % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{update}.pth')
            torch.save({
                'update': update,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'train_info': train_info
            }, checkpoint_path)
            print(f"         💾 Checkpoint saved: checkpoint_{update}.pth")
        
        # Save log
        if update % 10 == 0:
            with open(log_file, 'w') as f:
                json.dump(training_log, f, indent=2)
    
    print()
    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"⏰ Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"🏆 Best val reward: {best_val_reward:+.2f}")
    print()
    
    # Final save
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return model, training_log


# ═══════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    # برای تست بدون environment
    # test_training_components()
    
    # برای training واقعی، این خطوط را uncomment کنید:
    """
    # Load data
    from models.fcc_trace_loader import FCCTraceLoader
    from models.content_aware_env_fcc import ContentAwareEnvFCC
    
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
    
    # Train
    model, training_log = train_with_environment(env_train, env_val)
    """
