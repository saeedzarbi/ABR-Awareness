"""
Balanced Training Script - Fixed Reward Function
Based on successful baseline with careful improvements
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import random
from datetime import datetime
import time
from collections import deque
from typing import Dict, List, Tuple

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ================================================================================
# CRITICAL FIX: Balanced Configuration
# ================================================================================

class BalancedConfig:
    """Fixed configuration based on what actually works"""
    
    # Core PPO (keep what worked)
    learning_rate: float = 3e-4  # Original LR that worked
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    
    # FIXED REWARD FUNCTION (most important change)
    rebuffer_penalty: float = 10.0  # Reduced from 100 to 10 (still 2x original)
    smoothness_penalty: float = 1.0  # Back to original
    use_pensieve_reward: bool = False  # Use improved but not extreme
    
    # Moderate regularization
    entropy_coef: float = 0.01  # Original value
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0  # No L2 for now
    
    # Training settings
    batch_size: int = 64
    ppo_epochs: int = 4
    rollout_steps: int = 2048
    n_updates: int = 300  # Stop earlier to avoid overfitting
    
    # Evaluation
    eval_interval: int = 10
    checkpoint_interval: int = 20
    log_interval: int = 5
    n_eval_episodes: int = 10
    
    # Early stopping
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.5
    best_checkpoint_threshold: float = 100.0  # Save if > 100
    
    # Directories
    checkpoint_dir: str = 'results/balanced_training'
    log_file: str = 'training_log.json'
    
    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_path = os.path.join(self.checkpoint_dir, self.log_file)


# ================================================================================
# Simple Model (without excessive regularization)
# ================================================================================

def create_simple_model():
    """Use the original model that worked"""
    from models.content_aware_model import create_content_aware_model
    return create_content_aware_model()


# ================================================================================
# Balanced Reward Function
# ================================================================================

class BalancedRewardFunction:
    """Balanced reward that doesn't destroy training"""
    
    def __init__(self, config: BalancedConfig):
        self.rebuffer_penalty = config.rebuffer_penalty
        self.smoothness_penalty = config.smoothness_penalty
        self.use_pensieve = config.use_pensieve_reward
        
    def compute_reward(self, vmaf: float, rebuffer_time: float,
                      last_bitrate: int, current_bitrate: int) -> float:
        """
        Balanced reward computation
        Not too harsh, not too lenient
        """
        
        if self.use_pensieve:
            # Pensieve-style (but with reasonable penalties)
            quality = vmaf / 10.0  # VMAF/10 gives 0-10 range
            rebuffer_penalty = self.rebuffer_penalty * rebuffer_time
            
            if last_bitrate > 0:
                smoothness = abs(current_bitrate - last_bitrate) / 1000.0
                smoothness_penalty = self.smoothness_penalty * smoothness
            else:
                smoothness_penalty = 0.0
                
            reward = quality - rebuffer_penalty - smoothness_penalty
            
        else:
            # Simple linear reward (often more stable)
            # Prioritize low rebuffering
            if rebuffer_time < 0.1:  # No rebuffering
                quality_bonus = vmaf / 100.0 * 5.0
                bitrate_bonus = current_bitrate / 6000.0 * 2.0
                reward = 10.0 + quality_bonus + bitrate_bonus
            elif rebuffer_time < 1.0:  # Minor rebuffering
                reward = 5.0 - rebuffer_time * 5.0
            else:  # Significant rebuffering
                reward = -rebuffer_time * 10.0
        
        # Clip extreme values
        reward = np.clip(reward, -50.0, 50.0)
        
        return reward


# ================================================================================
# Simplified Trainer
# ================================================================================

class SimpleTrainer:
    """Simplified trainer without complex features"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        self.reward_func = BalancedRewardFunction(config)
        
    def compute_gae(self, rewards, values, dones):
        """Standard GAE computation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def train_step(self, rollout_data):
        """Simple PPO update"""
        
        advantages, returns = self.compute_gae(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones']
        )
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_log_probs = torch.FloatTensor(rollout_data['log_probs'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Training loop
        n_samples = len(rollout_data['states'])
        losses = []
        
        for _ in range(self.config.ppo_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                # Prepare batch
                batch_states = [rollout_data['states'][i] for i in batch_idx]
                batch_actions = torch.LongTensor([rollout_data['actions'][i] for i in batch_idx])
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Stack states
                batch_net = torch.stack([torch.FloatTensor(s['network']) for s in batch_states])
                batch_cont = torch.stack([torch.FloatTensor(s['content']) for s in batch_states])
                batch_vmaf = torch.stack([torch.FloatTensor(s['vmaf']) for s in batch_states])
                
                # Forward pass
                action_probs, values = self.model(batch_net, batch_cont, batch_vmaf)
                
                # Compute losses
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                   1 + self.config.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_coef * value_loss - 
                       self.config.entropy_coef * entropy)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                losses.append(loss.item())
        
        return {
            'loss': np.mean(losses),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }


# ================================================================================
# Main Training Loop
# ================================================================================

def train_balanced():
    """Main training with balanced settings"""
    
    print("=" * 80)
    print(f"🔄 Balanced Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Configuration
    config = BalancedConfig()
    
    print("\n📋 Configuration:")
    print(f"  • Rebuffer Penalty: {config.rebuffer_penalty} (balanced)")
    print(f"  • Learning Rate: {config.learning_rate}")
    print(f"  • Max Updates: {config.n_updates}")
    print(f"  • Early Stopping: {config.early_stopping_patience} patience")
    print("=" * 80)
    
    # Model and trainer
    model = create_simple_model()
    trainer = SimpleTrainer(model, config)
    
    # Environment
    from models.content_aware_env_v2 import ContentAwareEnvV2
    env = ContentAwareEnvV2(use_real_traces=True)
    
    # Training variables
    training_log = []
    best_val_reward = -float('inf')
    no_improvement = 0
    start_time = time.time()
    
    # Training loop
    for update in range(1, config.n_updates + 1):
        
        # Collect rollout
        rollout = {
            'states': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': []
        }
        
        state = env.reset()
        episode_rewards = []
        current_episode = 0
        
        for step in range(config.rollout_steps):
            # Get action
            net = torch.FloatTensor(state['network']).unsqueeze(0)
            cont = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, value = model(net, cont, vmaf)
            
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Environment step
            next_state, env_reward, done, info = env.step(action.item())
            
            # Compute balanced reward
            reward = trainer.reward_func.compute_reward(
                vmaf=info.get('vmaf', 50),
                rebuffer_time=info['rebuffer_time'],
                last_bitrate=env.past_bitrates[-2] if len(env.past_bitrates) > 1 else 0,
                current_bitrate=info['bitrate']
            )
            
            # Store
            rollout['states'].append(state)
            rollout['actions'].append(action.item())
            rollout['rewards'].append(reward)
            rollout['values'].append(value.item())
            rollout['log_probs'].append(log_prob.item())
            rollout['dones'].append(done)
            
            current_episode += reward
            
            if done:
                episode_rewards.append(current_episode)
                current_episode = 0
                state = env.reset()
            else:
                state = next_state
        
        # Train
        train_info = trainer.train_step(rollout)
        
        # Logging
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        # Validation
        if update % config.eval_interval == 0:
            val_rewards = []
            val_rebuffer = []
            val_bitrates = []
            
            for _ in range(config.n_eval_episodes):
                state = env.reset(split='test')  # Use test split for validation
                episode_reward = 0
                episode_rebuffer = 0
                episode_bitrates = []
                done = False
                
                while not done:
                    net = torch.FloatTensor(state['network']).unsqueeze(0)
                    cont = torch.FloatTensor(state['content']).unsqueeze(0)
                    vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
                    
                    with torch.no_grad():
                        action_probs, _ = model(net, cont, vmaf)
                    
                    action = action_probs.argmax(dim=1).item()
                    state, reward, done, info = env.step(action)
                    
                    # Use same reward calculation
                    reward = trainer.reward_func.compute_reward(
                        vmaf=info.get('vmaf', 50),
                        rebuffer_time=info['rebuffer_time'],
                        last_bitrate=env.past_bitrates[-2] if len(env.past_bitrates) > 1 else 0,
                        current_bitrate=info['bitrate']
                    )
                    
                    episode_reward += reward
                    episode_rebuffer += info['rebuffer_time']
                    episode_bitrates.append(info['bitrate'])
                
                val_rewards.append(episode_reward)
                val_rebuffer.append(episode_rebuffer)
                val_bitrates.append(np.mean(episode_bitrates))
            
            val_mean = np.mean(val_rewards)
            val_std = np.std(val_rewards)
            val_rebuf = np.mean(val_rebuffer)
            val_bitrate = np.mean(val_bitrates)
            
            # Check improvement
            improvement = val_mean - best_val_reward
            if improvement > config.early_stopping_min_delta:
                best_val_reward = val_mean
                no_improvement = 0
                
                # Save if good
                if val_mean > config.best_checkpoint_threshold:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'update': update,
                        'val_reward': val_mean,
                        'val_rebuffer': val_rebuf
                    }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
                    
                    print(f"  🏆 New best model! Reward: {val_mean:+.2f}, Rebuffer: {val_rebuf:.2f}s")
            else:
                no_improvement += 1
            
            # Log
            progress = (update / config.n_updates) * 100
            print(f"[{progress:5.1f}%] Update {update:3d} | "
                  f"Train: {mean_reward:+7.2f} | "
                  f"Val: {val_mean:+7.2f} (σ={val_std:.1f}) | "
                  f"Rebuf: {val_rebuf:.2f}s | "
                  f"BR: {val_bitrate:.0f}")
            
            # Early stopping
            if no_improvement >= config.early_stopping_patience:
                print(f"\n⏸️  Early stopping at update {update}")
                print(f"   Best reward: {best_val_reward:+.2f}")
                break
                
            # Success condition
            if val_mean > 110 and val_rebuf < 2.0:
                print(f"\n🎯 Target reached! Reward: {val_mean:+.2f}, Rebuffer: {val_rebuf:.2f}s")
                break
        
        # Regular checkpoint
        if update % config.checkpoint_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'update': update,
                'mean_reward': mean_reward
            }, os.path.join(config.checkpoint_dir, f'checkpoint_{update}.pth'))
            
            # Save log
            training_log.append({
                'update': update,
                'mean_reward': float(mean_reward),
                'timestamp': datetime.now().isoformat()
            })
            
            with open(config.log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
    
    # Complete
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 80)
    print(f"✅ Training Complete!")
    print(f"   Time: {total_time:.1f} minutes")
    print(f"   Best Reward: {best_val_reward:+.2f}")
    print(f"   Saved to: {config.checkpoint_dir}")
    print("=" * 80)
    
    return model


if __name__ == '__main__':
    try:
        model = train_balanced()
        print("\n🎯 Balanced training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()