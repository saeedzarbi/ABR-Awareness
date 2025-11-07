"""
Advanced Training Script for Content-Aware ABR
With improved regularization, reward tuning, and comprehensive monitoring
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
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ================================================================================
# Enhanced Configuration with Regularization
# ================================================================================

class Config:
    """Enhanced training configuration with anti-overfitting measures"""
    
    # Core PPO parameters
    learning_rate: float = 1e-4  # Reduced from 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    
    # Regularization (CRITICAL FOR OVERFITTING)
    entropy_coef: float = 0.1  # Increased from 0.05 for exploration
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    weight_decay: float = 1e-5  # L2 regularization
    dropout_rate: float = 0.1  # NEW: Dropout for regularization
    
    # Training schedule
    batch_size: int = 128  # Increased for stability
    ppo_epochs: int = 3  # Reduced from 4 to prevent overfitting
    rollout_steps: int = 2048
    n_updates: int = 400
    
    # Learning rate schedule
    use_lr_decay: bool = True
    lr_decay_rate: float = 0.95
    lr_decay_interval: int = 50
    min_lr: float = 1e-5
    
    # Reward function tuning (based on ablation study)
    rebuffer_penalty: float = 100.0  # INCREASED from 4.3
    smoothness_penalty: float = 2.0  # INCREASED from 1.0
    use_vmaf_bonus: bool = True
    vmaf_bonus_weight: float = 0.5
    
    # Evaluation and checkpointing
    eval_interval: int = 10
    checkpoint_interval: int = 20
    log_interval: int = 5
    n_eval_episodes: int = 20  # More episodes for stable evaluation
    
    # Early stopping with patience
    early_stopping_patience: int = 50  # Increased patience
    early_stopping_min_delta: float = 1.0
    target_reward: float = 120.0  # Stop if reached
    
    # Advanced features
    use_curriculum: bool = True  # Start with easier traces
    use_reward_normalization: bool = True
    use_gradient_monitoring: bool = True
    
    # Directories
    checkpoint_dir: str = 'results/advanced_training'
    log_file: str = 'training_log_advanced.json'
    
    def __init__(self):
        """Create directories and setup paths"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_path = os.path.join(self.checkpoint_dir, self.log_file)


# ================================================================================
# Enhanced Model with Regularization
# ================================================================================

class RegularizedContentAwareActor(nn.Module):
    """Content-aware model with dropout and batch normalization"""
    
    def __init__(self, state_dim=(6, 8), action_dim=6, content_dim=2, dropout_rate=0.1):
        super().__init__()
        
        # Network state encoder
        self.conv1 = nn.Conv1d(state_dim[0], 128, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        conv_out_size = 128 * 2
        
        # Content encoder with regularization
        self.content_fc1 = nn.Linear(content_dim, 32)
        self.content_bn1 = nn.BatchNorm1d(32)
        self.content_dropout = nn.Dropout(dropout_rate)
        self.content_fc2 = nn.Linear(32, 64)
        
        # VMAF encoder
        self.vmaf_fc = nn.Linear(action_dim, 32)
        self.vmaf_bn = nn.BatchNorm1d(32)
        
        # Fusion layer with attention mechanism
        fusion_input = conv_out_size + 64 + 32
        self.attention = nn.Linear(fusion_input, fusion_input)
        self.fusion_fc = nn.Linear(fusion_input, 128)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        
        # Output heads
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, network_state, content_features, vmaf_predictions, training=True):
        """Forward pass with optional training mode for dropout"""
        
        # Network encoding
        x = F.relu(self.bn1(self.conv1(network_state)))
        if training:
            x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        if training:
            x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        
        # Content encoding
        c = F.relu(self.content_bn1(self.content_fc1(content_features)))
        if training:
            c = self.content_dropout(c)
        c = F.relu(self.content_fc2(c))
        
        # VMAF encoding
        v = F.relu(self.vmaf_bn(self.vmaf_fc(vmaf_predictions)))
        
        # Fusion with attention
        combined = torch.cat([x, c, v], dim=1)
        attention_weights = torch.sigmoid(self.attention(combined))
        combined = combined * attention_weights  # Element-wise attention
        
        fused = F.relu(self.fusion_fc(combined))
        if training:
            fused = self.fusion_dropout(fused)
        
        # Output heads
        action_logits = self.actor_head(fused)
        action_prob = F.softmax(action_logits, dim=1)
        state_value = self.critic_head(fused)
        
        return action_prob, state_value


# ================================================================================
# Enhanced Reward Function
# ================================================================================

class EnhancedRewardFunction:
    """Improved reward function based on ablation study findings"""
    
    def __init__(self, config: Config):
        self.rebuffer_penalty = config.rebuffer_penalty
        self.smoothness_penalty = config.smoothness_penalty
        self.use_vmaf_bonus = config.use_vmaf_bonus
        self.vmaf_bonus_weight = config.vmaf_bonus_weight
        
        # Reward normalization statistics
        self.reward_history = deque(maxlen=1000)
        self.reward_mean = 0.0
        self.reward_std = 1.0
    
    def compute_reward(self, vmaf: float, rebuffer_time: float, 
                      last_bitrate: int, current_bitrate: int) -> float:
        """
        Enhanced reward computation with VMAF bonus
        
        Based on ablation study: higher rebuffering penalty is critical
        """
        # Base quality reward (VMAF normalized to 0-1)
        quality_reward = vmaf / 100.0 * 10.0  # Scale to reasonable range
        
        # Heavy rebuffering penalty (CRITICAL)
        rebuffer_penalty = self.rebuffer_penalty * rebuffer_time
        
        # Smoothness penalty (bitrate changes)
        if last_bitrate > 0:
            smoothness = abs(current_bitrate - last_bitrate) / 1000.0
            smoothness_penalty = self.smoothness_penalty * smoothness
        else:
            smoothness_penalty = 0.0
        
        # VMAF bonus for high quality (encourage quality when stable)
        vmaf_bonus = 0.0
        if self.use_vmaf_bonus and rebuffer_time == 0:
            if vmaf > 80:
                vmaf_bonus = (vmaf - 80) / 20.0 * self.vmaf_bonus_weight
        
        # Total reward
        reward = quality_reward - rebuffer_penalty - smoothness_penalty + vmaf_bonus
        
        # Update statistics for normalization
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-8
        
        return reward
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward for stable training"""
        if len(self.reward_history) > 100:
            return (reward - self.reward_mean) / self.reward_std
        return reward


# ================================================================================
# Training Utilities
# ================================================================================

class RolloutBuffer:
    """Efficient rollout storage with automatic reset"""
    
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def add(self, state, action, reward, value, log_prob, done):
        """Add transition to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)


# ================================================================================
# PPO Trainer with Advanced Features
# ================================================================================

class PPOTrainer:
    """Advanced PPO trainer with monitoring and regularization"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.use_lr_decay:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.lr_decay_interval,
                gamma=config.lr_decay_rate
            )
        
        # Reward function
        self.reward_func = EnhancedRewardFunction(config)
        
        # Monitoring
        self.gradient_norms = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
    
    def train_step(self, rollout: RolloutBuffer) -> Dict:
        """Single PPO update"""
        
        # Compute returns and advantages
        returns, advantages = rollout.compute_returns_and_advantages(
            self.config.gamma, self.config.gae_lambda
        )
        
        # Convert to tensors
        old_log_probs = torch.FloatTensor(rollout.log_probs)
        
        # Training metrics
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_grad_norms = []
        
        n_samples = len(rollout.states)
        
        for epoch in range(self.config.ppo_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                # Prepare batch
                batch_states = [rollout.states[i] for i in batch_idx]
                batch_actions = torch.LongTensor([rollout.actions[i] for i in batch_idx])
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Stack state components
                batch_net = torch.stack([torch.FloatTensor(s['network']) for s in batch_states])
                batch_cont = torch.stack([torch.FloatTensor(s['content']) for s in batch_states])
                batch_vmaf = torch.stack([torch.FloatTensor(s['vmaf']) for s in batch_states])
                
                # Forward pass (with dropout in training mode)
                action_probs, values = self.model(
                    batch_net, batch_cont, batch_vmaf, training=True
                )
                
                # Compute losses
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_epsilon, 
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                value_pred_clipped = batch_returns + torch.clamp(
                    values.squeeze() - batch_returns,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon
                )
                value_losses_1 = F.mse_loss(values.squeeze(), batch_returns)
                value_losses_2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = torch.max(value_losses_1, value_losses_2)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config.value_coef * value_loss - 
                    self.config.entropy_coef * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Record metrics
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())
                epoch_grad_norms.append(grad_norm.item())
        
        # Update scheduler
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        # Store metrics
        self.policy_losses.extend(epoch_policy_losses)
        self.value_losses.extend(epoch_value_losses)
        self.entropies.extend(epoch_entropies)
        self.gradient_norms.extend(epoch_grad_norms)
        
        return {
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses),
            'entropy': np.mean(epoch_entropies),
            'grad_norm': np.mean(epoch_grad_norms),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }


# ================================================================================
# Environment Manager with Curriculum Learning
# ================================================================================

class EnvironmentManager:
    """Manages training and validation environments with curriculum"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Import environment components
        from models.content_aware_env_fcc import ContentAwareEnvFCC
        from models.fcc_trace_loader import FCCTraceLoader
        
        # Initialize trace loader
        self.loader = FCCTraceLoader(
            fcc_trace_dir='data/network_traces/fcc',
            train_file='data/network_traces/fcc/splits/fcc_train.txt',
            val_file='data/network_traces/fcc/splits/fcc_val.txt',
            test_file='data/network_traces/fcc/splits/fcc_test.txt'
        )
        
        # Create environments
        self.env_train = ContentAwareEnvFCC(
            fcc_trace_loader=self.loader,
            features_file='data/features/si_ti_features.json',
            vmaf_file='data/vmaf/vmaf_table.json',
            video_dir='data/videos',
            mode='train'
        )
        
        self.env_val = ContentAwareEnvFCC(
            fcc_trace_loader=self.loader,
            features_file='data/features/si_ti_features.json',
            vmaf_file='data/vmaf/vmaf_table.json',
            video_dir='data/videos',
            mode='val'
        )
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.curriculum_thresholds = [50, 100, 150, 200]  # Updates for each stage
    
    def get_training_env(self, update: int):
        """Get training environment with optional curriculum"""
        if self.config.use_curriculum:
            # Progress through curriculum stages
            for i, threshold in enumerate(self.curriculum_thresholds):
                if update < threshold:
                    self.curriculum_stage = i
                    break
            else:
                self.curriculum_stage = len(self.curriculum_thresholds)
            
            # Modify environment difficulty based on stage
            # (This would require environment modifications in practice)
            # For now, we just use the standard environment
        
        return self.env_train
    
    def get_validation_env(self):
        """Get validation environment"""
        return self.env_val


# ================================================================================
# Training Loop with Monitoring
# ================================================================================

def train_advanced():
    """Main training function with all improvements"""
    
    # Initialize configuration
    config = Config()
    
    # Create model with regularization
    model = RegularizedContentAwareActor(dropout_rate=config.dropout_rate)
    
    # Initialize trainer
    trainer = PPOTrainer(model, config)
    
    # Initialize environment manager
    env_manager = EnvironmentManager(config)
    
    # Training metrics
    training_log = []
    best_val_reward = -float('inf')
    no_improvement_count = 0
    
    # Timing
    start_time = time.time()
    
    print("=" * 80)
    print(f"🚀 Advanced Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Checkpoint Directory: {config.checkpoint_dir}")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  • Learning Rate: {config.learning_rate}")
    print(f"  • Rebuffer Penalty: {config.rebuffer_penalty}")
    print(f"  • Entropy Coefficient: {config.entropy_coef}")
    print(f"  • Dropout Rate: {config.dropout_rate}")
    print(f"  • Weight Decay: {config.weight_decay}")
    print(f"  • Curriculum Learning: {config.use_curriculum}")
    print("=" * 80)
    
    # Training loop
    for update in range(1, config.n_updates + 1):
        update_start = time.time()
        
        # Get training environment (with curriculum)
        env = env_manager.get_training_env(update)
        
        # Collect rollout
        rollout = RolloutBuffer(config.rollout_steps)
        state = env.reset()
        
        model.eval()  # Disable dropout for rollout collection
        for step in range(config.rollout_steps):
            # Prepare state tensors
            net = torch.FloatTensor(state['network']).unsqueeze(0)
            cont = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action_probs, value = model(net, cont, vmaf, training=False)
            
            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Environment step
            next_state, reward, done, info = env.step(action.item())
            
            # Enhanced reward with new function
            reward = trainer.reward_func.compute_reward(
                vmaf=info.get('vmaf', 50),
                rebuffer_time=info['rebuffer_time'],
                last_bitrate=env.past_bitrates[-2] if len(env.past_bitrates) > 1 else 0,
                current_bitrate=info['bitrate']
            )
            
            # Normalize reward if enabled
            if config.use_reward_normalization:
                reward = trainer.reward_func.normalize_reward(reward)
            
            # Store transition
            rollout.add(state, action.item(), reward, value.item(), log_prob.item(), done)
            
            # Update state
            state = next_state if not done else env.reset()
        
        # PPO update
        model.train()  # Enable dropout for training
        train_info = trainer.train_step(rollout)
        
        # Calculate metrics
        mean_reward = np.mean(rollout.episode_rewards) if rollout.episode_rewards else 0
        mean_length = np.mean(rollout.episode_lengths) if rollout.episode_lengths else 0
        update_time = time.time() - update_start
        
        # Create log entry
        log_entry = {
            'update': update,
            'timestamp': datetime.now().isoformat(),
            'mean_reward': float(mean_reward),
            'mean_episode_length': float(mean_length),
            'n_episodes': len(rollout.episode_rewards),
            'policy_loss': float(train_info['policy_loss']),
            'value_loss': float(train_info['value_loss']),
            'entropy': float(train_info['entropy']),
            'grad_norm': float(train_info['grad_norm']),
            'learning_rate': float(train_info['learning_rate']),
            'update_time': float(update_time),
            'total_time': float(time.time() - start_time),
            'curriculum_stage': env_manager.curriculum_stage if config.use_curriculum else 0
        }
        
        # Validation
        if update % config.eval_interval == 0:
            model.eval()
            val_env = env_manager.get_validation_env()
            val_rewards = []
            val_rebuffer = []
            val_bitrates = []
            
            for episode in range(config.n_eval_episodes):
                state = val_env.reset()
                episode_reward = 0
                episode_rebuffer = 0
                episode_bitrates = []
                done = False
                
                while not done:
                    net = torch.FloatTensor(state['network']).unsqueeze(0)
                    cont = torch.FloatTensor(state['content']).unsqueeze(0)
                    vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
                    
                    with torch.no_grad():
                        action_probs, _ = model(net, cont, vmaf, training=False)
                    
                    # Greedy action for evaluation
                    action = action_probs.argmax(dim=1).item()
                    
                    state, reward, done, info = val_env.step(action)
                    
                    # Use same reward computation as training
                    reward = trainer.reward_func.compute_reward(
                        vmaf=info.get('vmaf', 50),
                        rebuffer_time=info['rebuffer_time'],
                        last_bitrate=val_env.past_bitrates[-2] if len(val_env.past_bitrates) > 1 else 0,
                        current_bitrate=info['bitrate']
                    )
                    
                    episode_reward += reward
                    episode_rebuffer += info['rebuffer_time']
                    episode_bitrates.append(info['bitrate'])
                
                val_rewards.append(episode_reward)
                val_rebuffer.append(episode_rebuffer)
                val_bitrates.append(np.mean(episode_bitrates))
            
            # Validation statistics
            val_mean_reward = np.mean(val_rewards)
            val_std_reward = np.std(val_rewards)
            val_mean_rebuffer = np.mean(val_rebuffer)
            val_mean_bitrate = np.mean(val_bitrates)
            
            log_entry.update({
                'val_reward_mean': float(val_mean_reward),
                'val_reward_std': float(val_std_reward),
                'val_rebuffer_mean': float(val_mean_rebuffer),
                'val_bitrate_mean': float(val_mean_bitrate)
            })
            
            # Check for improvement
            improvement = val_mean_reward - best_val_reward
            
            if improvement > config.early_stopping_min_delta:
                best_val_reward = val_mean_reward
                no_improvement_count = 0
                log_entry['new_best'] = True
                
                # Save best model
                best_path = os.path.join(config.checkpoint_dir, 'checkpoint_best.pth')
                torch.save({
                    'update': update,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'config': config.__dict__,
                    'val_reward': val_mean_reward,
                    'val_rebuffer': val_mean_rebuffer,
                    'val_bitrate': val_mean_bitrate
                }, best_path)
                
                print(f"  🏆 New best model! Reward: {val_mean_reward:+.2f}")
            else:
                no_improvement_count += 1
            
            # Early stopping check
            if no_improvement_count >= config.early_stopping_patience:
                print(f"\n⏸️  Early stopping at update {update}")
                print(f"   Best validation reward: {best_val_reward:+.2f}")
                break
            
            # Target reward check
            if val_mean_reward >= config.target_reward:
                print(f"\n🎯 Target reward reached! ({val_mean_reward:+.2f} >= {config.target_reward})")
                break
        
        # Append to log
        training_log.append(log_entry)
        
        # Console output
        if update % config.log_interval == 0:
            progress = (update / config.n_updates) * 100
            marker = "🏆" if log_entry.get('new_best', False) else "  "
            
            output = f"[{progress:5.1f}%] Update {update:3d} | "
            output += f"Reward: {mean_reward:+7.2f} | "
            
            if 'val_reward_mean' in log_entry:
                output += f"{marker} Val: {log_entry['val_reward_mean']:+7.2f} "
                output += f"(σ={log_entry['val_reward_std']:.1f}) | "
                output += f"Rebuf: {log_entry['val_rebuffer_mean']:.2f}s | "
                output += f"BR: {log_entry['val_bitrate_mean']:.0f}"
            
            output += f" | LR: {train_info['learning_rate']:.2e}"
            
            print(output)
        
        # Regular checkpoint
        if update % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_{update}.pth')
            torch.save({
                'update': update,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config.__dict__,
                'train_info': train_info,
                'training_log': training_log[-100:]  # Last 100 entries
            }, checkpoint_path)
        
        # Save log periodically
        if update % 10 == 0:
            with open(config.log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
    
    # Final save
    with open(config.log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Training complete
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 80)
    print(f"✅ Training Complete!")
    print(f"   Total Time: {total_time:.1f} minutes")
    print(f"   Best Validation Reward: {best_val_reward:+.2f}")
    print(f"   Final Update: {update}")
    print(f"   Checkpoints saved to: {config.checkpoint_dir}")
    print("=" * 80)
    
    # Plot training curves
    plot_training_curves(training_log, config.checkpoint_dir)
    
    return model, training_log


def plot_training_curves(training_log: List[Dict], save_dir: str):
    """Generate and save training curves"""
    
    # Extract data
    updates = [entry['update'] for entry in training_log]
    rewards = [entry['mean_reward'] for entry in training_log]
    
    val_updates = [entry['update'] for entry in training_log if 'val_reward_mean' in entry]
    val_rewards = [entry['val_reward_mean'] for entry in training_log if 'val_reward_mean' in entry]
    val_rebuffer = [entry['val_rebuffer_mean'] for entry in training_log if 'val_rebuffer_mean' in entry]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training reward
    axes[0, 0].plot(updates, rewards, label='Training', alpha=0.7)
    if val_updates:
        axes[0, 0].plot(val_updates, val_rewards, label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Update')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rebuffering
    if val_rebuffer:
        axes[0, 1].plot(val_updates, val_rebuffer, color='red', linewidth=2)
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Rebuffering (s)')
        axes[0, 1].set_title('Validation Rebuffering')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    policy_losses = [entry['policy_loss'] for entry in training_log if 'policy_loss' in entry]
    value_losses = [entry['value_loss'] for entry in training_log if 'value_loss' in entry]
    loss_updates = [entry['update'] for entry in training_log if 'policy_loss' in entry]
    
    if policy_losses:
        axes[1, 0].plot(loss_updates, policy_losses, label='Policy Loss', alpha=0.7)
        axes[1, 0].plot(loss_updates, value_losses, label='Value Loss', alpha=0.7)
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Entropy
    entropies = [entry['entropy'] for entry in training_log if 'entropy' in entry]
    if entropies:
        axes[1, 1].plot(loss_updates, entropies, color='green', linewidth=2)
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"\n📊 Training curves saved to {save_dir}/training_curves.png")


# ================================================================================
# Main Execution
# ================================================================================

if __name__ == '__main__':
    try:
        model, log = train_advanced()
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()