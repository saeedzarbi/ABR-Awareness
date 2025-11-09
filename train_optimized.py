"""
Optimized Training Script for Content-Aware ABR - FIXED VERSION
Key fixes for failed training run:
1. INCREASED exploration (higher entropy)
2. Slower entropy decay
3. More patient early stopping
4. Better learning rate schedule
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer


class OptimizedConfig:
    """
    Fixed configuration - addresses poor exploration issue
    """
    
    # Data paths
    fcc_trace_dir: str = 'data/fcc_traces'
    train_split: str = 'data/network_traces/fcc/splits/fcc_train.txt'
    val_split: str = 'data/network_traces/fcc/splits/fcc_val.txt'
    test_split: str = 'data/network_traces/fcc/splits/fcc_test.txt'
    features_file: str = 'data/features/si_ti_features.json'
    vmaf_file: str = 'data/vmaf/vmaf_table.json'
    video_dir: str = 'data/videos'
    
    # PPO hyperparameters - FIXED FOR BETTER EXPLORATION
    total_timesteps: int = 600_000
    rollout_steps: int = 2048
    batch_size: int = 32  # Smaller for more frequent updates
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    
    # CRITICAL: Much higher entropy for exploration
    entropy_coef: float = 0.05  # 5x higher than before
    entropy_decay: float = 0.998  # Much slower decay
    entropy_min: float = 0.005  # Higher floor
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Learning rate schedule - slower decay
    use_lr_schedule: bool = True
    lr_decay_rate: float = 0.995  # Slower than 0.99
    lr_decay_interval: int = 30  # Less frequent (was 20)
    lr_min: float = 1e-5
    
    # Training control
    target_update: int = 240
    max_updates: int = 300
    eval_interval: int = 10
    checkpoint_interval: int = 20
    log_interval: int = 5
    n_eval_episodes: int = 10
    
    # More patient early stopping
    early_stopping_patience: int = 30  # Was 20
    early_stopping_min_delta: float = 2.0  # Higher threshold
    target_reward: float = 100.0
    target_rebuffer: float = 2.0  # More realistic
    
    # Output
    output_dir: str = 'results/optimized_training_v2'
    run_name: str = f'fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class SimpleLogger:
    """Lightweight logger for training"""
    
    def __init__(self, log_dir: str, run_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'{run_name}.jsonl')
        self.metrics_history = []
    
    def log(self, update: int, metrics: Dict):
        """Log metrics"""
        entry = {
            'update': update,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(entry)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def print_progress(self, update: int, metrics: Dict):
        """Print training progress"""
        reward_20 = metrics.get('reward_20', 0)
        reward_100 = metrics.get('reward_100', 0)
        
        print(f"Update {update:3d} | "
              f"R(20)={reward_20:+7.2f} | "
              f"R(100)={reward_100:+7.2f} | "
              f"LR={metrics.get('lr', 0):.2e} | "
              f"Ent={metrics.get('entropy', 0):.4f}")


def create_environment(config: OptimizedConfig, mode: str):
    """Create environment with proper configuration"""
    
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir=config.fcc_trace_dir,
        train_file=config.train_split,
        val_file=config.val_split,
        test_file=config.test_split
    )
    
    env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file=config.features_file,
        vmaf_file=config.vmaf_file,
        video_dir=config.video_dir,
        mode=mode
    )
    
    return env, fcc_loader


def evaluate_on_validation(
    model: ContentAwareActor,
    val_env: ContentAwareEnvFCC,
    n_episodes: int = 10,
    device: str = 'cpu'
) -> Dict:
    """Evaluate model on validation set"""
    
    episode_results = []
    
    for ep in range(n_episodes):
        state = val_env.reset()
        
        ep_reward = 0
        ep_rebuffer = 0
        ep_vmafs = []
        ep_bitrates = []
        done = False
        
        while not done:
            with torch.no_grad():
                net = torch.FloatTensor(state['network']).unsqueeze(0).to(device)
                cont = torch.FloatTensor(state['content']).unsqueeze(0).to(device)
                vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(device)
                
                action_probs, _ = model(net, cont, vmaf)
                action = action_probs.argmax(dim=1).item()
            
            state, reward, done, info = val_env.step(action)
            
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_vmafs.append(info.get('vmaf', 0))
            ep_bitrates.append(info['bitrate'])
        
        episode_results.append({
            'reward': ep_reward,
            'rebuffer': ep_rebuffer,
            'vmaf': np.mean(ep_vmafs),
            'bitrate': np.mean(ep_bitrates)
        })
    
    return {
        'mean_reward': np.mean([r['reward'] for r in episode_results]),
        'std_reward': np.std([r['reward'] for r in episode_results]),
        'mean_rebuffer': np.mean([r['rebuffer'] for r in episode_results]),
        'mean_vmaf': np.mean([r['vmaf'] for r in episode_results]),
        'mean_bitrate': np.mean([r['bitrate'] for r in episode_results])
    }


def train_optimized():
    """Main training function"""
    
    print("="*80)
    print("🚀 FIXED TRAINING: Content-Aware ABR v2")
    print("="*80)
    print(f"   FIXES: Higher entropy + Slower decay + Better exploration")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Configuration
    config = OptimizedConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Batch size: {config.batch_size} (smaller = more updates)")
    print(f"   Learning rate: {config.learning_rate} → {config.lr_min}")
    print(f"   Entropy: {config.entropy_coef} → {config.entropy_min} (HIGHER for exploration)")
    print(f"   Entropy decay: {config.entropy_decay} (SLOWER)")
    print(f"   Early stopping patience: {config.early_stopping_patience}")
    
    # Create environments
    print(f"\n🏗️  Creating Environments...")
    train_env, fcc_loader = create_environment(config, mode='train')
    val_env, _ = create_environment(config, mode='val')
    print(f"   ✅ Environments ready")
    
    # Create model
    print(f"\n🧠 Creating Model...")
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Create trainer
    print(f"\n🎓 Creating PPO Trainer...")
    trainer = PPOTrainer(
        model=model,
        env=train_env,
        lr=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size
    )
    
    # Logger
    logger = SimpleLogger(
        log_dir=os.path.join(config.output_dir, 'logs'),
        run_name=config.run_name
    )
    
    # Training state
    best_val_reward = float('-inf')
    no_improvement_count = 0
    update_count = 0
    timestep = 0
    current_entropy = config.entropy_coef
    current_lr = config.learning_rate
    
    print(f"\n" + "="*80)
    print("🚂 TRAINING LOOP")
    print("="*80)
    
    while update_count < config.max_updates and timestep < config.total_timesteps:
        
        # Collect rollout
        rollout = trainer.collect_rollout(n_steps=config.rollout_steps)
        timestep += len(rollout)
        
        # Update policy
        train_info = trainer.update_policy(rollout)
        update_count += 1
        
        # Learning rate decay (slower)
        if config.use_lr_schedule and update_count % config.lr_decay_interval == 0:
            current_lr = max(current_lr * config.lr_decay_rate, config.lr_min)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # Entropy decay (much slower)
        current_entropy = max(current_entropy * config.entropy_decay, config.entropy_min)
        trainer.entropy_coef = current_entropy
        
        # Logging
        if update_count % config.log_interval == 0:
            recent_20 = trainer.episode_rewards[-20:] if len(trainer.episode_rewards) >= 20 else trainer.episode_rewards
            recent_100 = trainer.episode_rewards[-100:] if len(trainer.episode_rewards) >= 100 else trainer.episode_rewards
            
            metrics = {
                'reward_20': np.mean(recent_20) if recent_20 else 0,
                'reward_100': np.mean(recent_100) if recent_100 else 0,
                'lr': current_lr,
                'entropy': current_entropy,
                'policy_loss': train_info['policy_loss'],
                'value_loss': train_info['value_loss']
            }
            
            logger.log(update_count, metrics)
            logger.print_progress(update_count, metrics)
        
        # Evaluation
        if update_count % config.eval_interval == 0:
            print(f"\n{'─'*80}")
            print(f"📊 EVALUATION at Update {update_count}")
            print(f"{'─'*80}")
            
            eval_results = evaluate_on_validation(model, val_env, config.n_eval_episodes, device)
            
            print(f"   Reward:    {eval_results['mean_reward']:+.2f} (σ={eval_results['std_reward']:.2f})")
            print(f"   Rebuffer:  {eval_results['mean_rebuffer']:.2f}s")
            print(f"   VMAF:      {eval_results['mean_vmaf']:.1f}")
            print(f"   Bitrate:   {eval_results['mean_bitrate']:.0f} kbps")
            
            # Check improvement
            improvement = eval_results['mean_reward'] - best_val_reward
            
            if improvement > config.early_stopping_min_delta:
                best_val_reward = eval_results['mean_reward']
                no_improvement_count = 0
                
                # Save best model
                best_path = os.path.join(config.output_dir, 'best_model.pth')
                torch.save({
                    'update': update_count,
                    'timestep': timestep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'reward': eval_results['mean_reward'],
                    'rebuffer': eval_results['mean_rebuffer'],
                    'vmaf': eval_results['mean_vmaf'],
                    'config': vars(config)
                }, best_path)
                
                print(f"\n   🏆 New best! Saved to best_model.pth")
            else:
                no_improvement_count += 1
                print(f"   ⚠️  No improvement ({no_improvement_count}/{config.early_stopping_patience})")
            
            # Target reached?
            if (eval_results['mean_reward'] > config.target_reward and 
                eval_results['mean_rebuffer'] < config.target_rebuffer):
                print(f"\n   🎯 TARGET REACHED!")
                print(f"      Stopping at update {update_count}")
                break
            
            # Early stopping
            if no_improvement_count >= config.early_stopping_patience:
                print(f"\n   ⏸️  Early stopping triggered")
                print(f"      Best reward: {best_val_reward:+.2f}")
                break
        
        # Regular checkpoint
        if update_count % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f'checkpoint_{update_count}.pth')
            torch.save({
                'update': update_count,
                'timestep': timestep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_reward': best_val_reward
            }, ckpt_path)
            print(f"\n   💾 Checkpoint: checkpoint_{update_count}.pth")
        
        # Target update reached
        if update_count >= config.target_update:
            print(f"\n   ✅ Target update {config.target_update} reached")
            print(f"      Stopping to avoid overfitting")
            break
    
    # Final evaluation
    print(f"\n" + "="*80)
    print("📊 FINAL EVALUATION")
    print("="*80)
    
    final_results = evaluate_on_validation(model, val_env, n_episodes=20, device=device)
    
    print(f"\n   Final Performance (20 episodes):")
    print(f"   • Reward:    {final_results['mean_reward']:+.2f} ± {final_results['std_reward']:.2f}")
    print(f"   • Rebuffer:  {final_results['mean_rebuffer']:.2f}s")
    print(f"   • VMAF:      {final_results['mean_vmaf']:.1f}")
    print(f"   • Bitrate:   {final_results['mean_bitrate']:.0f} kbps")
    
    # Save final
    final_path = os.path.join(config.output_dir, f'final_model_{update_count}.pth')
    torch.save({
        'update': update_count,
        'model_state_dict': model.state_dict(),
        'final_results': final_results,
        'config': vars(config)
    }, final_path)
    
    print(f"\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"   Total updates: {update_count}")
    print(f"   Total timesteps: {timestep:,}")
    print(f"   Best val reward: {best_val_reward:+.2f}")
    print(f"   Final reward: {final_results['mean_reward']:+.2f}")
    print(f"   Saved to: {config.output_dir}")
    print("="*80)
    
    return model, final_results


if __name__ == '__main__':
    try:
        model, results = train_optimized()
        
        print(f"\n🎉 Success!")
        print(f"   Best validation reward: {results['mean_reward']:+.2f}")
        print(f"   Target: +116.46 (checkpoint_100)")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()