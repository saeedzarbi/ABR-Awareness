"""
Advanced Training Script for Content-Aware ABR System
Combines modular architecture with balanced hyperparameters
Target: IEEE TCSVT journal submission
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger


class AdvancedTrainingConfig:
    """
    Optimized configuration for content-aware ABR training
    Based on empirical findings from ablation studies
    """
    
    # Network traces
    fcc_trace_dir: str = 'data/fcc_traces'
    train_split: str = 'data/network_traces/fcc/splits/fcc_train.txt'
    val_split: str = 'data/network_traces/fcc/splits/fcc_val.txt'
    test_split: str = 'data/network_traces/fcc/splits/fcc_test.txt'
    
    # Video features
    features_file: str = 'data/features/si_ti_features.json'
    vmaf_file: str = 'data/vmaf/vmaf_table.json'
    video_dir: str = 'data/videos'
    
    # PPO hyperparameters (balanced configuration)
    total_timesteps: int = 1_000_000
    rollout_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    
    # Regularization (based on ablation study insights)
    entropy_coef: float = 0.01  # Moderate exploration
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0
    
    # Reward function tuning (critical based on ablation)
    rebuffer_penalty: float = 10.0  # Balanced penalty
    smoothness_penalty: float = 1.0
    use_vmaf_reward: bool = True  # Content-aware QoE
    
    # Training control
    max_updates: int = 300  # Early stop to prevent overfitting
    eval_interval: int = 10
    checkpoint_interval: int = 20
    log_interval: int = 5
    n_eval_episodes: int = 10
    
    # Early stopping
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.5
    best_reward_threshold: float = 100.0
    
    # Output directories
    output_dir: str = 'results/advanced_trainings'
    log_dir: str = 'results/logs'
    run_name: str = f'advanced_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


def create_environments(config: AdvancedTrainingConfig) -> tuple:
    """
    Create training and validation environments with FCC traces
    
    Returns:
        tuple: (train_env, val_env, fcc_loader)
    """
    print("\n📦 Loading FCC Network Traces...")
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir=config.fcc_trace_dir,
        train_file=config.train_split,
        val_file=config.val_split,
        test_file=config.test_split
    )
    
    print(f"   ✓ Train traces: {len(fcc_loader.train_traces)}")
    print(f"   ✓ Val traces: {len(fcc_loader.val_traces)}")
    print(f"   ✓ Test traces: {len(fcc_loader.test_traces)}")
    
    print("\n🏗️  Creating Content-Aware Environments...")
    train_env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file=config.features_file,
        vmaf_file=config.vmaf_file,
        video_dir=config.video_dir,
        mode='train'
    )
    
    val_env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file=config.features_file,
        vmaf_file=config.vmaf_file,
        video_dir=config.video_dir,
        mode='val'
    )
    
    print("   ✅ Environments ready (VMAF-based reward)")
    
    return train_env, val_env, fcc_loader


def create_model(config: AdvancedTrainingConfig, device: torch.device) -> ContentAwareActor:
    """
    Initialize content-aware actor model
    
    Args:
        config: Training configuration
        device: Target device (cuda/cpu)
        
    Returns:
        ContentAwareActor model
    """
    print("\n🧠 Creating Content-Aware Model...")
    model = ContentAwareActor(
        state_dim=(6, 8),  # Network state: 6 features × 8 history
        action_dim=6,      # 6 bitrate levels
        content_dim=2      # SI/TI features
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(f"   ✓ Model device: {device}")
    
    return model


def evaluate_model(
    trainer: PPOTrainer,
    val_env: ContentAwareEnvFCC,
    n_episodes: int = 10
) -> Dict[str, float]:
    """
    Evaluate model performance on validation set
    
    Args:
        trainer: PPO trainer with trained model
        val_env: Validation environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dict with evaluation metrics
    """
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    episode_vmafs = []
    
    for ep in range(n_episodes):
        state = val_env.reset()
        episode_reward = 0.0
        episode_rebuffer = 0.0
        bitrates = []
        vmafs = []
        done = False
        
        while not done:
            # Get action from policy (deterministic for eval)
            with torch.no_grad():
                net = torch.FloatTensor(state['network']).unsqueeze(0)
                cont = torch.FloatTensor(state['content']).unsqueeze(0)
                vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
                
                action_probs, _ = trainer.model(net, cont, vmaf)
                action = action_probs.argmax(dim=1).item()
            
            # Step environment
            next_state, reward, done, info = val_env.step(action)
            
            episode_reward += reward
            episode_rebuffer += info.get('rebuffer_time', 0.0)
            bitrates.append(info.get('bitrate', 0))
            vmafs.append(info.get('vmaf', 0))
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(episode_rebuffer)
        episode_bitrates.append(sum(bitrates) / len(bitrates) if bitrates else 0)
        episode_vmafs.append(sum(vmafs) / len(vmafs) if vmafs else 0)
    
    return {
        'mean_reward': sum(episode_rewards) / len(episode_rewards),
        'std_reward': torch.std(torch.FloatTensor(episode_rewards)).item(),
        'mean_rebuffer': sum(episode_rebuffers) / len(episode_rebuffers),
        'mean_bitrate': sum(episode_bitrates) / len(episode_bitrates),
        'mean_vmaf': sum(episode_vmafs) / len(episode_vmafs)
    }


def train_advanced():
    """
    Main training loop with advanced features
    """
    print("=" * 80)
    print(f"🚀 ADVANCED TRAINING: Content-Aware ABR System")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Configuration
    config = AdvancedTrainingConfig()
    
    print("\n📋 Training Configuration:")
    print(f"   • Total timesteps: {config.total_timesteps:,}")
    print(f"   • Max updates: {config.max_updates}")
    print(f"   • Learning rate: {config.learning_rate}")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Rebuffer penalty: {config.rebuffer_penalty}")
    print(f"   • Early stopping patience: {config.early_stopping_patience}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    # Create environments
    train_env, val_env, fcc_loader = create_environments(config)
    
    # Create model
    model = create_model(config, device)
    
    # Create logger
    logger = TrainingLogger(
        log_dir=config.log_dir,
        run_name=config.run_name
    )
    print(f"\n📊 Logging to: {logger.log_file}")
    
    # Create PPO trainer
    print("\n🎓 Initializing PPO Trainer...")
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
    trainer.external_logger = logger
    
    print("   ✅ PPO trainer ready")
    
    # Training variables
    best_val_reward = float('-inf')
    no_improvement_count = 0
    update_count = 0
    timestep = 0
    
    print("\n" + "=" * 80)
    print("🚂 TRAINING STARTED")
    print("=" * 80)
    
    while timestep < config.total_timesteps and update_count < config.max_updates:
        # Collect rollout
        rollout = trainer.collect_rollout(n_steps=config.rollout_steps)
        timestep += len(rollout)
        
        # Update policy
        train_info = trainer.update_policy(rollout)
        update_count += 1
        
        # Log training metrics
        log_data = {
            'policy_loss': train_info['policy_loss'],
            'value_loss': train_info['value_loss'],
            'entropy': train_info['entropy']
        }
        log_entry = logger.log_update(update_count, timestep, log_data)
        
        # Print progress
        if update_count % config.log_interval == 0:
            logger.print_update(log_entry)
        
        # Evaluation
        if update_count % config.eval_interval == 0:
            print("\n" + "-" * 80)
            print(f"📊 EVALUATION at Update {update_count}")
            print("-" * 80)
            
            eval_metrics = evaluate_model(trainer, val_env, config.n_eval_episodes)
            
            print(f"   Reward: {eval_metrics['mean_reward']:+.2f} (σ={eval_metrics['std_reward']:.2f})")
            print(f"   Rebuffering: {eval_metrics['mean_rebuffer']:.2f}s")
            print(f"   Bitrate: {eval_metrics['mean_bitrate']:.0f} kbps")
            print(f"   VMAF: {eval_metrics['mean_vmaf']:.1f}")
            
            # Check improvement
            improvement = eval_metrics['mean_reward'] - best_val_reward
            
            if improvement > config.early_stopping_min_delta:
                best_val_reward = eval_metrics['mean_reward']
                no_improvement_count = 0
                
                # Save best model
                if eval_metrics['mean_reward'] > config.best_reward_threshold:
                    best_path = os.path.join(config.output_dir, 'best_model.pth')
                    torch.save({
                        'update': update_count,
                        'timestep': timestep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'reward': eval_metrics['mean_reward'],
                        'rebuffer': eval_metrics['mean_rebuffer'],
                        'config': vars(config)
                    }, best_path)
                    print(f"\n   🏆 New best model saved! Reward: {eval_metrics['mean_reward']:+.2f}")
            else:
                no_improvement_count += 1
                print(f"   ⚠️  No improvement for {no_improvement_count} evaluations")
            
            # Early stopping check
            if no_improvement_count >= config.early_stopping_patience:
                print(f"\n   ⏸️  Early stopping triggered at update {update_count}")
                print(f"      Best reward: {best_val_reward:+.2f}")
                break
            
            # Success condition
            if (eval_metrics['mean_reward'] > 110.0 and 
                eval_metrics['mean_rebuffer'] < 2.0):
                print(f"\n   🎯 TARGET ACHIEVED!")
                print(f"      Reward: {eval_metrics['mean_reward']:+.2f}")
                print(f"      Rebuffering: {eval_metrics['mean_rebuffer']:.2f}s")
                break
        
        # Regular checkpoint
        if update_count % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f'checkpoint_{update_count}.pth')
            torch.save({
                'update': update_count,
                'timestep': timestep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_reward': best_val_reward
            }, ckpt_path)
            print(f"\n   💾 Checkpoint saved: {ckpt_path}")
    
    # Training complete
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"   Total updates: {update_count}")
    print(f"   Total timesteps: {timestep:,}")
    print(f"   Best validation reward: {best_val_reward:+.2f}")
    print(f"   Models saved in: {config.output_dir}")
    print("=" * 80)
    
    return model, best_val_reward


if __name__ == '__main__':
    try:
        model, best_reward = train_advanced()
        print(f"\n🎯 Training completed successfully!")
        print(f"   Best reward achieved: {best_reward:+.2f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()