"""
ULTIMATE TRAINING: Best of Both Worlds
1. Data Augmentation (از script شما)
2. Higher Entropy (از script شما)
3. Curriculum Learning (از approach ما)
4. Improved Reward (از approach ما)
5. Aggressive Early Stopping (از script شما)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import json

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.curriculum_env import CurriculumEnvironment
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.warmstart import warmstart_model


class UltimateConfig:
    """
    Ultimate configuration combining best practices
    """
    
    # Data paths
    fcc_trace_dir: str = 'data/fcc_traces'
    train_split: str = 'data/network_traces/fcc/splits/fcc_train.txt'
    val_split: str = 'data/network_traces/fcc/splits/fcc_val.txt'
    test_split: str = 'data/network_traces/fcc/splits/fcc_test.txt'
    features_file: str = 'data/features/si_ti_features.json'
    vmaf_file: str = 'data/vmaf/vmaf_table.json'
    video_dir: str = 'data/videos'
    
    # PPO hyperparameters - BEST SETTINGS
    total_timesteps: int = 600_000
    rollout_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    
    # CRITICAL: High entropy from your script
    entropy_coef: float = 0.15  # ✅ از script شما
    entropy_decay: float = 0.995
    entropy_min: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_decay_rate: float = 0.99
    lr_decay_interval: int = 30
    lr_min: float = 1e-5
    
    # Data Augmentation - از script شما
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_end_update: int = 150
    
    # Warmstart
    use_warmstart: bool = True
    warmstart_samples: int = 10000
    warmstart_epochs: int = 15
    
    # Training control
    target_update: int = 200
    max_updates: int = 250
    eval_interval: int = 25  # ✅ از script شما
    checkpoint_interval: int = 50
    log_interval: int = 10
    n_eval_episodes: int = 10
    
    # Aggressive early stopping - از script شما
    early_stopping_patience: int = 5  # ✅ کمتر از 30
    early_stopping_min_delta: float = 2.0
    target_reward: float = 100.0
    
    # Output
    output_dir: str = 'results/ultimate_training'
    run_name: str = f'ultimate_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class AugmentedFCCTraceLoader(FCCTraceLoader):
    """
    FCC Loader با data augmentation - از script شما
    """
    
    def __init__(self, *args, augmentation_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_prob = augmentation_prob
        print(f"   ✅ Data Augmentation enabled (p={augmentation_prob})")
    
    def augment_trace(self, trace_data):
        """
        Augment trace data
        """
        augmented = trace_data.copy()
        
        # Method 1: Gaussian noise on throughput
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.1, len(augmented))
            augmented[:, 1] += noise
            augmented[:, 1] = np.clip(augmented[:, 1], 0.1, 10.0)
        
        # Method 2: Bandwidth scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            augmented[:, 1] *= scale
            augmented[:, 1] = np.clip(augmented[:, 1], 0.1, 10.0)
        
        # Method 3: Time jitter
        if np.random.random() < 0.3:
            jitter = np.random.uniform(0.95, 1.05)
            augmented[:, 0] *= jitter
        
        return augmented
    
    def get_trace(self, mode='train'):
        """Get trace with optional augmentation"""
        trace = super().get_trace(mode)
        
        # Only augment in train mode
        if mode == 'train' and np.random.random() < self.augmentation_prob:
            trace = self.augment_trace(trace)
        
        return trace


class SimpleLogger:
    """Simple logger"""
    
    def __init__(self, log_dir: str, run_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'{run_name}.jsonl')
        self.metrics_history = []
    
    def log(self, update: int, metrics: Dict):
        entry = {
            'update': update,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(entry)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def print_progress(self, update: int, metrics: Dict):
        reward_20 = metrics.get('reward_20', 0)
        difficulty = metrics.get('difficulty', 0)
        
        print(f"Update {update:3d} | "
              f"R(20)={reward_20:+7.2f} | "
              f"Diff={difficulty:.2f} | "
              f"LR={metrics.get('lr', 0):.2e} | "
              f"Ent={metrics.get('entropy', 0):.4f}")


def create_environment(config: UltimateConfig, mode: str):
    """Create environment with augmentation"""
    
    # Create augmented loader
    if config.use_augmentation:
        fcc_loader = AugmentedFCCTraceLoader(
            fcc_trace_dir=config.fcc_trace_dir,
            train_file=config.train_split,
            val_file=config.val_split,
            test_file=config.test_split,
            augmentation_prob=config.augmentation_prob
        )
    else:
        fcc_loader = FCCTraceLoader(
            fcc_trace_dir=config.fcc_trace_dir,
            train_file=config.train_split,
            val_file=config.val_split,
            test_file=config.test_split
        )
    
    # Create curriculum environment
    env = CurriculumEnvironment(
        fcc_trace_loader=fcc_loader,
        features_file=config.features_file,
        vmaf_file=config.vmaf_file,
        video_dir=config.video_dir,
        mode=mode
    )
    
    return env, fcc_loader


def evaluate_on_validation(
    model,
    val_env,
    n_episodes: int = 10,
    device: str = 'cpu'
) -> Dict:
    """Evaluate model"""
    
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


def print_action_distribution(trainer):
    """Print action distribution"""
    if not hasattr(trainer, 'recent_actions'):
        return
    
    if len(trainer.recent_actions) > 0:
        action_counts = np.bincount(trainer.recent_actions[-1000:], minlength=6)
        total = action_counts.sum()
        
        if total > 0:
            bitrate_levels = [300, 750, 1850, 2850, 4300, 6000]
            print("\n   Action Distribution (last 1000 steps):")
            for i, count in enumerate(action_counts):
                pct = count / total * 100
                bar = '█' * int(pct / 2)
                print(f"      Action {i} ({bitrate_levels[i]:4d} kbps): {pct:5.1f}% {bar}")


def train_ultimate():
    """
    Ultimate training with all improvements
    """
    
    print("="*80)
    print("🚀 ULTIMATE TRAINING: Best of Both Worlds")
    print("="*80)
    print("   Features:")
    print("   ✓ Data Augmentation (noise, scaling, jitter)")
    print("   ✓ High Entropy (0.15 → 0.005)")
    print("   ✓ Curriculum Learning (easy → hard)")
    print("   ✓ Improved Reward (bitrate bonus)")
    print("   ✓ Warmstart Initialization")
    print("   ✓ Aggressive Early Stopping (patience=5)")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Configuration
    config = UltimateConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Data Augmentation: {config.use_augmentation} (p={config.augmentation_prob})")
    print(f"   Curriculum: {config.use_curriculum}")
    print(f"   Warmstart: {config.use_warmstart}")
    print(f"   Entropy: {config.entropy_coef} → {config.entropy_min}")
    print(f"   Early Stopping: patience={config.early_stopping_patience}")
    
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
    
    # Warmstart
    if config.use_warmstart:
        print(f"\n🔥 Warmstart Initialization...")
        model = warmstart_model(
            model,
            n_samples=config.warmstart_samples,
            n_epochs=config.warmstart_epochs,
            lr=1e-3
        )
    
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
    
    # Add action tracking
    trainer.recent_actions = []
    
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
        
        # Update curriculum difficulty
        if config.use_curriculum and hasattr(train_env, 'curriculum_loader'):
            difficulty = train_env.curriculum_loader.update_difficulty(
                update_count,
                config.curriculum_end_update
            )
            train_env.set_difficulty(difficulty)
        else:
            difficulty = 0.0
        
        # Collect rollout
        rollout = trainer.collect_rollout(n_steps=config.rollout_steps)
        timestep += len(rollout)
        
        # Track actions
        if hasattr(rollout, 'actions'):
            trainer.recent_actions.extend(rollout.actions)
            trainer.recent_actions = trainer.recent_actions[-5000:]
        
        # Update policy
        train_info = trainer.update_policy(rollout)
        update_count += 1
        
        # Learning rate decay
        if config.use_lr_schedule and update_count % config.lr_decay_interval == 0:
            current_lr = max(current_lr * config.lr_decay_rate, config.lr_min)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # Entropy decay
        current_entropy = max(current_entropy * config.entropy_decay, config.entropy_min)
        trainer.entropy_coef = current_entropy
        
        # Logging
        if update_count % config.log_interval == 0:
            recent_20 = trainer.episode_rewards[-20:] if len(trainer.episode_rewards) >= 20 else trainer.episode_rewards
            
            metrics = {
                'reward_20': np.mean(recent_20) if recent_20 else 0,
                'difficulty': difficulty,
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
            
            # Print action distribution
            print_action_distribution(trainer)
            
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
                    'reward': eval_results['mean_reward'],
                    'rebuffer': eval_results['mean_rebuffer'],
                    'vmaf': eval_results['mean_vmaf'],
                    'bitrate': eval_results['mean_bitrate'],
                    'difficulty': difficulty,
                    'config': vars(config)
                }, best_path)
                
                print(f"\n   🏆 New best! (+{improvement:.2f}) Saved to best_model.pth")
            else:
                no_improvement_count += 1
                print(f"   ⚠️  No improvement ({no_improvement_count}/{config.early_stopping_patience})")
            
            # Target reached?
            if eval_results['mean_reward'] > config.target_reward:
                print(f"\n   🎯 TARGET REACHED!")
                print(f"      Reward: {eval_results['mean_reward']:.2f} > {config.target_reward}")
                break
            
            # Early stopping
            if no_improvement_count >= config.early_stopping_patience:
                print(f"\n   ⏸️  Early stopping triggered")
                print(f"      Best reward: {best_val_reward:+.2f}")
                print(f"      Stopping at update {update_count}")
                break
        
        # Checkpoint
        if update_count % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f'checkpoint_{update_count}.pth')
            torch.save({
                'update': update_count,
                'timestep': timestep,
                'model_state_dict': model.state_dict(),
                'best_val_reward': best_val_reward,
                'difficulty': difficulty
            }, ckpt_path)
            print(f"\n   💾 Checkpoint: checkpoint_{update_count}.pth")
    
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
    
    # Final action distribution
    print(f"\n   Final Action Distribution:")
    print_action_distribution(trainer)
    
    # Compare with baseline
    print(f"\n   Comparison with Hybrid Baseline:")
    print(f"   • Hybrid:    +101.34 reward")
    print(f"   • Ours:      {final_results['mean_reward']:+.2f} reward")
    
    if final_results['mean_reward'] > 101.34:
        improvement = final_results['mean_reward'] - 101.34
        print(f"   • Result:    ✅ BEAT BASELINE by {improvement:+.2f}")
    else:
        gap = 101.34 - final_results['mean_reward']
        print(f"   • Result:    ❌ Below baseline by {gap:.2f}")
    
    print(f"\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"   Total updates: {update_count}")
    print(f"   Total timesteps: {timestep:,}")
    print(f"   Best val reward: {best_val_reward:+.2f}")
    print(f"   Final reward: {final_results['mean_reward']:+.2f}")
    print(f"   Final bitrate: {final_results['mean_bitrate']:.0f} kbps")
    print(f"   Saved to: {config.output_dir}")
    print("="*80)
    
    return model, final_results


if __name__ == '__main__':
    try:
        model, results = train_ultimate()
        
        print(f"\n🎉 Training Complete!")
        print(f"   Final reward: {results['mean_reward']:+.2f}")
        print(f"   Final bitrate: {results['mean_bitrate']:.0f} kbps")
        print(f"   Target: Beat +101.34 (Hybrid baseline)")
        
        if results['mean_reward'] > 101.34:
            print(f"   ✅ SUCCESS!")
        elif results['mean_reward'] > 80:
            print(f"   ⚠️  Close, but not quite there")
        else:
            print(f"   ❌ Need to write paper about challenges")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()