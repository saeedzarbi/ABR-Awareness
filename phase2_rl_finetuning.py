"""
Phase 2: RL Fine-tuning with PPO
Load pretrained BC model and fine-tune with PPO
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import json

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer


class FineTuningConfig:
    """
    Configuration for RL fine-tuning
    """
    
    # Data paths
    fcc_trace_dir: str = 'data/fcc_traces'
    train_split: str = 'data/network_traces/fcc/splits/fcc_train.txt'
    val_split: str = 'data/network_traces/fcc/splits/fcc_val.txt'
    test_split: str = 'data/network_traces/fcc/splits/fcc_test.txt'
    features_file: str = 'data/features/si_ti_features.json'
    vmaf_file: str = 'data/vmaf/vmaf_table.json'
    video_dir: str = 'data/videos'
    
    # PPO hyperparameters - CONSERVATIVE for fine-tuning
    total_timesteps: int = 400_000  # ~160 updates
    rollout_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 1e-4  # Lower for fine-tuning
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1  # Smaller clip for stability
    
    # Lower entropy (already have good policy)
    entropy_coef: float = 0.02
    entropy_decay: float = 0.997
    entropy_min: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_decay_rate: float = 0.99
    lr_decay_interval: int = 20
    lr_min: float = 1e-6
    
    # Training control
    target_update: int = 160
    max_updates: int = 200
    eval_interval: int = 10
    checkpoint_interval: int = 20
    log_interval: int = 5
    n_eval_episodes: int = 10
    
    # Early stopping
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1.0
    target_reward: float = 105.0  # Beat Hybrid baseline (+101.34)
    target_rebuffer: float = 4.0
    
    # Pretrained model
    pretrained_path: str = 'results/bc_pretrained.pth'
    
    # Output
    output_dir: str = 'results/phase2_finetuning'
    run_name: str = f'finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class SimpleLogger:
    """Lightweight logger"""
    
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
        reward_100 = metrics.get('reward_100', 0)
        
        print(f"Update {update:3d} | "
              f"R(20)={reward_20:+7.2f} | "
              f"R(100)={reward_100:+7.2f} | "
              f"LR={metrics.get('lr', 0):.2e} | "
              f"Ent={metrics.get('entropy', 0):.4f}")


def create_environment(config: FineTuningConfig, mode: str):
    """Create environment"""
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


def finetune_with_ppo():
    """Main fine-tuning function"""
    
    print("="*80)
    print("🚀 PHASE 2: RL FINE-TUNING")
    print("="*80)
    print(f"   Goal: Beat Hybrid baseline (+101.34 reward)")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Configuration
    config = FineTuningConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Learning rate: {config.learning_rate} (low for fine-tuning)")
    print(f"   Clip epsilon: {config.clip_epsilon} (small for stability)")
    print(f"   Entropy: {config.entropy_coef} (low, already have good policy)")
    print(f"   Target reward: {config.target_reward}")
    
    # Load pretrained model
    print(f"\n🔄 Loading Pretrained Model...")
    
    if not os.path.exists(config.pretrained_path):
        print(f"❌ Pretrained model not found: {config.pretrained_path}")
        print("   Please run phase1_behavioral_cloning.py first!")
        return
    
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    ).to(device)
    
    checkpoint = torch.load(config.pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded pretrained model")
    print(f"   BC Accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"   Epoch: {checkpoint['epoch']}")
    
    # Create environments
    print(f"\n🏗️  Creating Environments...")
    train_env, fcc_loader = create_environment(config, mode='train')
    val_env, _ = create_environment(config, mode='val')
    print(f"   ✅ Environments ready")
    
    # Evaluate pretrained model
    print(f"\n📊 Evaluating Pretrained Model...")
    pretrained_results = evaluate_on_validation(model, val_env, config.n_eval_episodes, device)
    print(f"   Reward:    {pretrained_results['mean_reward']:+.2f}")
    print(f"   Rebuffer:  {pretrained_results['mean_rebuffer']:.2f}s")
    print(f"   VMAF:      {pretrained_results['mean_vmaf']:.1f}")
    print(f"   Bitrate:   {pretrained_results['mean_bitrate']:.0f} kbps")
    
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
    best_val_reward = pretrained_results['mean_reward']
    no_improvement_count = 0
    update_count = 0
    timestep = 0
    current_entropy = config.entropy_coef
    current_lr = config.learning_rate
    
    print(f"\n" + "="*80)
    print("🚂 FINE-TUNING LOOP")
    print("="*80)
    
    while update_count < config.max_updates and timestep < config.total_timesteps:
        
        # Collect rollout
        rollout = trainer.collect_rollout(n_steps=config.rollout_steps)
        timestep += len(rollout)
        
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
                    'bitrate': eval_results['mean_bitrate'],
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
        'pretrained_results': pretrained_results,
        'config': vars(config)
    }, final_path)
    
    # Compare with pretrained
    print(f"\n" + "="*80)
    print("📈 IMPROVEMENT ANALYSIS")
    print("="*80)
    
    reward_improvement = final_results['mean_reward'] - pretrained_results['mean_reward']
    print(f"\nPretrained (BC only):    {pretrained_results['mean_reward']:+.2f}")
    print(f"After fine-tuning (RL):  {final_results['mean_reward']:+.2f}")
    print(f"Improvement:             {reward_improvement:+.2f}")
    
    print(f"\nTarget (Hybrid baseline): +101.34")
    if final_results['mean_reward'] > 101.34:
        print(f"✅ BEAT BASELINE by {final_results['mean_reward'] - 101.34:.2f}")
    else:
        print(f"❌ Below baseline by {101.34 - final_results['mean_reward']:.2f}")
    
    print(f"\n" + "="*80)
    print("✅ PHASE 2 COMPLETE")
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
        model, results = finetune_with_ppo()
        
        print(f"\n🎉 Fine-tuning Complete!")
        print(f"   Final reward: {results['mean_reward']:+.2f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()