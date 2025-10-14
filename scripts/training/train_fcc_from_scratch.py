# scripts/training/train_fcc_from_scratch.py

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import ContentAwareActorCritic
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.logger import Logger

def train_on_fcc():
    """Train model from scratch on FCC traces"""
    
    print("="*60)
    print("🚀 Training ABR Model on FCC Traces (From Scratch)")
    print("="*60)
    
    # Configuration
    CONFIG = {
        # FCC Traces
        'fcc_trace_dir': 'data/network_traces/fcc',
        'train_split': 'data/network_traces/fcc/splits/fcc_train.txt',
        'val_split': 'data/network_traces/fcc/splits/fcc_val.txt',
        'test_split': 'data/network_traces/fcc/splits/fcc_test.txt',
        
        # Video data
        'features_file': 'data/features/video_features.json',
        'vmaf_file': 'data/vmaf/vmaf_scores.json',
        'video_dir': 'data/videos',
        
        # Training
        'total_timesteps': 1_000_000,
        'batch_size': 64,
        'n_epochs': 4,
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.10,  # High exploration
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Evaluation
        'eval_interval': 50,  # Every 50 updates
        'save_interval': 100,  # Every 100 updates
        
        # Output
        'output_dir': 'results/fcc_training',
        'log_file': 'results/fcc_training/train.jsonl'
    }
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # Load FCC traces
    print("\n📦 Loading FCC traces...")
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir=CONFIG['fcc_trace_dir'],
        train_file=CONFIG['train_split'],
        val_file=CONFIG['val_split'],
        test_file=CONFIG['test_split']
    )
    
    # Create environments
    print("\n🏗️  Creating environments...")
    train_env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file=CONFIG['features_file'],
        vmaf_file=CONFIG['vmaf_file'],
        video_dir=CONFIG['video_dir'],
        mode='train'
    )
    
    val_env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file=CONFIG['features_file'],
        vmaf_file=CONFIG['vmaf_file'],
        video_dir=CONFIG['video_dir'],
        mode='val'
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = ContentAwareActorCritic(
        network_input_shape=(6, 8),
        content_feature_dim=2,
        vmaf_feature_dim=6,
        action_dim=6
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Create logger
    logger = Logger(CONFIG['log_file'])
    
    # Create trainer
    print("\n🎓 Creating PPO trainer...")
    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        env=train_env,
        val_env=val_env,
        device=device,
        logger=logger,
        batch_size=CONFIG['batch_size'],
        n_epochs=CONFIG['n_epochs'],
        gamma=CONFIG['gamma'],
        gae_lambda=CONFIG['gae_lambda'],
        clip_epsilon=CONFIG['clip_epsilon'],
        entropy_coef=CONFIG['entropy_coef'],
        value_coef=CONFIG['value_coef'],
        max_grad_norm=CONFIG['max_grad_norm']
    )
    
    # Training loop
    print("\n" + "="*60)
    print("🚂 Starting Training on FCC Traces")
    print("="*60)
    print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
    print(f"Evaluation every: {CONFIG['eval_interval']} updates")
    print(f"Checkpoint every: {CONFIG['save_interval']} updates")
    print("="*60 + "\n")
    
    best_reward = float('-inf')
    update_count = 0
    timestep = 0
    
    while timestep < CONFIG['total_timesteps']:
        # Collect rollouts
        rollout = trainer.collect_rollout(n_steps=2048)
        timestep += len(rollout['rewards'])
        
        # Train
        train_info = trainer.train_step(rollout)
        update_count += 1
        
        # Log
        log_data = {
            'update': update_count,
            'timestep': timestep,
            'train_reward': train_info['episode_reward'],
            'policy_loss': train_info['policy_loss'],
            'value_loss': train_info['value_loss'],
            'entropy': train_info['entropy']
        }
        logger.log(log_data)
        
        print(f"Update {update_count:4d} | Timestep {timestep:7d} | "
              f"Reward: {train_info['episode_reward']:7.2f} | "
              f"Entropy: {train_info['entropy']:.3f}")
        
        # Evaluation
        if update_count % CONFIG['eval_interval'] == 0:
            print(f"\n📊 Evaluation at update {update_count}...")
            val_reward = trainer.evaluate(n_episodes=20)
            print(f"   Validation Reward: {val_reward:.2f}")
            
            # Save best model
            if val_reward > best_reward:
                best_reward = val_reward
                best_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
                torch.save({
                    'update': update_count,
                    'timestep': timestep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_reward': best_reward
                }, best_path)
                print(f"   ✅ New best model saved! Reward: {best_reward:.2f}\n")
        
        # Regular checkpoint
        if update_count % CONFIG['save_interval'] == 0:
            ckpt_path = os.path.join(CONFIG['output_dir'], 
                                    f'checkpoint_{update_count}.pth')
            torch.save({
                'update': update_count,
                'timestep': timestep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_reward': best_reward
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}\n")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best validation reward: {best_reward:.2f}")
    print(f"Total updates: {update_count}")
    print(f"Total timesteps: {timestep:,}")
    print(f"Models saved in: {CONFIG['output_dir']}")
    print("="*60)

if __name__ == '__main__':
    train_on_fcc()