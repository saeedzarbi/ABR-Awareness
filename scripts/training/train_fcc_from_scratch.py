# scripts/training/train_fcc_from_scratch.py

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger

def train_on_fcc():
    """Train model from scratch on FCC traces"""
    
    print("="*60)
    print("🚀 Training ABR Model on FCC Traces (From Scratch)")
    print("="*60)
    
    # Configuration
    CONFIG = {
        # FCC Traces
        'fcc_trace_dir': 'data/fcc_traces',
        'train_split': 'data/network_traces/fcc/splits/fcc_train.txt',
        'val_split': 'data/network_traces/fcc/splits/fcc_val.txt',
        'test_split': 'data/network_traces/fcc/splits/fcc_test.txt',
        
        # Video data
        'features_file': 'data/features/si_ti_features.json',  # ✅ تغییر
        'vmaf_file': 'data/vmaf/vmaf_table.json',              # ✅ تغییر
        'video_dir': 'data/videos',
        
        # Training
        'total_timesteps': 1_000_000,
        'batch_size': 64,
        'n_epochs': 4,
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.10,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Evaluation
        'eval_interval': 50,
        'save_interval': 100,
        
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
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Create logger
    logger = TrainingLogger(
        log_dir='results/logs',
        run_name='fcc_training'
    )
    
    # Create trainer
    print("\n🎓 Creating PPO trainer...")
    trainer = PPOTrainer(
        model=model,
        env=train_env,
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        gae_lambda=CONFIG['gae_lambda'],
        clip_epsilon=CONFIG['clip_epsilon'],
        value_coef=CONFIG['value_coef'],
        entropy_coef=CONFIG['entropy_coef'],
        max_grad_norm=CONFIG['max_grad_norm'],
        n_epochs=CONFIG['n_epochs'],
        batch_size=CONFIG['batch_size']
    )
    
    # Set external logger
    trainer.external_logger = logger
    
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
        timestep += len(rollout)
        
        # Train
        train_info = trainer.update_policy(rollout)
        update_count += 1
        
        # Log
        log_data = {
            'policy_loss': train_info['policy_loss'],
            'value_loss': train_info['value_loss'],
            'entropy': train_info['entropy']
        }
        log_entry = logger.log_update(update_count, timestep, log_data)
        
        if update_count % 10 == 0:
            logger.print_update(log_entry)
        
        # Save checkpoint
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
    print(f"Total updates: {update_count}")
    print(f"Total timesteps: {timestep:,}")
    print(f"Models saved in: {CONFIG['output_dir']}")
    print("="*60)

if __name__ == '__main__':
    train_on_fcc()