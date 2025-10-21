"""
scripts/training/train_ablation.py
===================================
Train ablated models for ablation study
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

from models.ablation_models import (
    AblatedActor_NoSITI, 
    AblatedActor_NoVMAF, 
    AblatedActor_NetworkOnly
)
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger

def train_ablation(ablation_type='no_siti'):
    """
    Train one ablated model
    
    Args:
        ablation_type: 'no_siti', 'no_vmaf', 'network_only'
    """
    
    print("="*80)
    print(f"🚀 Training Ablation Model: {ablation_type.upper()}")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'ablation_type': ablation_type,
        'fcc_trace_dir': 'data/fcc_traces',
        'train_split': 'data/network_traces/fcc/splits/fcc_train.txt',
        'val_split': 'data/network_traces/fcc/splits/fcc_val.txt',
        'test_split': 'data/network_traces/fcc/splits/fcc_test.txt',
        'features_file': 'data/features/si_ti_features.json',
        'vmaf_file': 'data/vmaf/vmaf_table.json',
        'video_dir': 'data/videos',
        
        # Training (same as your best model)
        'total_timesteps': 500_000,  # 500K like your model
        'batch_size': 64,
        'n_epochs': 4,
        'lr': 1e-4,  # low learning rate like fine-tuning
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.15,
        'entropy_coef': 0.05,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Evaluation
        'eval_interval': 25,
        'save_interval': 50,
        
        # Output
        'output_dir': f'results/ablation_{ablation_type}',
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # Load data
    print("\n📦 Loading data...")
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir=CONFIG['fcc_trace_dir'],
        train_file=CONFIG['train_split'],
        val_file=CONFIG['val_split'],
        test_file=CONFIG['test_split']
    )
    
    # Create environments
    print("\n🏗️  Creating environment...")
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
    
    # Create ablated model
    print(f"\n🧠 Creating {ablation_type} model...")
    if ablation_type == 'no_siti':
        model = AblatedActor_NoSITI(state_dim=(6,8), action_dim=6, content_dim=2)
        print("   ⚠️  Model WITHOUT SI/TI features")
    elif ablation_type == 'no_vmaf':
        model = AblatedActor_NoVMAF(state_dim=(6,8), action_dim=6, content_dim=2)
        print("   ⚠️  Model WITHOUT VMAF predictions")
    elif ablation_type == 'network_only':
        model = AblatedActor_NetworkOnly(state_dim=(6,8), action_dim=6, content_dim=2)
        print("   ⚠️  Model WITHOUT any content features (like Pensieve)")
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Create logger
    logger = TrainingLogger(
        log_dir='results/logs',
        run_name=f'ablation_{ablation_type}'
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
    
    trainer.external_logger = logger
    
    # Training loop
    print("\n" + "="*80)
    print(f"🚂 Starting Training: {ablation_type.upper()}")
    print("="*80)
    print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
    print(f"Evaluation every: {CONFIG['eval_interval']} updates")
    print("="*80 + "\n")
    
    best_reward = float('-inf')
    update_count = 0
    timestep = 0
    no_improvement = 0
    patience = 5
    
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
        
        # Validation
        if update_count % CONFIG['eval_interval'] == 0:
            print(f"\n📊 Validation at update {update_count}...")
            
            val_rewards = []
            for _ in range(10):
                state = val_env.reset()
                ep_reward = 0
                done = False
                
                while not done:
                    net = torch.FloatTensor(state['network']).unsqueeze(0).to(device)
                    cont = torch.FloatTensor(state['content']).unsqueeze(0).to(device)
                    vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        probs, _ = model(net, cont, vmaf)
                        action = probs.argmax(dim=1).item()
                    
                    state, reward, done, info = val_env.step(action)
                    ep_reward += reward
                
                val_rewards.append(ep_reward)
            
            val_mean = np.mean(val_rewards)
            val_std = np.std(val_rewards)
            
            print(f"   Val Reward: {val_mean:+.2f} ± {val_std:.2f}")
            
            # Check improvement
            if val_mean > best_reward + 1.0:
                print(f"   🏆 New best! (+{val_mean - best_reward:.2f})")
                best_reward = val_mean
                no_improvement = 0
                
                # Save best
                best_path = os.path.join(CONFIG['output_dir'], 'checkpoint_best.pth')
                torch.save({
                    'update': update_count,
                    'timestep': timestep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_reward': best_reward,
                    'config': CONFIG
                }, best_path)
            else:
                no_improvement += 1
                print(f"   ⚠️  No improvement ({no_improvement}/{patience})")
            
            # Early stopping
            if no_improvement >= patience:
                print("\n⏸️  Early stopping triggered!")
                break
        
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
            print(f"💾 Checkpoint saved: checkpoint_{update_count}.pth\n")
    
    print("\n" + "="*80)
    print(f"✅ Training Complete: {ablation_type.upper()}")
    print("="*80)
    print(f"Total updates: {update_count}")
    print(f"Best reward: {best_reward:+.2f}")
    print(f"Models saved in: {CONFIG['output_dir']}")
    print("="*80)


def train_all_ablations():
    """Train all three ablation models"""
    
    ablation_types = ['no_siti', 'no_vmaf', 'network_only']
    
    print("="*80)
    print("🧪 TRAINING ALL ABLATION MODELS")
    print("="*80)
    print(f"Models to train: {ablation_types}")
    print("This will take several hours...")
    print("="*80)
    
    for i, abl_type in enumerate(ablation_types, 1):
        print(f"\n\n{'='*80}")
        print(f"📊 [{i}/3] Training: {abl_type}")
        print(f"{'='*80}\n")
        
        train_ablation(abl_type)
        
        print(f"\n✅ [{i}/3] Completed: {abl_type}")
    
    print("\n" + "="*80)
    print("🎉 ALL ABLATION MODELS TRAINED!")
    print("="*80)
    print("\nSaved in:")
    for abl_type in ablation_types:
        print(f"   - results/ablation_{abl_type}/")
    print("\nNext step: Run evaluation to compare all models")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='all',
                       choices=['all', 'no_siti', 'no_vmaf', 'network_only'],
                       help='Which ablation to train')
    args = parser.parse_args()
    
    if args.type == 'all':
        train_all_ablations()
    else:
        train_ablation(args.type)