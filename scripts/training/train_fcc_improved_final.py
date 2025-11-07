"""
Training بهبود یافته با:
1. Data Augmentation
2. Early Stopping
3. Higher Entropy + Decay
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger

def train_on_fcc_improved():
    """Train model with improvements"""
    
    print("="*70)
    print("🚀 IMPROVED Training: Data Aug + Early Stop + Entropy")
    print("="*70)
    
    # Configuration
    CONFIG = {
        # FCC Traces
        'fcc_trace_dir': 'data/fcc_traces',
        'train_split': 'data/network_traces/fcc/splits/fcc_train.txt',
        'val_split': 'data/network_traces/fcc/splits/fcc_val.txt',
        'test_split': 'data/network_traces/fcc/splits/fcc_test.txt',
        
        # Video data
        'features_file': 'data/features/si_ti_features.json',
        'vmaf_file': 'data/vmaf/vmaf_table.json',
        'video_dir': 'data/videos',
        
        # Training
        'total_timesteps': 1_000_000,
        'batch_size': 64,
        'n_epochs': 4,
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.15,        # ✅ بالاتر شد
        'entropy_decay': 0.995,       # ✅ کاهش تدریجی
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # ✅ Data Augmentation
        'augmentation_prob': 0.5,     # 50% احتمال augmentation
        
        # ✅ Early Stopping
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1.0,
        
        # Evaluation
        'eval_interval': 25,          # زودتر ارزیابی
        'save_interval': 50,
        
        # Output
        'output_dir': 'results/fcc_training_improved_new',
        'log_file': 'results/fcc_training_improved/train.jsonl'
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # ═══════════════════════════════════════════════════════════
    # ✅ Enhanced FCC Trace Loader with Augmentation
    # ═══════════════════════════════════════════════════════════
    
    print("\n📦 Loading FCC traces with augmentation...")
    
    class AugmentedFCCTraceLoader(FCCTraceLoader):
        """FCC Loader با data augmentation"""
        
        def __init__(self, *args, augmentation_prob=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.augmentation_prob = augmentation_prob
            print(f"   ✅ Data Augmentation enabled (p={augmentation_prob})")
        
        def augment_trace(self, trace_data):
            """
            Augment trace data
            Returns: augmented trace (same format as input)
            """
            augmented = trace_data.copy()
            
            # روش 1: Gaussian noise on throughput
            if np.random.random() < 0.5:
                noise = np.random.normal(0, 0.1, len(augmented))
                augmented[:, 1] += noise
                augmented[:, 1] = np.clip(augmented[:, 1], 0.1, 10.0)
            
            # روش 2: Bandwidth scaling
            if np.random.random() < 0.5:
                scale = np.random.uniform(0.8, 1.2)
                augmented[:, 1] *= scale
                augmented[:, 1] = np.clip(augmented[:, 1], 0.1, 10.0)
            
            # روش 3: Time jitter
            if np.random.random() < 0.3:
                jitter = np.random.uniform(0.95, 1.05)
                augmented[:, 0] *= jitter
            
            return augmented
        
        def get_trace(self, mode='train'):
            """Get trace with optional augmentation"""
            trace = super().get_trace(mode)
            
            # فقط در train mode augment کن
            if mode == 'train' and np.random.random() < self.augmentation_prob:
                trace = self.augment_trace(trace)
            
            return trace
    
    fcc_loader = AugmentedFCCTraceLoader(
        fcc_trace_dir=CONFIG['fcc_trace_dir'],
        train_file=CONFIG['train_split'],
        val_file=CONFIG['val_split'],
        test_file=CONFIG['test_split'],
        augmentation_prob=CONFIG['augmentation_prob']
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    logger = TrainingLogger(
        log_dir='results/logs',
        run_name='fcc_training_improved'
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
    
    # ═══════════════════════════════════════════════════════════
    # Training loop with improvements
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🚂 Starting Improved Training")
    print("="*70)
    print(f"✅ Data Augmentation: {CONFIG['augmentation_prob']*100:.0f}% probability")
    print(f"✅ Early Stopping: patience={CONFIG['early_stopping_patience']}")
    print(f"✅ Entropy: {CONFIG['entropy_coef']} with decay={CONFIG['entropy_decay']}")
    print("="*70 + "\n")
    
    best_val_reward = float('-inf')
    no_improvement_count = 0
    update_count = 0
    timestep = 0
    current_entropy_coef = CONFIG['entropy_coef']
    
    while timestep < CONFIG['total_timesteps']:
        # Collect rollouts
        rollout = trainer.collect_rollout(n_steps=2048)
        timestep += len(rollout)
        
        # ✅ Update entropy coefficient (decay)
        trainer.entropy_coef = current_entropy_coef
        current_entropy_coef *= CONFIG['entropy_decay']
        
        # Train
        train_info = trainer.update_policy(rollout)
        update_count += 1
        
        # Log
        log_data = {
            'policy_loss': train_info['policy_loss'],
            'value_loss': train_info['value_loss'],
            'entropy': train_info['entropy'],
            'entropy_coef': trainer.entropy_coef
        }
        log_entry = logger.log_update(update_count, timestep, log_data)
        
        if update_count % 10 == 0:
            logger.print_update(log_entry)
            print(f"         Entropy coef: {trainer.entropy_coef:.4f}")
        
        # ✅ Evaluation for early stopping
        if update_count % CONFIG['eval_interval'] == 0:
            print(f"\n📊 Evaluation at update {update_count}...")
            
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
            
            # ✅ Early stopping check
            improvement = val_mean - best_val_reward
            
            if improvement > CONFIG['early_stopping_min_delta']:
                print(f"   🏆 New best! Improvement: +{improvement:.2f}")
                best_val_reward = val_mean
                no_improvement_count = 0
                
                # Save best model
                best_path = os.path.join(CONFIG['output_dir'], 'checkpoint_best.pth')
                torch.save({
                    'update': update_count,
                    'timestep': timestep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_reward': best_val_reward,
                    'config': CONFIG
                }, best_path)
                print(f"   💾 Best model saved")
            else:
                no_improvement_count += 1
                print(f"   ⚠️  No improvement ({no_improvement_count}/{CONFIG['early_stopping_patience']})")
            
            # Check early stopping
            if no_improvement_count >= CONFIG['early_stopping_patience']:
                print("\n" + "="*70)
                print("⏸️  EARLY STOPPING TRIGGERED")
                print("="*70)
                print(f"No improvement for {no_improvement_count} evaluations")
                print(f"Best val reward: {best_val_reward:+.2f}")
                print(f"Stopping at update {update_count}, timestep {timestep:,}")
                print("="*70)
                break
            
            print()
        
        # Save checkpoint
        if update_count % CONFIG['save_interval'] == 0:
            ckpt_path = os.path.join(CONFIG['output_dir'], 
                                    f'checkpoint_{update_count}.pth')
            torch.save({
                'update': update_count,
                'timestep': timestep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_reward': best_val_reward,
                'config': CONFIG
            }, ckpt_path)
            print(f"💾 Checkpoint saved: checkpoint_{update_count}.pth\n")
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Total updates: {update_count}")
    print(f"Total timesteps: {timestep:,}")
    print(f"Best val reward: {best_val_reward:+.2f}")
    print(f"Models saved in: {CONFIG['output_dir']}")
    print("="*70)

if __name__ == '__main__':
    train_on_fcc_improved()