# scripts/training/train_pensieve.py

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ✅ تغییر ۱: ایمپورت مدل و محیط جدید
from models.pensieve_actor_compatible import PensieveActorCompatible
from models.pensieve_env_fcc import PensieveEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger

def train_pensieve_on_fcc():
    """Train Pensieve (compatible) model from scratch on FCC traces"""
    
    print("="*60)
    # ✅ تغییر ۲: تغییر عنوان
    print("🚀 Training Pensieve (Content-BLIND) Model on FCC Traces")
    print("="*60)
    
    # Configuration
    CONFIG = {
        # ... (تمام تنظیمات CONFIG را از فایل اصلی کپی کنید) ...
        'fcc_trace_dir': 'data/fcc_traces',
        'train_split': 'data/network_traces/fcc/splits/fcc_train.txt',
        'val_split': 'data/network_traces/fcc/splits/fcc_val.txt',
        'test_split': 'data/network_traces/fcc/splits/fcc_test.txt',
        'features_file': 'data/features/si_ti_features.json',
        'vmaf_file': 'data/vmaf/vmaf_table.json',
        'video_dir': 'data/videos',
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
        'eval_interval': 50,
        'save_interval': 100,
        
        # ✅ تغییر ۳: دایرکتوری خروجی جدید برای مدل Pensieve
        'output_dir': 'results/pensieve_fcc_training',
        'log_file': 'results/pensieve_fcc_training/train.jsonl'
    }
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    # Load FCC traces (بدون تغییر)
    print("\n📦 Loading FCC traces...")
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir=CONFIG['fcc_trace_dir'],
        train_file=CONFIG['train_split'],
        val_file=CONFIG['val_split'],
        test_file=CONFIG['test_split']
    )
    
    # ✅ تغییر ۴: استفاده از محیط Pensieve
    print("\n🏗️  Creating Pensieve environments...")
    train_env = PensieveEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file=CONFIG['features_file'],
        vmaf_file=CONFIG['vmaf_file'],
        video_dir=CONFIG['video_dir'],
        mode='train'
    )
    
    val_env = PensieveEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file=CONFIG['features_file'],
        vmaf_file=CONFIG['vmaf_file'],
        video_dir=CONFIG['video_dir'],
        mode='val'
    )
    
    # ✅ تغییر ۵: استفاده از مدل Pensieve
    print("\n🧠 Creating Pensieve (compatible) model...")
    model = PensieveActorCompatible(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2  # این پارامتر لازم است اما مدل از آن استفاده نمی‌کند
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # --- بقیه اسکریپت را از فایل اصلی کپی کنید ---
    # (optimizer, logger, trainer, و حلقه آموزش)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    logger = TrainingLogger(
        log_dir='results/logs',
        run_name='pensieve_fcc_training' # نام لاگ را هم عوض می‌کنیم
    )
    
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
    
    print("\n" + "="*60)
    print("🚂 Starting Training (Pensieve Model)")
    print("="*60)
    
    best_reward = float('-inf')
    update_count = 0
    timestep = 0
    
    while timestep < CONFIG['total_timesteps']:
        rollout = trainer.collect_rollout(n_steps=2048)
        timestep += len(rollout)
        
        train_info = trainer.update_policy(rollout)
        update_count += 1
        
        log_data = {
            'policy_loss': train_info['policy_loss'],
            'value_loss': train_info['value_loss'],
            'entropy': train_info['entropy']
        }
        log_entry = logger.log_update(update_count, timestep, log_data)
        
        if update_count % 10 == 0:
            logger.print_update(log_entry)
        
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

if __name__ == '__main__':
    train_pensieve_on_fcc()