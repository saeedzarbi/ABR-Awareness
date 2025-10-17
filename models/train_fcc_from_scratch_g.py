# scripts/training/train_fcc_from_scratch.py

import os
import sys
import torch
from pathlib import Path

# اطمینان از اینکه ماژول‌های پروژه قابل ایمپورت هستند
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger

def train_on_fcc():
    """Train Content-Aware model from scratch on FCC traces"""

    print("="*60)
    print("🚀 Training YOUR Content-Aware ABR Model on FCC Traces")
    print("="*60)

    # --- تنظیمات ---
    CONFIG = {
        # مسیرهای داده شبکه FCC
        'fcc_trace_dir': 'data/fcc_traces',
        'train_split': 'data/network_traces/fcc/splits/fcc_train.txt',
        'val_split': 'data/network_traces/fcc/splits/fcc_val.txt',
        'test_split': 'data/network_traces/fcc/splits/fcc_test.txt',

        # مسیرهای داده ویدئو و ویژگی‌ها
        'features_file': 'data/features/si_ti_features.json',
        'vmaf_file': 'data/vmaf/vmaf_table.json',
        'video_dir': 'data/videos',

        # پارامترهای آموزش PPO
        'total_timesteps': 1_000_000, # تعداد کل گام‌های زمانی آموزش
        'batch_size': 64,             # اندازه بچ برای هر آپدیت
        'n_epochs': 4,                # تعداد تکرار روی هر بچ در هر آپدیت
        'lr': 3e-4,                   # نرخ یادگیری
        'gamma': 0.99,                # ضریب تخفیف پاداش‌های آینده
        'gae_lambda': 0.95,           # پارامتر GAE برای تخمین Advantage
        'clip_epsilon': 0.2,          # محدوده کلیپ کردن نسبت احتمال در PPO
        'entropy_coef': 0.10,         # ضریب جریمه انتروپی (برای تشویق کاوش)
        'value_coef': 0.5,            # ضریب تابع هزینه مقدار (Value Loss)
        'max_grad_norm': 0.5,         # حداکثر نرم گرادیان برای کلیپ کردن

        # تنظیمات ارزیابی و ذخیره‌سازی
        'eval_interval': 50,          # ارزیابی هر N آپدیت
        'save_interval': 100,         # ذخیره checkpoint هر N آپدیت

        # مسیرهای خروجی
        'output_dir': 'results/fcc_training', # دایرکتوری ذخیره مدل‌ها
        'log_file': 'results/fcc_training/train.jsonl' # فایل لاگ پیشرفت
    }

    # ساخت دایرکتوری خروجی اگر وجود ندارد
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # تعیین دستگاه (GPU یا CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")

    # لود کردن داده‌های شبکه FCC
    print("\n📦 Loading FCC traces...")
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir=CONFIG['fcc_trace_dir'],
        train_file=CONFIG['train_split'],
        val_file=CONFIG['val_split'],
        test_file=CONFIG['test_split']
    )
    print(f"   📊 FCC Traces Loaded:\n     Train: {len(fcc_loader.train_traces)}\n     Val: {len(fcc_loader.val_traces)}\n     Test: {len(fcc_loader.test_traces)}")


    # ساخت محیط‌های آموزش و اعتبارسنجی (محیط اصلی شما)
    print("\n🏗️  Creating Content-Aware environments...")
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
    print("   ✅ Environments created (using VMAF-based reward)")

    # ساخت مدل آگاه از محتوا
    print("\n🧠 Creating Content-Aware model...")
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # ساخت بهینه‌ساز (Optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # ساخت لاگر برای ثبت نتایج
    logger = TrainingLogger(
        log_dir='results/logs',
        run_name='fcc_training' # نام فایل لاگ اصلی شما
    )
    print(f"   📊 Logging to: {logger.log_file}")


    # ساخت ترینر PPO
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

    # اتصال لاگر خارجی به ترینر
    trainer.external_logger = logger

    # --- حلقه آموزش ---
    print("\n" + "="*60)
    print("🚂 Starting Training (Your Content-Aware Model)")
    print("="*60)
    print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
    print(f"Evaluation every: {CONFIG['eval_interval']} updates")
    print(f"Checkpoint every: {CONFIG['save_interval']} updates")
    print("="*60 + "\n")

    best_reward = float('-inf')
    update_count = 0
    timestep = 0

    while timestep < CONFIG['total_timesteps']:
        # جمع‌آوری داده از محیط
        rollout = trainer.collect_rollout(n_steps=2048) # n_steps معمولاً مضربی از batch_size است
        timestep += len(rollout)

        # آپدیت مدل با داده‌های جمع‌آوری شده
        train_info = trainer.update_policy(rollout)
        update_count += 1

        # ثبت لاگ‌های آموزش
        log_data = {
            'policy_loss': train_info['policy_loss'],
            'value_loss': train_info['value_loss'],
            'entropy': train_info['entropy']
        }
        log_entry = logger.log_update(update_count, timestep, log_data)

        # نمایش لاگ‌ها در کنسول هر ۱۰ آپدیت
        if update_count % 10 == 0:
            logger.print_update(log_entry)

        # ذخیره checkpoint مدل هر N آپدیت
        if update_count % CONFIG['save_interval'] == 0:
            ckpt_path = os.path.join(CONFIG['output_dir'],
                                    f'checkpoint_{update_count}.pth')
            try:
                torch.save({
                    'update': update_count,
                    'timestep': timestep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_reward': best_reward # (می‌توانید منطق best_reward را اضافه کنید)
                }, ckpt_path)
                print(f"\n💾 Checkpoint saved: {ckpt_path}\n")
            except Exception as e:
                print(f"\n❌ Error saving checkpoint: {e}\n")


    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Total updates: {update_count}")
    print(f"Total timesteps: {timestep:,}")
    print(f"Models saved in: {CONFIG['output_dir']}")
    print("="*60)

if __name__ == '__main__':
    train_on_fcc()