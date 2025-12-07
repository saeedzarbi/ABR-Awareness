import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch
from src.environment.abr_multi_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

class CurriculumConfig:
    # --- تنظیمات استراتژی یادگیری ---
    
    # مرحله ۱: گرم‌کردن با ویدیوی آسان (برای ایجاد اعتماد به نفس)
    PHASE1_VIDEOS = ['bigbuckbunny']
    PHASE1_STEPS = 300_000
    
    # مرحله ۲: سخت‌کردن با اضافه کردن ویدیوهای پیچیده
    PHASE2_VIDEOS = ['bigbuckbunny', 'crowd_run', 'tearsofsteel_short']
    PHASE2_STEPS = 700_000
    
    # ویدیو تست
    TEST_VIDEOS = ['park_joy']
    
    # تنظیمات مدل (پایدار)
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    LEARNING_RATE = 5e-5  # نرخ یادگیری پایین برای ثبات
    N_STEPS = 4096
    BATCH_SIZE = 256
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.1
    ENT_COEF = 0.05
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(video_list, is_eval=False, rank=0):
    """تابعی که محیط را با لیست ویدیوهای خاص می‌سازد"""
    def _init():
        trace_path = PATHS['test_traces'] if is_eval else PATHS['train_traces']
        env = ABREnv(
            video_names=video_list, # لیست ویدیوها اینجا دینامیک است
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=CurriculumConfig.MAX_CHUNKS,
            random_seed=rank * 100
        )
        return Monitor(env, info_keywords=('avg_quality', 'total_rebuffer'))
    return _init

def main():
    print("\n" + "="*70)
    print("🎓 Starting Curriculum Learning Strategy")
    print("="*70)
    
    save_dir = PATHS['models'] / 'ppo_curriculum'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_curriculum'
    
    # ---------------------------------------------------------
    # PHASE 1: WARM-UP (فقط ویدیوهای آسان)
    # ---------------------------------------------------------
    print(f"\n🔥 [Phase 1] Warm-up Training on: {CurriculumConfig.PHASE1_VIDEOS}")
    print(f"   Target: Learn to output high bitrates without fear.")
    
    env_phase1 = SubprocVecEnv([
        make_env(CurriculumConfig.PHASE1_VIDEOS, is_eval=False, rank=i) 
        for i in range(CurriculumConfig.NUM_ENVS)
    ])
    
    model = PPO(
        'MlpPolicy',
        env_phase1,
        learning_rate=CurriculumConfig.LEARNING_RATE,
        n_steps=CurriculumConfig.N_STEPS,
        batch_size=CurriculumConfig.BATCH_SIZE,
        n_epochs=CurriculumConfig.N_EPOCHS,
        gamma=CurriculumConfig.GAMMA,
        gae_lambda=CurriculumConfig.GAE_LAMBDA,
        clip_range=CurriculumConfig.CLIP_RANGE,
        ent_coef=CurriculumConfig.ENT_COEF,
        verbose=1,
        device=CurriculumConfig.DEVICE,
        tensorboard_log=str(log_dir)
    )
    
    # آموزش فاز ۱
    model.learn(total_timesteps=CurriculumConfig.PHASE1_STEPS, progress_bar=True)
    
    # ذخیره مدل گرم شده
    warmup_path = save_dir / "warmup_model"
    model.save(warmup_path)
    print(f"✓ Phase 1 Complete. Warmup model saved.")
    
    # بستن محیط‌های قدیمی برای آزادسازی رم
    env_phase1.close()
    
    # ---------------------------------------------------------
    # PHASE 2: HARDENING (ویدیوهای آسان + سخت)
    # ---------------------------------------------------------
    print(f"\n💪 [Phase 2] Hardening Training on: {CurriculumConfig.PHASE2_VIDEOS}")
    print(f"   Target: Adapt high-quality policy to complex scenes.")
    
    # ساخت محیط جدید با لیست کامل
    env_phase2 = SubprocVecEnv([
        make_env(CurriculumConfig.PHASE2_VIDEOS, is_eval=False, rank=i) 
        for i in range(CurriculumConfig.NUM_ENVS)
    ])
    
    # محیط تست
    eval_env = SubprocVecEnv([make_env(CurriculumConfig.TEST_VIDEOS, is_eval=True)])
    
    # لود کردن مدل از فاز ۱ (نکته کلیدی: محیط جدید را به آن می‌دهیم)
    model = PPO.load(warmup_path, env=env_phase2)
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=20000, save_path=str(save_dir / 'checkpoints'), name_prefix='curriculum'),
        EvalCallback(eval_env, best_model_save_path=str(save_dir / 'best_model'), log_path=str(log_dir / 'eval'), eval_freq=10000)
    ])
    
    # ادامه آموزش
    model.learn(
        total_timesteps=CurriculumConfig.PHASE2_STEPS, 
        callback=callbacks, 
        progress_bar=True,
        reset_num_timesteps=False # ادامه شمارش گام‌ها از 300هزار
    )
    
    model.save(save_dir / "final_model_curriculum")
    print("✓ Curriculum Training Successfully Completed.")

if __name__ == '__main__':
    main()