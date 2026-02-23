import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
from src.environment.abr_multi_env_l import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

# ==========================================
# 🎯 تنظیم نام پوشه جدید برای ذخیره مدل
# ==========================================
RUN_NAME = "ppo_proposed_v2_fresh"  # می‌توانید این نام را هر چیزی که دوست دارید بگذارید
# ==========================================

class EnhancedLoggingCallback(BaseCallback):
    def __init__(self, log_dir: Path, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.action_log_path = self.log_dir / "actions_detailed.csv"
        self.episode_log_path = self.log_dir / "episodes_detailed.csv"
        self._init_log_files()
        
    def _init_log_files(self):
        with open(self.action_log_path, 'w') as f:
            f.write("step,mean_action,action_distribution,action_variance,action_entropy\n")
        with open(self.episode_log_path, 'w') as f:
            f.write("step,episode_count,avg_reward,avg_vmaf,avg_rebuffer_rate,episode_length\n")
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            actions = self.locals.get('actions', [])
            if len(actions) > 0:
                mean_action = np.mean(actions)
                action_var = np.var(actions)
                action_dist = np.bincount(actions, minlength=6) / len(actions)
                action_dist_safe = action_dist + 1e-10
                entropy = -np.sum(action_dist_safe * np.log(action_dist_safe))
                with open(self.action_log_path, 'a') as f:
                    dist_str = ';'.join([f'{p:.3f}' for p in action_dist])
                    f.write(f"{self.num_timesteps},{mean_action:.2f},{dist_str},{action_var:.2f},{entropy:.3f}\n")
                    
        if 'infos' in self.locals:
            for info in self.locals.get('infos', []):
                if 'episode' in info:
                    ep_info = info['episode']
                    avg_vmaf = info.get('avg_quality', 0.0)
                    rebuffer_rate = (info.get('total_rebuffer', 0.0) / (ep_info['l'] * 4.0)) * 100 if ep_info['l'] > 0 else 0
                    with open(self.episode_log_path, 'a') as f:
                        f.write(f"{self.num_timesteps},{self.n_calls},{ep_info['r']:.2f},{avg_vmaf:.2f},{rebuffer_rate:.2f},{ep_info['l']}\n")
        return True

class MainTrainingConfig:
    TRAIN_VIDEOS = ['bigbuckbunny', 'crowd_run', 'tearsofsteel_short']
    TEST_VIDEOS = ['sintel'] 
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    LEARNING_RATE = 3e-4
    N_STEPS = 4096          
    BATCH_SIZE = 128
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.05 
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    TOTAL_TIMESTEPS = 800_000  # تعداد قدم‌ها برای آموزش کامل
    EVAL_FREQ = 20_000
    SAVE_FREQ = 50_000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(rank: int, seed: int = 0, is_eval: bool = False):
    def _init():
        video_list = MainTrainingConfig.TEST_VIDEOS if is_eval else MainTrainingConfig.TRAIN_VIDEOS
        trace_path = PATHS['test_traces'] if is_eval else PATHS['train_traces']
            
        env = ABREnv(
            video_names=video_list,
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=MainTrainingConfig.MAX_CHUNKS,
            random_seed=seed + rank,
            use_lyapunov=True, # 🟢 مدل اصلی لیاپانوف دارد
            use_future=True    # 🟢 مدل اصلی آینده را می‌بیند
        )
        return Monitor(env, info_keywords=('avg_quality', 'total_rebuffer'))
    return _init

def main():
    print(f"\n🚀 Starting training for PROPOSED MODEL in directory: {RUN_NAME}")
    
    save_dir = PATHS['models'] / RUN_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(MainTrainingConfig.NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=True)])
    
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=MainTrainingConfig.LEARNING_RATE,
        n_steps=MainTrainingConfig.N_STEPS,
        batch_size=MainTrainingConfig.BATCH_SIZE,
        n_epochs=MainTrainingConfig.N_EPOCHS,
        gamma=MainTrainingConfig.GAMMA,
        gae_lambda=MainTrainingConfig.GAE_LAMBDA,
        clip_range=MainTrainingConfig.CLIP_RANGE,
        ent_coef=MainTrainingConfig.ENT_COEF,
        vf_coef=MainTrainingConfig.VF_COEF,
        max_grad_norm=MainTrainingConfig.MAX_GRAD_NORM,
        verbose=1,
        device=MainTrainingConfig.DEVICE,
        tensorboard_log=str(log_dir)
    )
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=MainTrainingConfig.SAVE_FREQ // MainTrainingConfig.NUM_ENVS, save_path=str(save_dir / 'checkpoints'), name_prefix=RUN_NAME, save_replay_buffer=False, save_vecnormalize=False),
        EvalCallback(eval_env, best_model_save_path=str(save_dir / 'best_model'), log_path=str(log_dir / 'eval'), eval_freq=MainTrainingConfig.EVAL_FREQ // MainTrainingConfig.NUM_ENVS, n_eval_episodes=10, deterministic=True, render=False, verbose=0),
        EnhancedLoggingCallback(log_dir=log_dir, log_freq=5000, verbose=0)
    ])
    
    try:
        model.learn(total_timesteps=MainTrainingConfig.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
        model.save(save_dir / 'final_model')
        print(f"\n✅ Training completed! Model saved in 'results/models/{RUN_NAME}'")
    except KeyboardInterrupt:
        model.save(save_dir / 'interrupted_model')
        print("\n⚠️ Training interrupted and saved.")
    finally:
        train_env.close()
        eval_env.close()

if __name__ == '__main__':
    main()