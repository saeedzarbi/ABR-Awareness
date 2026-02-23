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

# ... (Logging callbacks kept exactly the same for stability) ...
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
            self._log_actions()
        if 'infos' in self.locals:
            self._collect_episode_data()
        return True
    
    def _log_actions(self):
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
            
    def _collect_episode_data(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                avg_reward = ep_info['r']
                ep_length = ep_info['l']
                avg_vmaf = info.get('avg_quality', 0.0)
                total_rebuffer = info.get('total_rebuffer', 0.0)
                rebuffer_rate = (total_rebuffer / (ep_length * 4.0)) * 100 if ep_length > 0 else 0
                
                with open(self.episode_log_path, 'a') as f:
                    f.write(f"{self.num_timesteps},{self.n_calls},{avg_reward:.2f},"
                           f"{avg_vmaf:.2f},{rebuffer_rate:.2f},{ep_length}\n")

class ActionLogCallback(BaseCallback):
    def __init__(self, log_freq: int = 10000, log_file: str = "actions_history.txt", verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_path = PATHS['logs'] / log_file
        with open(self.log_path, "w") as f:
            f.write("Step, Mean_Bitrate_Index, Last_Actions_Sample\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            actions = self.locals['actions']
            mean_action = np.mean(actions)
            with open(self.log_path, "a") as f:
                f.write(f"{self.num_timesteps}, {mean_action:.2f}, {list(actions)}\n")
        return True

class TrainingConfigAblation:
    """
    Configuration for Automated Ablation Study Training
    """
    TRAIN_VIDEOS = ['bigbuckbunny', 'crowd_run', 'tearsofsteel_short']
    TEST_VIDEOS = ['sintel'] 
    
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # Hyperparameters
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
    
    # Reduced timesteps for faster ablation (Original was 1M, 400k is usually enough to see the difference)
    TOTAL_TIMESTEPS = 400_000 
    EVAL_FREQ = 20_000
    SAVE_FREQ = 50_000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# We pass use_lyapunov and use_future to the environment builder
def make_env(rank: int, seed: int = 0, is_eval: bool = False, use_lyap: bool = True, use_fut: bool = True):
    def _init():
        if is_eval:
            video_list = TrainingConfigAblation.TEST_VIDEOS
            trace_path = PATHS['test_traces']
        else:
            video_list = TrainingConfigAblation.TRAIN_VIDEOS
            trace_path = PATHS['train_traces']
            
        if not video_list: video_list = ['bigbuckbunny']

        env = ABREnv(
            video_names=video_list,
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=TrainingConfigAblation.MAX_CHUNKS,
            random_seed=seed + rank,
            use_lyapunov=use_lyap, # <--- Added Toggle
            use_future=use_fut     # <--- Added Toggle
        )
        return Monitor(env, info_keywords=('avg_quality', 'total_rebuffer'))
    return _init

def train_ablation_variant(variant_name: str, use_lyap: bool, use_fut: bool):
    """Trains a specific variant for the ablation study."""
    
    print("\n" + "="*70)
    print(f"🚀 Training Ablation Variant: {variant_name.upper()}")
    print(f"   Lyapunov Penalty: {'ON' if use_lyap else 'OFF'}")
    print(f"   Future Info (f_next): {'ON' if use_fut else 'OFF'}")
    print("="*70 + "\n")
    
    # Create specific directories for this variant
    save_dir = PATHS['models'] / f'ablation_{variant_name}'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / f'ablation_{variant_name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False, use_lyap=use_lyap, use_fut=use_fut) for i in range(TrainingConfigAblation.NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=True, use_lyap=use_lyap, use_fut=use_fut)])
    
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=TrainingConfigAblation.LEARNING_RATE,
        n_steps=TrainingConfigAblation.N_STEPS,
        batch_size=TrainingConfigAblation.BATCH_SIZE,
        n_epochs=TrainingConfigAblation.N_EPOCHS,
        gamma=TrainingConfigAblation.GAMMA,
        gae_lambda=TrainingConfigAblation.GAE_LAMBDA,
        clip_range=TrainingConfigAblation.CLIP_RANGE,
        ent_coef=TrainingConfigAblation.ENT_COEF,
        vf_coef=TrainingConfigAblation.VF_COEF,
        max_grad_norm=TrainingConfigAblation.MAX_GRAD_NORM,
        verbose=1,
        device=TrainingConfigAblation.DEVICE,
        tensorboard_log=str(log_dir)
    )
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=TrainingConfigAblation.SAVE_FREQ // TrainingConfigAblation.NUM_ENVS, save_path=str(save_dir / 'checkpoints'), name_prefix=f'{variant_name}', save_replay_buffer=False, save_vecnormalize=False),
        EvalCallback(eval_env, best_model_save_path=str(save_dir / 'best_model'), log_path=str(log_dir / 'eval'), eval_freq=TrainingConfigAblation.EVAL_FREQ // TrainingConfigAblation.NUM_ENVS, n_eval_episodes=10, deterministic=True, render=False, verbose=0),
        ActionLogCallback(log_freq=40000, log_file=f"actions_{variant_name}.txt"),
        EnhancedLoggingCallback(log_dir=log_dir, log_freq=5000, verbose=0)
    ])
    
    try:
        model.learn(total_timesteps=TrainingConfigAblation.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
        model.save(save_dir / 'final_model')
        print(f"\n✅ Training for {variant_name} completed successfully!")
    except KeyboardInterrupt:
        print(f"\n⚠️ Training for {variant_name} interrupted by user")
        model.save(save_dir / 'interrupted_model')
    finally:
        train_env.close()
        eval_env.close()

def main():
    # 1. Base PPO (No Lyapunov, No Future Info)
    train_ablation_variant("base_ppo", use_lyap=False, use_fut=False)
    
    # 2. PPO + Future Info (Pensieve-Oracle)
    train_ablation_variant("ppo_future", use_lyap=False, use_fut=True)
    
    # 3. PPO + Lyapunov (No Future Info)
    train_ablation_variant("ppo_lyapunov", use_lyap=True, use_fut=False)
    
    print("\n🎉 ALL ABLATION MODELS TRAINED SUCCESSFULLY! 🎉")
    print("Check 'results/models/' folder for 'ablation_base_ppo', 'ablation_ppo_future', and 'ablation_ppo_lyapunov'.")

if __name__ == '__main__':
    main()