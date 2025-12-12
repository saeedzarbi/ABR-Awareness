import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
from src.environment.abr_multi_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

# ============================================================================
# Enhanced Logging Callback
# ============================================================================

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
            
            self.logger.record("actions/mean", mean_action)
            self.logger.record("actions/variance", action_var)
            self.logger.record("actions/entropy", entropy)
            for i, prob in enumerate(action_dist):
                self.logger.record(f"actions/bitrate_{i}_prob", prob)
    
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
                
                self.logger.record("episode/avg_reward", avg_reward)
                self.logger.record("episode/avg_vmaf", avg_vmaf)
                self.logger.record("episode/rebuffer_rate", rebuffer_rate)
                self.logger.record("episode/length", ep_length)

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
            self.logger.record("custom/mean_action_idx", mean_action)
        return True

# ============================================================================
# Training Configuration V21
# ============================================================================

class TrainingConfigV10:
    """
    Training configuration for V21 (Pure Quality / No CrowdRun)
    """
    
    TRAIN_VIDEOS = [
        'bigbuckbunny',    
        'crowd_run',
        'tearsofsteel_short' 
    ]
    
    TEST_VIDEOS = ['sintel'] 
    
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # --- Hyperparameters ---
    LEARNING_RATE = 3e-4
    N_STEPS = 4096          
    BATCH_SIZE = 128
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    
    # Lower entropy (0.01) to converge fast to High Bitrate strategies
    # ENT_COEF = 0.01         
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    ENT_COEF = 0.03 # Balanced exploration
    TOTAL_TIMESTEPS = 1_000_000 
    EVAL_FREQ = 20_000
    SAVE_FREQ = 50_000
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(rank: int, seed: int = 0, is_eval: bool = False):
    def _init():
        if is_eval:
            video_list = TrainingConfigV10.TEST_VIDEOS
            trace_path = PATHS['test_traces']
        else:
            video_list = TrainingConfigV10.TRAIN_VIDEOS
            trace_path = PATHS['train_traces']
            
        if not video_list: video_list = ['bigbuckbunny']

        env = ABREnv(
            video_names=video_list,
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=TrainingConfigV10.MAX_CHUNKS,
            random_seed=seed + rank
        )
        return Monitor(env, info_keywords=('avg_quality', 'total_rebuffer'))
    return _init

def main():
    print("\n" + "="*70)
    print(f"🚀 Training PPO V21: Pure Quality (No CrowdRun)")
    print("="*70)
    print(f"📚 Training Videos: {TrainingConfigV10.TRAIN_VIDEOS}")
    print(f"🧪 Test Videos: {TrainingConfigV10.TEST_VIDEOS}")
    print("\n📊 Configuration:")
    print(f"   REBUF_PENALTY_BASE: 1.0 (Very Low Risk)") 
    print(f"   SMOOTH_PENALTY: 1.0 (High Stability)")
    print(f"   Entropy Coef: {TrainingConfigV10.ENT_COEF} (Fast Convergence)")
    print(f"   Total Timesteps: {TrainingConfigV10.TOTAL_TIMESTEPS:,}")
    print("="*70 + "\n")
    
    save_dir = PATHS['models'] / 'ppo_abr_multi_dynamic_21'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_abr_multi_dynamic_21'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(TrainingConfigV10.NUM_ENVS)])
    
    if len(list(PATHS['test_traces'].glob('*.json'))) > 0:
        eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=True)])
    else:
        print("⚠️ Warning: No test traces found. Using training traces for eval.")
        eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=False)])
    
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=TrainingConfigV10.LEARNING_RATE,
        n_steps=TrainingConfigV10.N_STEPS,
        batch_size=TrainingConfigV10.BATCH_SIZE,
        n_epochs=TrainingConfigV10.N_EPOCHS,
        gamma=TrainingConfigV10.GAMMA,
        gae_lambda=TrainingConfigV10.GAE_LAMBDA,
        clip_range=TrainingConfigV10.CLIP_RANGE,
        ent_coef=TrainingConfigV10.ENT_COEF,
        vf_coef=TrainingConfigV10.VF_COEF,
        max_grad_norm=TrainingConfigV10.MAX_GRAD_NORM,
        verbose=1,
        device=TrainingConfigV10.DEVICE,
        tensorboard_log=str(log_dir)
    )
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=TrainingConfigV10.SAVE_FREQ // TrainingConfigV10.NUM_ENVS, save_path=str(save_dir / 'checkpoints'), name_prefix='ppo_multi_dynamic_21', save_replay_buffer=False, save_vecnormalize=False),
        EvalCallback(eval_env, best_model_save_path=str(save_dir / 'best_model'), log_path=str(log_dir / 'eval'), eval_freq=TrainingConfigV10.EVAL_FREQ // TrainingConfigV10.NUM_ENVS, n_eval_episodes=20, deterministic=True, render=False, verbose=1),
        ActionLogCallback(log_freq=40000, log_file="actions_history.txt"),
        EnhancedLoggingCallback(log_dir=log_dir, log_freq=5000, verbose=1)
    ])
    
    try:
        model.learn(total_timesteps=TrainingConfigV10.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
        model.save(save_dir / 'final_model')
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        model.save(save_dir / 'interrupted_model')
    
    finally:
        train_env.close()
        eval_env.close()

if __name__ == '__main__':
    main()