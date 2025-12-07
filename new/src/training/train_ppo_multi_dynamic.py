# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.parent))

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
# from stable_baselines3.common.monitor import Monitor
# import torch
# from src.environment.abr_multi_env import ABREnv
# from configs.paths import get_paths

# PATHS = get_paths()

# class TrainingConfigV4:
#     # --- VIDEO DATASET CONFIGURATION ---
#     # Training Set: Mixed content (Normal, High Motion, Zoom, Texture)
#     # The agent learns to handle these variations.
#     TRAIN_VIDEOS = [
#         'bigbuckbunny',    # Standard Baseline
#         'crowd_run',       # High Motion (High TI) -> Tests Buffer Stability
#         'tearsofsteel_short'        # Zoom/Pan -> Tests Prediction Robustness
#     ]
    
#     # Validation Set: Unseen video to test "Generalization"
#     # Ideally, 'park_joy' (High Detail) is good here.
#     # If you haven't downloaded it, use 'bigbuckbunny' or one from the train list.
#     TEST_VIDEOS = ['parkoy'] 
    
#     MAX_CHUNKS = 48
#     NUM_ENVS = 8
    
#     # --- BALANCED HYPERPARAMETERS ---
#     LEARNING_RATE = 3e-4    
#     N_STEPS = 4096          
#     BATCH_SIZE = 128
#     N_EPOCHS = 10
#     GAMMA = 0.98
#     GAE_LAMBDA = 0.95
#     CLIP_RANGE = 0.2
    
#     # Entropy: 0.1 is good for exploration across different video types
#     ENT_COEF = 0.10         
#     VF_COEF = 0.5
#     MAX_GRAD_NORM = 0.5
    
#     TOTAL_TIMESTEPS = 800_000
#     EVAL_FREQ = 10_000
#     SAVE_FREQ = 20_000
    
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# def make_env(rank: int, seed: int = 0, is_eval: bool = False):
#     """
#     Utility function for multiprocessed env.
    
#     :param is_eval: If True, uses TEST_VIDEOS and Test Traces.
#                     If False, uses TRAIN_VIDEOS and Train Traces.
#     """
#     def _init():
#         # 1. Select Video Set
#         if is_eval:
#             video_list = TrainingConfigV4.TEST_VIDEOS
#             trace_path = PATHS['test_traces']
#             mode_name = "EVAL"
#         else:
#             video_list = TrainingConfigV4.TRAIN_VIDEOS
#             trace_path = PATHS['train_traces']
#             mode_name = "TRAIN"
            
#         # Fallback if list is empty (prevent crash)
#         if not video_list:
#             video_list = ['bigbuckbunny']

#         # 2. Create Environment with list of videos
#         env = ABREnv(
#             video_names=video_list,  # <--- PASSING THE LIST HERE
#             trace_dir=str(trace_path), 
#             vmaf_dir=str(PATHS['vmaf_scores']),
#             siti_dir=str(PATHS['content_features']),
#             max_chunks=TrainingConfigV4.MAX_CHUNKS,
#             random_seed=seed + rank
#         )
#         return Monitor(env)
#     return _init

# def main():
#     print("\n" + "="*70)
#     print(f"🚀 Training PPO Multi-Dynamic: Multi-Video Content-Aware Mode")
#     print(f"📚 Training Videos: {TrainingConfigV4.TRAIN_VIDEOS}")
#     print(f"🧪 Evaluation Video: {TrainingConfigV4.TEST_VIDEOS}")
#     print("="*70)
    
#     save_dir = PATHS['models'] / 'ppo_abr_multi_dynamic_5'
#     save_dir.mkdir(parents=True, exist_ok=True)
#     log_dir = PATHS['logs'] / 'ppo_abr_multi_dynamic_5'
    
#     # Create Training Environments (8 parallel envs)
#     train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(TrainingConfigV4.NUM_ENVS)])
    
#     # Create Evaluation Environment
#     # We use the test traces and the unseen TEST_VIDEOS
#     if len(list(PATHS['test_traces'].glob('*.json'))) > 0:
#         eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=True)])
#     else:
#         print("⚠ Warning: No test traces found. Using training traces for eval.")
#         eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=False)])
    
#     model = PPO(
#         'MlpPolicy',
#         train_env,
#         learning_rate=TrainingConfigV4.LEARNING_RATE,
#         n_steps=TrainingConfigV4.N_STEPS,
#         batch_size=TrainingConfigV4.BATCH_SIZE,
#         n_epochs=TrainingConfigV4.N_EPOCHS,
#         gamma=TrainingConfigV4.GAMMA,
#         gae_lambda=TrainingConfigV4.GAE_LAMBDA,
#         clip_range=TrainingConfigV4.CLIP_RANGE,
#         ent_coef=TrainingConfigV4.ENT_COEF,
#         verbose=1,
#         device=TrainingConfigV4.DEVICE,
#         tensorboard_log=str(log_dir)
#     )
    
#     callbacks = CallbackList([
#         CheckpointCallback(
#             save_freq=TrainingConfigV4.SAVE_FREQ // TrainingConfigV4.NUM_ENVS, 
#             save_path=str(save_dir / 'checkpoints'), 
#             name_prefix='ppo_multi_dynamic_5'
#         ),
#         EvalCallback(
#             eval_env, 
#             best_model_save_path=str(save_dir / 'best_model'), 
#             log_path=str(log_dir / 'eval'), 
#             eval_freq=TrainingConfigV4.EVAL_FREQ // TrainingConfigV4.NUM_ENVS, 
#             n_eval_episodes=20, # Increased episodes to cover different videos/traces
#             deterministic=True
#         )
#     ])
    
#     print("Starting training...")
#     model.learn(total_timesteps=TrainingConfigV4.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
#     model.save(save_dir / 'final_model')
#     print("✓ Training completed.")

# if __name__ == '__main__':
#     main()
    
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

class ActionLogCallback(BaseCallback):
    """
    Log model actions (bitrate selections) to a text file periodically.
    """
    def __init__(self, log_freq: int = 10000, log_file: str = "training_actions.log", verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_path = PATHS['logs'] / log_file
        # Create/Clear the file
        with open(self.log_path, "w") as f:
            f.write("Step, Mean_Bitrate_Index, Last_Actions_Sample\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            actions = self.locals['actions'] # آرایه‌ای از اکشن‌ها (مثلاً 8 محیط)
            mean_action = np.mean(actions)
            
            with open(self.log_path, "a") as f:
                f.write(f"{self.num_timesteps}, {mean_action:.2f}, {list(actions)}\n")
            
            self.logger.record("custom/mean_action_idx", mean_action)
            
        return True

class TrainingConfigV5:
    TRAIN_VIDEOS = [
        'bigbuckbunny',    
        'crowd_run',       
        'tearsofsteel_short' 
    ]
    
    TEST_VIDEOS = ['parkjoy'] 
    
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # --- HYPERPARAMETERS ---
    LEARNING_RATE = 5e-5    
    
    N_STEPS = 4096          
    BATCH_SIZE = 256        
    N_EPOCHS = 4            
    CLIP_RANGE = 0.1        
    GAMMA = 0.99            
    GAE_LAMBDA = 0.95
    
    ENT_COEF = 0.05         
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    TOTAL_TIMESTEPS = 1_000_000 
    EVAL_FREQ = 20_000
    SAVE_FREQ = 50_000
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def make_env(rank: int, seed: int = 0, is_eval: bool = False):
    def _init():
        if is_eval:
            video_list = TrainingConfigV5.TEST_VIDEOS
            trace_path = PATHS['test_traces']
        else:
            video_list = TrainingConfigV5.TRAIN_VIDEOS
            trace_path = PATHS['train_traces']
            
        if not video_list: video_list = ['bigbuckbunny']

        env = ABREnv(
            video_names=video_list,
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=TrainingConfigV5.MAX_CHUNKS,
            random_seed=seed + rank
        )
        

        return Monitor(env, info_keywords=('avg_quality', 'total_rebuffer'))
    return _init

def main():
    print("\n" + "="*70)
    print(f"🚀 Training PPO with Detailed Action Logging")
    print("="*70)
    
    save_dir = PATHS['models'] / 'ppo_abr_multi_dynamic_8'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_abr_multi_dynamic_8'
    
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(TrainingConfigV5.NUM_ENVS)])
    
    if len(list(PATHS['test_traces'].glob('*.json'))) > 0:
        eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=True)])
    else:
        eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=False)])
    
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=TrainingConfigV5.LEARNING_RATE,
        n_steps=TrainingConfigV5.N_STEPS,
        batch_size=TrainingConfigV5.BATCH_SIZE,
        n_epochs=TrainingConfigV5.N_EPOCHS,
        gamma=TrainingConfigV5.GAMMA,
        gae_lambda=TrainingConfigV5.GAE_LAMBDA,
        clip_range=TrainingConfigV5.CLIP_RANGE,
        ent_coef=TrainingConfigV5.ENT_COEF,
        verbose=1,
        device=TrainingConfigV5.DEVICE,
        tensorboard_log=str(log_dir)
    )
    
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=TrainingConfigV5.SAVE_FREQ // TrainingConfigV5.NUM_ENVS, 
            save_path=str(save_dir / 'checkpoints'), 
            name_prefix='ppo_multi_dynamic_8'
        ),
        EvalCallback(
            eval_env, 
            best_model_save_path=str(save_dir / 'best_model'), 
            log_path=str(log_dir / 'eval'), 
            eval_freq=TrainingConfigV5.EVAL_FREQ // TrainingConfigV5.NUM_ENVS, 
            n_eval_episodes=20,
            deterministic=True
        ),
        ActionLogCallback(log_freq=5000, log_file="actions_history.txt") 
    ])
    
    print("Starting training...")
    model.learn(total_timesteps=TrainingConfigV5.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save(save_dir / 'final_model')
    print("✓ Training completed. Check 'logs/actions_history.txt' for details.")

if __name__ == '__main__':
    main()