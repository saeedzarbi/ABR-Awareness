import sys
from pathlib import Path
import numpy as np
import torch
import time
from gymnasium import ObservationWrapper

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.environment.abr_multi_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

class ContentBlindWrapper(ObservationWrapper):
    """
    Masks explicit content features (SI, TI, VMAF lookahead) to simulate Pensieve.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        # Indices 0-7: Throughput history (Keep)
        # Index 8: Buffer level (Keep)
        # Index 9: Last bitrate (Keep)
        # Indices 10-11: SI/TI (MASK -> 0)
        # Indices 12-17: VMAF lookahead (MASK -> 0)
        
        modified_obs = obs.copy()
        modified_obs[10:] = 0.0 
        return modified_obs

class PensieveConfig:
    # --- SAME DATASET AS PROPOSED METHOD ---
    TRAIN_VIDEOS = [
        'bigbuckbunny',    
        'crowd_run',       # High Motion
        'tearsofsteel_short'        # Zoom
    ]
    
    # Unseen video for fair generalization test
    TEST_VIDEOS = ['parkjoy'] 
    
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # Pensieve original hyperparameters
    LEARNING_RATE = 2.5e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    ENT_COEF = 0.05
    
    TOTAL_TIMESTEPS = 800_000
    SAVE_FREQ = 20_000
    EVAL_FREQ = 10_000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(rank: int, seed: int = 0, is_eval: bool = False):
    def _init():
        # 1. Select Video Set (Matching Proposed Method)
        if is_eval:
            video_list = PensieveConfig.TEST_VIDEOS
            trace_path = PATHS['test_traces']
        else:
            video_list = PensieveConfig.TRAIN_VIDEOS
            trace_path = PATHS['train_traces']
            
        if not video_list: video_list = ['bigbuckbunny']

        # 2. Create Environment
        env = ABREnv(
            video_names=video_list,  # <-- Using the list
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=PensieveConfig.MAX_CHUNKS,
            random_seed=seed + rank
        )
        
        # 3. Apply Wrapper to blind the agent
        env = ContentBlindWrapper(env)
        
        return Monitor(env)
    return _init

def main():
    print("\n" + "="*70)
    print("🎓 Training Pensieve-Style Agent (Content-Blind / Multi-Video)")
    print(f"📚 Training Videos: {PensieveConfig.TRAIN_VIDEOS}")
    print(f"🧪 Evaluation Video: {PensieveConfig.TEST_VIDEOS}")
    print("="*70 + "\n")
    
    save_dir = PATHS['models'] / 'pensieve_multi_vmaf_new_8'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'pensieve_multi_vmaf_new_8'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(PensieveConfig.NUM_ENVS)])
    
    if len(list(PATHS['test_traces'].glob('*.json'))) > 0:
        eval_env = SubprocVecEnv([make_env(0, 4242, is_eval=True)])
    else:
        print("⚠ Warning: Test traces not found, using train traces for eval.")
        eval_env = SubprocVecEnv([make_env(0, 4242, is_eval=False)])
    
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=PensieveConfig.LEARNING_RATE,
        n_steps=PensieveConfig.N_STEPS,
        batch_size=PensieveConfig.BATCH_SIZE,
        n_epochs=PensieveConfig.N_EPOCHS,
        gamma=PensieveConfig.GAMMA,
        ent_coef=PensieveConfig.ENT_COEF,
        verbose=1,
        device=PensieveConfig.DEVICE,
        tensorboard_log=str(log_dir)
    )
    
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=PensieveConfig.SAVE_FREQ // PensieveConfig.NUM_ENVS,
            save_path=str(save_dir / 'checkpoints'),
            name_prefix='pensieve_multi_vmaf_new_8'
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / 'best_model'),
            log_path=str(log_dir / 'eval'),
            eval_freq=PensieveConfig.EVAL_FREQ // PensieveConfig.NUM_ENVS,
            n_eval_episodes=20, # Increased for robust stats
            deterministic=True,
            verbose=1
        )
    ])
    
    print("Starting training loop...")
    model.learn(total_timesteps=PensieveConfig.TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save(save_dir / 'final_model')
    print("✓ Training finished.")

if __name__ == '__main__':
    main()