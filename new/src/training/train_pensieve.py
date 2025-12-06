"""
Train "Content-Blind" Agent (Pensieve Equivalent) on VMAF Reward.
"""

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

from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

class ContentBlindWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        # Mask content features (indices 10 onwards)
        modified_obs = obs.copy()
        modified_obs[10:] = 0.0 
        return modified_obs

class PensieveConfig:
    # UPDATE: Use same video as Proposed Method
    VIDEO_NAME = 'bigbuckbunny'
    
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    LEARNING_RATE = 2.5e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    ENT_COEF = 0.05
    
    TOTAL_TIMESTEPS = 600_000
    SAVE_FREQ = 20_000
    EVAL_FREQ = 10_000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(rank: int, seed: int = 0, is_eval: bool = False):
    def _init():
        if is_eval:
            trace_path = PATHS['test_traces']
        else:
            trace_path = PATHS['train_traces']
            
        env = ABREnv(
            video_name=PensieveConfig.VIDEO_NAME,
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=PensieveConfig.MAX_CHUNKS,
            random_seed=seed + rank
        )
        env = ContentBlindWrapper(env)
        return Monitor(env)
    return _init

def main():
    print("\n" + "="*70)
    print("🎓 Training Pensieve-Style Agent (Content-Blind)")
    print(f"📹 Target Video: {PensieveConfig.VIDEO_NAME}")
    print("="*70 + "\n")
    
    save_dir = PATHS['models'] / 'pensieve_retrained_vmaf_2'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'pensieve_retrained_vmaf_2'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    num_envs = PensieveConfig.NUM_ENVS
    
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(num_envs)])
    
    if len(list(PATHS['test_traces'].glob('*.json'))) > 0:
        eval_env = SubprocVecEnv([make_env(0, 4242, is_eval=True)])
    else:
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
    
    checkpoint_cb = CheckpointCallback(
        save_freq=PensieveConfig.SAVE_FREQ // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='pensieve_vmaf_2',
        save_replay_buffer=False
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=PensieveConfig.EVAL_FREQ // num_envs,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    print("Starting training loop...")
    try:
        model.learn(
            total_timesteps=PensieveConfig.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
        model.save(save_dir / 'final_model')
        print("✓ Training finished.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        model.save(save_dir / 'interrupted_model')
    
    finally:
        train_env.close()
        eval_env.close()

if __name__ == '__main__':
    main()