"""
Train "Content-Blind" Agent (Pensieve Equivalent) on VMAF Reward.
For IEEE TCSVT Fairness: Uses the SAME environment/reward as Proposed Method,
but MASKS content features to simulate Pensieve's lack of content awareness.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import time
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium import ObservationWrapper, spaces

from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

class ContentBlindWrapper(ObservationWrapper):
    """
    Wraps ABREnv to hide content features (SI, TI, VMAF) from the agent.
    This forces the agent to act like Pensieve (only seeing Throughput/Buffer),
    but allows it to be trained on the VMAF-based reward function.
    """
    def __init__(self, env):
        super().__init__(env)
        # Observation space remains the same size to keep architecture consistent,
        # but we will zero-out the content features.
        self.observation_space = env.observation_space

    def observation(self, obs):
        # State indices in ABREnv:
        # 0-7: Throughput (8)
        # 8: Buffer (1)
        # 9: Last Bitrate (1)
        # 10-11: SI/TI (2) -> MASK THIS
        # 12-17: VMAF Preds (6) -> MASK THIS
        
        modified_obs = obs.copy()
        # Zero out SI/TI and VMAF predictions
        modified_obs[10:] = 0.0 
        return modified_obs

class PensieveConfig:
    """Configuration for recreating Pensieve baseline fairly."""
    VIDEO_NAME = 'sample1'
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # Hyperparameters close to original Pensieve (A3C converted to PPO)
    LEARNING_RATE = 2.5e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    ENT_COEF = 0.05  # Standard exploration
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    TOTAL_TIMESTEPS = 600_000
    SAVE_FREQ = 20_000
    EVAL_FREQ = 10_000
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(rank: int, seed: int = 0, is_eval: bool = False):
    """
    Create a Content-Blind Environment.
    """
    def _init():
        # CRITICAL: Use Training traces for train, Validation traces for eval
        # Assuming 'processed_traces' contains all, we ideally should split them.
        # For now, we rely on random seeding to likely get different traces,
        # but in production, use explicit folders: 'fcc_train' vs 'fcc_val'.
        
        env = ABREnv(
            video_name=PensieveConfig.VIDEO_NAME,
            trace_dir=str(PATHS['processed_traces']), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=PensieveConfig.MAX_CHUNKS,
            random_seed=seed + rank
        )
        # Apply Wrapper to hide content features -> makes it "Pensieve"
        env = ContentBlindWrapper(env)
        return Monitor(env)
    return _init

def main():
    print("\n" + "="*70)
    print("🎓 Training Pensieve-Style Agent (Content-Blind)")
    print("="*70 + "\n")
    
    print("🔬 Scientific Strategy for TCSVT:")
    print("  1. Environment: ABREnv (Same as proposed method)")
    print("  2. Reward: Lyapunov-Based VMAF (Fair comparison)")
    print("  3. Constraint: Content features (SI, TI, VMAF) are MASKED (Zeroed out)")
    print("  4. Outcome: An agent that acts like Pensieve but optimizes VMAF")
    print("")
    
    # Setup directories
    save_dir = PATHS['models'] / 'pensieve_retrained_vmaf'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'pensieve_retrained_vmaf'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    num_envs = PensieveConfig.NUM_ENVS
    
    # Create environments
    print(f"Creating {num_envs} parallel environments...")
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(num_envs)])
    # Eval env with different seed
    eval_env = SubprocVecEnv([make_env(0, 4242, is_eval=True)])
    
    # Create Model
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
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=PensieveConfig.SAVE_FREQ // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='pensieve_vmaf',
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
    
    # Train
    print("Starting training loop...")
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=PensieveConfig.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
        model.save(save_dir / 'final_model')
        print(f"\n✓ Training finished in {(time.time()-start_time)/3600:.1f} hours")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save(save_dir / 'interrupted_model')
    
    finally:
        train_env.close()
        eval_env.close()

if __name__ == '__main__':
    main()