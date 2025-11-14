"""
Train PPO agent for content-aware ABR.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
import torch
import numpy as np
from typing import Optional
import os

# Import environment
from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()


class TrainingConfig:
    """Training hyperparameters."""
    
    # Environment settings
    VIDEO_NAME = 'sample1'
    MAX_CHUNKS = 48
    NUM_ENVS = 4  # Parallel environments
    
    # PPO hyperparameters
    LEARNING_RATE = 3e-4
    N_STEPS = 2048  # Steps per environment per update
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01  # Entropy coefficient (exploration)
    VF_COEF = 0.5    # Value function coefficient
    MAX_GRAD_NORM = 0.5
    
    # Training settings
    TOTAL_TIMESTEPS = 500_000  # Total training steps
    EVAL_FREQ = 10_000         # Evaluate every N steps
    SAVE_FREQ = 20_000         # Save checkpoint every N steps
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_env(rank: int, seed: int = 0):
    """
    Create a single environment instance.
    
    Args:
        rank: Environment ID
        seed: Random seed
        
    Returns:
        Callable that creates the environment
    """
    def _init():
        env = ABREnv(
            video_name=TrainingConfig.VIDEO_NAME,
            trace_dir=str(PATHS['processed_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=TrainingConfig.MAX_CHUNKS,
            random_seed=seed + rank
        )
        env = Monitor(env)
        return env
    return _init


def create_parallel_envs(num_envs: int = 4, start_seed: int = 0):
    """
    Create parallel environments for faster training.
    
    Args:
        num_envs: Number of parallel environments
        start_seed: Starting random seed
        
    Returns:
        Vectorized environment
    """
    if num_envs == 1:
        return DummyVecEnv([make_env(0, start_seed)])
    else:
        # Use SubprocVecEnv for true parallelism
        return SubprocVecEnv([
            make_env(i, start_seed) 
            for i in range(num_envs)
        ])


def train_ppo(
    total_timesteps: int = TrainingConfig.TOTAL_TIMESTEPS,
    num_envs: int = TrainingConfig.NUM_ENVS,
    save_dir: Optional[str] = None,
    verbose: int = 1
):
    """
    Train PPO agent.
    
    Args:
        total_timesteps: Total training timesteps
        num_envs: Number of parallel environments
        save_dir: Directory to save models
        verbose: Verbosity level
    """
    print("\n" + "="*60)
    print("🚀 Training Content-Aware ABR Agent with PPO")
    print("="*60 + "\n")
    
    # Setup directories
    if save_dir is None:
        save_dir = PATHS['models'] / 'ppo_abr'
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_abr'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Save directory: {save_dir}")
    print(f"📊 Log directory: {log_dir}")
    print(f"🖥️  Device: {TrainingConfig.DEVICE}")
    print(f"🔢 Parallel environments: {num_envs}\n")
    
    # Create environments
    print("Creating environments...")
    train_env = create_parallel_envs(num_envs=num_envs, start_seed=0)
    eval_env = create_parallel_envs(num_envs=1, start_seed=1000)
    print("✓ Environments created\n")
    
    # Print training configuration
    print("Training Configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Learning rate: {TrainingConfig.LEARNING_RATE}")
    print(f"  Batch size: {TrainingConfig.BATCH_SIZE}")
    print(f"  N steps: {TrainingConfig.N_STEPS}")
    print(f"  N epochs: {TrainingConfig.N_EPOCHS}")
    print(f"  Gamma: {TrainingConfig.GAMMA}")
    print(f"  Entropy coef: {TrainingConfig.ENT_COEF}\n")
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=TrainingConfig.LEARNING_RATE,
        n_steps=TrainingConfig.N_STEPS,
        batch_size=TrainingConfig.BATCH_SIZE,
        n_epochs=TrainingConfig.N_EPOCHS,
        gamma=TrainingConfig.GAMMA,
        gae_lambda=TrainingConfig.GAE_LAMBDA,
        clip_range=TrainingConfig.CLIP_RANGE,
        ent_coef=TrainingConfig.ENT_COEF,
        vf_coef=TrainingConfig.VF_COEF,
        max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
        verbose=verbose,
        device=TrainingConfig.DEVICE,
        tensorboard_log=str(log_dir)
    )
    print("✓ Model created\n")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=TrainingConfig.SAVE_FREQ // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='ppo_abr',
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=TrainingConfig.EVAL_FREQ // num_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    # Start training
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = save_dir / 'final_model'
        model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        interrupted_model_path = save_dir / 'interrupted_model'
        model.save(interrupted_model_path)
        print(f"✓ Model saved to: {interrupted_model_path}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("  1. Evaluate model: python src/evaluation/evaluate.py")
    print("  2. View logs: tensorboard --logdir results/logs/ppo_abr")


def quick_test():
    """Quick test with minimal training."""
    print("\n🧪 Quick Test Mode (10K timesteps)\n")
    
    train_ppo(
        total_timesteps=10_000,
        num_envs=2,
        verbose=1
    )


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent for ABR')
    parser.add_argument(
        '--timesteps',
        type=int,
        default=500_000,
        help='Total training timesteps (default: 500K)'
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=4,
        help='Number of parallel environments (default: 4)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with 10K timesteps'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save models'
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        train_ppo(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            save_dir=args.save_dir
        )


if __name__ == '__main__':
    main()