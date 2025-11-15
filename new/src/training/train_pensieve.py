"""
Train Pensieve-style PPO agent.
Replicates Pensieve approach with PPO instead of A3C.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch
import time

from src.environment.pensieve_env import PensieveEnv
from configs.paths import get_paths

PATHS = get_paths()


class PensieveConfig:
    """Pensieve training configuration."""
    
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # PPO hyperparameters (similar to Pensieve's A3C)
    LEARNING_RATE = 3e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.08  # Less exploration than our V4
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    TOTAL_TIMESTEPS = 800_000
    EVAL_FREQ = 10_000
    SAVE_FREQ = 20_000
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_env(rank: int, seed: int = 0):
    """Create Pensieve environment."""
    def _init():
        env = PensieveEnv(
            trace_dir=str(PATHS['processed_traces']),
            max_chunks=PensieveConfig.MAX_CHUNKS,
            random_seed=seed + rank
        )
        return Monitor(env)
    return _init


def main():
    """Train Pensieve-style agent."""
    print("\n" + "="*70)
    print("🎓 Training Pensieve-Style PPO Agent")
    print("="*70 + "\n")
    
    print("📝 Pensieve Characteristics:")
    print("  - Simple linear reward (no content-awareness)")
    print("  - State: past throughput + past bitrates + buffer")
    print("  - Original: A3C, Our version: PPO")
    print("  - Reward: Quality - 4.3×Rebuffer - 1.0×Smooth")
    print("")
    
    # Setup
    save_dir = PATHS['models'] / 'pensieve'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'pensieve'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = PensieveConfig.DEVICE
    num_envs = PensieveConfig.NUM_ENVS
    
    print(f"💾 Models: {save_dir}")
    print(f"📊 Logs: {log_dir}")
    print(f"🖥️  Device: {device}")
    print(f"🔢 Parallel envs: {num_envs}\n")
    
    # Create environments
    print("Creating environments...")
    train_env = SubprocVecEnv([
        make_env(i, 0) for i in range(num_envs)
    ])
    eval_env = SubprocVecEnv([make_env(0, 1000)])
    print("✓ Environments ready\n")
    
    # Create model
    print("Creating Pensieve PPO model...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=PensieveConfig.LEARNING_RATE,
        n_steps=PensieveConfig.N_STEPS,
        batch_size=PensieveConfig.BATCH_SIZE,
        n_epochs=PensieveConfig.N_EPOCHS,
        gamma=PensieveConfig.GAMMA,
        gae_lambda=PensieveConfig.GAE_LAMBDA,
        clip_range=PensieveConfig.CLIP_RANGE,
        ent_coef=PensieveConfig.ENT_COEF,
        vf_coef=PensieveConfig.VF_COEF,
        max_grad_norm=PensieveConfig.MAX_GRAD_NORM,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir)
    )
    print("✓ Model created\n")
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=PensieveConfig.SAVE_FREQ // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='pensieve',
        save_replay_buffer=False
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=PensieveConfig.EVAL_FREQ // num_envs,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # Training
    print("="*70)
    print("Starting training...")
    print(f"Total timesteps: {PensieveConfig.TOTAL_TIMESTEPS:,}")
    print(f"Estimated time: ~5 hours")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=PensieveConfig.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
        
        final_path = save_dir / 'final_model'
        model.save(final_path)
        
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        
        print(f"\n✓ Training completed in {hours:.1f} hours")
        print(f"✓ Final model: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted")
        interrupted_path = save_dir / 'interrupted_model'
        model.save(interrupted_path)
        print(f"✓ Saved to: {interrupted_path}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    print("\n" + "="*70)
    print("Next: Compare Pensieve with our methods")
    print("  python src/evaluation/comprehensive_comparison.py --episodes 20")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()