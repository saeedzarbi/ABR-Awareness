"""
Train PPO V2 with improved reward function.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch

from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()


def make_env(rank: int, seed: int = 0):
    """Create environment."""
    def _init():
        env = ABREnv(
            video_name='sample1',
            trace_dir=str(PATHS['processed_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=48,
            random_seed=seed + rank
        )
        return Monitor(env)
    return _init


def main():
    print("\n" + "="*70)
    print("🚀 Training PPO V2 with Improved Reward Function")
    print("="*70 + "\n")
    
    print("🔧 Improvements:")
    print("  1. Rebuffer penalty: 4.3 → 10.0")
    print("  2. Smooth penalty: 1.0 → 0.2")
    print("  3. Only penalize large bitrate jumps (>2 levels)")
    print("  4. Buffer-aware reward (penalty for low buffer)")
    print("  5. Exploration: ENT_COEF 0.01 → 0.03")
    print("  6. Total timesteps: 500,000\n")
    
    # Setup
    save_dir = PATHS['models'] / 'ppo_abr_v2'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_abr_v2'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 8
    
    print(f"💾 Save: {save_dir}")
    print(f"📊 Logs: {log_dir}")
    print(f"🖥️  Device: {device}")
    print(f"🔢 Envs: {num_envs}\n")
    
    # Create environments
    print("Creating environments...")
    train_env = SubprocVecEnv([make_env(i, 0) for i in range(num_envs)])
    eval_env = SubprocVecEnv([make_env(0, 1000)])
    print("✓ Done\n")
    
    # Create model
    print("Creating PPO model...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,  # Increased exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir)
    )
    print("✓ Done\n")
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=20000 // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='ppo_v2'
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=10000 // num_envs,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Train
    print("="*70)
    print("Starting training (500K timesteps, ~4-6 hours)...")
    print("="*70 + "\n")
    
    try:
        model.learn(
            total_timesteps=500000,
            callback=CallbackList([checkpoint_cb, eval_cb]),
            progress_bar=True
        )
        
        model.save(save_dir / 'final_model')
        print(f"\n✓ Saved to: {save_dir / 'final_model'}")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted - saving...")
        model.save(save_dir / 'interrupted_model')
        print(f"✓ Saved to: {save_dir / 'interrupted_model'}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    print("\n✓ Training completed!")


if __name__ == '__main__':
    main()