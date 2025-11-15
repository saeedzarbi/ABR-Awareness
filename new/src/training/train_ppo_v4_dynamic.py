"""
Train PPO V4 with Buffer-Aware Dynamic Reward.
Key innovation: Reward weights adapt based on buffer safety level.
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

from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()


class TrainingConfigV4:
    """V4: Buffer-aware dynamic reward."""
    
    VIDEO_NAME = 'sample1'
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # PPO hyperparameters - OPTIMIZED FOR EXPLORATION
    LEARNING_RATE = 2e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.05          # Increased: 0.04 → 0.05 (more exploration)
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    TOTAL_TIMESTEPS = 600_000
    EVAL_FREQ = 10_000
    SAVE_FREQ = 20_000
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_env(rank: int, seed: int = 0):
    """Create environment instance."""
    def _init():
        env = ABREnv(
            video_name=TrainingConfigV4.VIDEO_NAME,
            trace_dir=str(PATHS['processed_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=TrainingConfigV4.MAX_CHUNKS,
            random_seed=seed + rank
        )
        return Monitor(env)
    return _init


def main():
    """Main training loop."""
    print("\n" + "="*70)
    print("🚀 Training PPO V4: Buffer-Aware Dynamic Reward")
    print("="*70 + "\n")
    
    print("🔧 V4 Key Innovation:")
    print("  Buffer-aware dynamic reward weights:")
    print("")
    print("  Buffer > 15s (SAFE):   Quality×3.5, Rebuffer×3.0")
    print("  Buffer 10-15s (GOOD):  Quality×3.0, Rebuffer×4.5")
    print("  Buffer 5-10s (MED):    Quality×2.5, Rebuffer×5.5")
    print("  Buffer < 5s (DANGER):  Quality×2.0, Rebuffer×7.0")
    print("")
    print("  Expected outcome:")
    print("    ✓ Aggressive when safe (high buffer)")
    print("    ✓ Conservative when risky (low buffer)")
    print("    ✓ Higher average bitrate (1000-1300 Kbps)")
    print("    ✓ Better quality (0.78-0.82)")
    print("")
    
    print("🔧 Other improvements:")
    print("  - Exploration: ENT_COEF 0.04 → 0.05")
    print("  - Smooth penalty: 0.3 → 0.25")
    print("  - Total timesteps: 600,000")
    print("")
    
    # Setup directories
    save_dir = PATHS['models'] / 'ppo_abr_v4'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_abr_v4'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = TrainingConfigV4.DEVICE
    num_envs = TrainingConfigV4.NUM_ENVS
    
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
    
    # Create PPO model
    print("Creating PPO V4 model...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=TrainingConfigV4.LEARNING_RATE,
        n_steps=TrainingConfigV4.N_STEPS,
        batch_size=TrainingConfigV4.BATCH_SIZE,
        n_epochs=TrainingConfigV4.N_EPOCHS,
        gamma=TrainingConfigV4.GAMMA,
        gae_lambda=TrainingConfigV4.GAE_LAMBDA,
        clip_range=TrainingConfigV4.CLIP_RANGE,
        ent_coef=TrainingConfigV4.ENT_COEF,
        vf_coef=TrainingConfigV4.VF_COEF,
        max_grad_norm=TrainingConfigV4.MAX_GRAD_NORM,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir)
    )
    print("✓ Model created\n")
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=TrainingConfigV4.SAVE_FREQ // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='ppo_v4',
        save_replay_buffer=False
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=TrainingConfigV4.EVAL_FREQ // num_envs,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # Training
    print("="*70)
    print("Starting training...")
    print(f"Total timesteps: {TrainingConfigV4.TOTAL_TIMESTEPS:,}")
    print(f"Estimated time: ~5-7 hours")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TrainingConfigV4.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
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
    print("Next: Evaluate V4 and compare with V3")
    print("  python src/evaluation/compare_v3_v4.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()