"""
Train PPO V3 with balanced reward function.
Emphasis on quality while maintaining low rebuffering.
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


class TrainingConfigV3:
    """V3: Balanced reward with quality emphasis."""
    
    VIDEO_NAME = 'sample1'
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # PPO hyperparameters - TUNED FOR BALANCE
    LEARNING_RATE = 2e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.04          # Increased exploration (was 0.03)
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    TOTAL_TIMESTEPS = 600_000  # More training
    EVAL_FREQ = 10_000
    SAVE_FREQ = 20_000
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_env(rank: int, seed: int = 0):
    """Create environment instance."""
    def _init():
        env = ABREnv(
            video_name=TrainingConfigV3.VIDEO_NAME,
            trace_dir=str(PATHS['processed_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=TrainingConfigV3.MAX_CHUNKS,
            random_seed=seed + rank
        )
        return Monitor(env)
    return _init


def main():
    """Main training loop."""
    print("\n" + "="*70)
    print("🚀 Training PPO V3: Balanced Quality-Stability Policy")
    print("="*70 + "\n")
    
    print("🔧 Key Improvements:")
    print("  1. Rebuffer penalty: 10.0 → 6.0 (less conservative)")
    print("  2. Quality weight: 1.0 → 2.0 (emphasize quality)")
    print("  3. Smooth penalty: 0.2 → 0.3 (discourage large jumps)")
    print("  4. Buffer-aware reward (bonus for healthy buffer)")
    print("  5. Free small bitrate changes (<= 2 levels)")
    print("  6. Exploration: ENT_COEF 0.03 → 0.04")
    print("  7. Total timesteps: 500K → 600K\n")
    
    # Setup directories
    save_dir = PATHS['models'] / 'ppo_abr_v3'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_abr_v3'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = TrainingConfigV3.DEVICE
    num_envs = TrainingConfigV3.NUM_ENVS
    
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
    print("Creating PPO V3 model...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=TrainingConfigV3.LEARNING_RATE,
        n_steps=TrainingConfigV3.N_STEPS,
        batch_size=TrainingConfigV3.BATCH_SIZE,
        n_epochs=TrainingConfigV3.N_EPOCHS,
        gamma=TrainingConfigV3.GAMMA,
        gae_lambda=TrainingConfigV3.GAE_LAMBDA,
        clip_range=TrainingConfigV3.CLIP_RANGE,
        ent_coef=TrainingConfigV3.ENT_COEF,
        vf_coef=TrainingConfigV3.VF_COEF,
        max_grad_norm=TrainingConfigV3.MAX_GRAD_NORM,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir)
    )
    print("✓ Model created\n")
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=TrainingConfigV3.SAVE_FREQ // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='ppo_v3',
        save_replay_buffer=False
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=TrainingConfigV3.EVAL_FREQ // num_envs,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # Training
    print("="*70)
    print("Starting training...")
    print(f"Total timesteps: {TrainingConfigV3.TOTAL_TIMESTEPS:,}")
    print(f"Estimated time: ~5-7 hours")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TrainingConfigV3.TOTAL_TIMESTEPS,
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
    print("Next: Evaluate V3")
    print("  python src/evaluation/quick_eval.py --model results/models/ppo_abr_v3/best_model/best_model")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()