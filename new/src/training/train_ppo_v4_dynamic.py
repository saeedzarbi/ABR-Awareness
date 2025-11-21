import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch
import time
import os

from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

class TrainingConfigV4:
    """
    V4 Config: Lyapunov-based Control for ABR.
    Optimized for stability and VMAF maximization.
    """
    
    # UPDATE: Changed video name to match new VMAF data
    VIDEO_NAME = 'bigbuckbunny'
    
    MAX_CHUNKS = 48
    NUM_ENVS = 8
    
    # PPO hyperparameters - Tuned for stability
    LEARNING_RATE = 2e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.98
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.05
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    TOTAL_TIMESTEPS = 600_000
    EVAL_FREQ = 10_000
    SAVE_FREQ = 20_000
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(rank: int, seed: int = 0, is_eval: bool = False):
    """
    Create environment instance.
    """
    def _init():
        # Use Training traces for training, Test traces for evaluation callback
        if is_eval:
            trace_path = PATHS['test_traces']
        else:
            trace_path = PATHS['train_traces']
            
        env = ABREnv(
            video_name=TrainingConfigV4.VIDEO_NAME,
            trace_dir=str(trace_path), 
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=TrainingConfigV4.MAX_CHUNKS,
            random_seed=seed + rank
        )
        return Monitor(env)
    return _init

def main():
    print("\n" + "="*70)
    print(f"🚀 Training PPO V4: Lyapunov-Based Control")
    print(f"📹 Target Video: {TrainingConfigV4.VIDEO_NAME}")
    print("="*70 + "\n")
    
    # Setup directories
    save_dir = PATHS['models'] / 'ppo_abr_v4_lyapunov'
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / 'ppo_abr_v4_lyapunov'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = TrainingConfigV4.DEVICE
    num_envs = TrainingConfigV4.NUM_ENVS
    
    # Check data availability
    print(f"📂 Training Data: {len(list(PATHS['train_traces'].glob('*.json')))} traces")
    
    # Create environments
    train_env = SubprocVecEnv([make_env(i, 0, is_eval=False) for i in range(num_envs)])
    
    # Use only 1 env for evaluation to save resources, ensure test traces exist
    if len(list(PATHS['test_traces'].glob('*.json'))) > 0:
        eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=True)])
    else:
        print("⚠ No test traces found for EvalCallback, using training traces instead.")
        eval_env = SubprocVecEnv([make_env(0, 1000, is_eval=False)])
    
    # Create PPO model
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
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir)
    )
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=TrainingConfigV4.SAVE_FREQ // num_envs,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix='ppo_lyapunov',
        save_replay_buffer=False
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(log_dir / 'eval'),
        eval_freq=TrainingConfigV4.EVAL_FREQ // num_envs,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # Training
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=TrainingConfigV4.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
        
        model.save(save_dir / 'final_model')
        print("✓ Training completed and model saved.")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted manually.")
        model.save(save_dir / 'interrupted_model')
        print("✓ Saved interrupted model.")
    
    finally:
        train_env.close()
        eval_env.close()

if __name__ == '__main__':
    main()