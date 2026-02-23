"""
Unified training script for the Proposed model and all Ablation variants.

Usage:
    python -m src.training.train_final                   # Train Proposed only
    python -m src.training.train_final --all             # Train Proposed + 3 Ablation
    python -m src.training.train_final --variant base    # Train Base PPO only
    python -m src.training.train_final --variant future  # Train PPO+Future only
    python -m src.training.train_final --variant lyap    # Train PPO+Lyapunov only

All variants share the same hyperparameters and architecture (Table II).
Only use_lyapunov and use_future toggles differ.
"""

import sys
from pathlib import Path
import numpy as np
import argparse
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
import torch
from src.environment.abr_multi_env_l import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

VARIANTS = {
    'proposed': {
        'folder': 'ppo_proposed_v3_lyapunov',
        'use_lyapunov': True,
        'use_future': True,
    },
    'base': {
        'folder': 'ablation_base_ppo',
        'use_lyapunov': False,
        'use_future': False,
    },
    'future': {
        'folder': 'ablation_ppo_future',
        'use_lyapunov': False,
        'use_future': True,
    },
    'lyap': {
        'folder': 'ablation_ppo_lyapunov',
        'use_lyapunov': True,
        'use_future': False,
    },
}


class Config:
    """Hyperparameters matching Table II of the paper."""
    TRAIN_VIDEOS = ['bigbuckbunny', 'crowd_run', 'tearsofsteel_short']
    TEST_VIDEOS = ['sintel']
    MAX_CHUNKS = 48
    NUM_ENVS = 8

    LEARNING_RATE = 2.5e-4
    N_STEPS = 4096
    BATCH_SIZE = 128
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    TOTAL_TIMESTEPS = 2_400_000   # 50,000 episodes x 48 steps
    EVAL_FREQ = 20_000
    SAVE_FREQ = 50_000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainingLogger(BaseCallback):
    def __init__(self, log_dir: Path, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.action_log = self.log_dir / "actions_detailed.csv"
        self.episode_log = self.log_dir / "episodes_detailed.csv"
        self._init_files()

    def _init_files(self):
        with open(self.action_log, 'w') as f:
            f.write("step,mean_action,action_distribution,action_variance,action_entropy\n")
        with open(self.episode_log, 'w') as f:
            f.write("step,episode_count,avg_reward,avg_vmaf,avg_rebuffer_rate,episode_length\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            actions = self.locals.get('actions', [])
            if len(actions) > 0:
                dist = np.bincount(actions, minlength=6) / len(actions)
                dist_safe = dist + 1e-10
                entropy = -np.sum(dist_safe * np.log(dist_safe))
                with open(self.action_log, 'a') as f:
                    dist_str = ';'.join(f'{p:.3f}' for p in dist)
                    f.write(f"{self.num_timesteps},{np.mean(actions):.2f},"
                            f"{dist_str},{np.var(actions):.2f},{entropy:.3f}\n")

        for info in self.locals.get('infos', []):
            if 'episode' in info:
                ep = info['episode']
                vmaf = info.get('avg_quality', 0.0)
                rebuf = (info.get('total_rebuffer', 0.0) / (ep['l'] * 4.0)) * 100 if ep['l'] > 0 else 0
                with open(self.episode_log, 'a') as f:
                    f.write(f"{self.num_timesteps},{self.n_calls},"
                            f"{ep['r']:.2f},{vmaf:.2f},{rebuf:.2f},{ep['l']}\n")
        return True


def make_env(rank, seed=0, is_eval=False, use_lyapunov=True, use_future=True):
    def _init():
        videos = Config.TEST_VIDEOS if is_eval else Config.TRAIN_VIDEOS
        traces = PATHS['test_traces'] if is_eval else PATHS['train_traces']
        env = ABREnv(
            video_names=videos,
            trace_dir=str(traces),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=Config.MAX_CHUNKS,
            random_seed=seed + rank,
            use_lyapunov=use_lyapunov,
            use_future=use_future,
        )
        return Monitor(env, info_keywords=('avg_quality', 'total_rebuffer'))
    return _init


def train_variant(name: str):
    cfg = VARIANTS[name]
    folder = cfg['folder']
    use_lyap = cfg['use_lyapunov']
    use_fut = cfg['use_future']

    print("\n" + "=" * 70)
    print(f"  Training: {name.upper()}  |  Lyapunov={use_lyap}  |  Future={use_fut}")
    print(f"  Save dir: results/models/{folder}")
    print("=" * 70)

    save_dir = PATHS['models'] / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS['logs'] / folder
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv([
        make_env(i, 0, False, use_lyap, use_fut)
        for i in range(Config.NUM_ENVS)
    ])
    eval_env = SubprocVecEnv([
        make_env(0, 1000, True, use_lyap, use_fut)
    ])

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=Config.LEARNING_RATE,
        n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE,
        n_epochs=Config.N_EPOCHS,
        gamma=Config.GAMMA,
        gae_lambda=Config.GAE_LAMBDA,
        clip_range=Config.CLIP_RANGE,
        ent_coef=Config.ENT_COEF,
        vf_coef=Config.VF_COEF,
        max_grad_norm=Config.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=Config.DEVICE,
        tensorboard_log=str(log_dir),
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=Config.SAVE_FREQ // Config.NUM_ENVS,
            save_path=str(save_dir / 'checkpoints'),
            name_prefix=folder,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / 'best_model'),
            log_path=str(log_dir / 'eval'),
            eval_freq=Config.EVAL_FREQ // Config.NUM_ENVS,
            n_eval_episodes=10,
            deterministic=True,
        ),
        TrainingLogger(log_dir=log_dir),
    ])

    try:
        model.learn(
            total_timesteps=Config.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )
        model.save(save_dir / 'final_model')
        print(f"\n  >>> {name.upper()} training complete!")
    except KeyboardInterrupt:
        model.save(save_dir / 'interrupted_model')
        print(f"\n  >>> {name.upper()} interrupted, model saved.")
    finally:
        train_env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Proposed + Ablation models")
    parser.add_argument(
        '--variant',
        choices=['proposed', 'base', 'future', 'lyap'],
        default='proposed',
        help='Which variant to train (default: proposed)',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Train all 4 variants sequentially (Proposed + 3 Ablation)',
    )
    args = parser.parse_args()

    if args.all:
        for name in ['proposed', 'base', 'future', 'lyap']:
            train_variant(name)
        print("\n" + "=" * 70)
        print("  ALL 4 MODELS TRAINED SUCCESSFULLY")
        print("=" * 70)
    else:
        train_variant(args.variant)


if __name__ == '__main__':
    main()
