"""
Unified training script V2 — fixes policy collapse from V1.

Key changes:
  1. Uses abr_multi_env_v2 (VBR, np_random seeding, diverse init)
  2. ENT_COEF  0.01 → 0.05   (maintains exploration)
  3. Linear LR schedule        (3e-4 → 1e-5)
  4. Network   128,128 → 256,256
  5. Timesteps 2.4M → 5M
  6. EntropyWatchdog callback  (early warning for mode collapse)

Usage:
    python -m src.training.train_final_v2                   # Proposed only
    python -m src.training.train_final_v2 --all             # All 4 variants
    python -m src.training.train_final_v2 --variant base    # Ablation: Base PPO
    python -m src.training.train_final_v2 --variant future  # Ablation: PPO+Future
    python -m src.training.train_final_v2 --variant lyap    # Ablation: PPO+Lyapunov
"""

import sys
from pathlib import Path
import numpy as np
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import torch

from src.environment.abr_multi_env_v2 import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

VARIANTS = {
    'proposed': {
        'folder': 'ppo_proposed_v4',
        'use_lyapunov': True,
        'use_future': True,
    },
    'base': {
        'folder': 'ablation_base_ppo_v2',
        'use_lyapunov': False,
        'use_future': False,
    },
    'future': {
        'folder': 'ablation_ppo_future_v2',
        'use_lyapunov': False,
        'use_future': True,
    },
    'lyap': {
        'folder': 'ablation_ppo_lyapunov_v2',
        'use_lyapunov': True,
        'use_future': False,
    },
}


def linear_schedule(initial_value: float, final_value: float = 1e-5):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func


class Config:
    """Hyperparameters — fixes for policy collapse."""
    TRAIN_VIDEOS = ['bigbuckbunny', 'crowd_run', 'tearsofsteel_short']
    TEST_VIDEOS = ['sintel']
    MAX_CHUNKS = 48
    NUM_ENVS = 8

    LEARNING_RATE = linear_schedule(3e-4, 1e-5)
    N_STEPS = 4096
    BATCH_SIZE = 256
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.05
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    TOTAL_TIMESTEPS = 5_000_000
    EVAL_FREQ = 20_000
    SAVE_FREQ = 50_000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class EntropyWatchdog(BaseCallback):
    """Warns when action entropy drops dangerously low (mode collapse risk)."""

    def __init__(self, min_entropy: float = 0.5, check_freq: int = 5000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.min_entropy = min_entropy
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        actions = self.locals.get('actions', np.array([]))
        if len(actions) == 0:
            return True
        flat = actions.flatten().astype(int)
        dist = np.bincount(flat, minlength=6) / len(flat)
        entropy = -np.sum((dist + 1e-10) * np.log(dist + 1e-10))
        max_ent = np.log(6)
        if entropy < self.min_entropy:
            print(
                f"\n⚠️  LOW ENTROPY at step {self.num_timesteps}: "
                f"{entropy:.3f}/{max_ent:.2f}  "
                f"dist={[f'{p:.2f}' for p in dist]}"
            )
        return True


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
            f.write("step,mean_action,action_distribution,action_variance,"
                    "action_entropy\n")
        with open(self.episode_log, 'w') as f:
            f.write("step,episode_count,avg_reward,avg_vmaf,"
                    "avg_rebuffer_rate,episode_length\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            actions = self.locals.get('actions', [])
            if len(actions) > 0:
                flat = np.array(actions).flatten()
                dist = np.bincount(flat.astype(int), minlength=6) / len(flat)
                dist_safe = dist + 1e-10
                entropy = -np.sum(dist_safe * np.log(dist_safe))
                with open(self.action_log, 'a') as f:
                    dist_str = ';'.join(f'{p:.3f}' for p in dist)
                    f.write(
                        f"{self.num_timesteps},{np.mean(flat):.2f},"
                        f"{dist_str},{np.var(flat):.2f},{entropy:.3f}\n"
                    )

        for info in self.locals.get('infos', []):
            if 'episode' in info:
                ep = info['episode']
                vmaf = info.get('avg_quality', 0.0)
                rebuf = (
                    (info.get('total_rebuffer', 0.0) / (ep['l'] * 4.0)) * 100
                    if ep['l'] > 0 else 0
                )
                with open(self.episode_log, 'a') as f:
                    f.write(
                        f"{self.num_timesteps},{self.n_calls},"
                        f"{ep['r']:.2f},{vmaf:.2f},{rebuf:.2f},{ep['l']}\n"
                    )
        return True


def make_env(rank, seed=0, is_eval=False,
             use_lyapunov=True, use_future=True):
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
    print(f"  Training V2: {name.upper()}  |  Lyapunov={use_lyap}"
          f"  |  Future={use_fut}")
    print(f"  Save dir: results/models/{folder}")
    print(f"  ENT_COEF={Config.ENT_COEF}  BATCH={Config.BATCH_SIZE}"
          f"  STEPS={Config.TOTAL_TIMESTEPS:,}")
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
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
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
        EntropyWatchdog(min_entropy=0.5, check_freq=5000),
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
    parser = argparse.ArgumentParser(
        description="Train Proposed + Ablation models (V2 — fixed)"
    )
    parser.add_argument(
        '--variant',
        choices=['proposed', 'base', 'future', 'lyap'],
        default='proposed',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Train all 4 variants sequentially',
    )
    args = parser.parse_args()

    if args.all:
        for name in ['proposed', 'base', 'future', 'lyap']:
            train_variant(name)
        print("\n" + "=" * 70)
        print("  ALL 4 V2 MODELS TRAINED SUCCESSFULLY")
        print("=" * 70)
    else:
        train_variant(args.variant)


if __name__ == '__main__':
    main()
