"""
Unified training script (V3) for all learning-based models:
- Proposed (Lyapunov + Future)
- Ablation Base (no Lyapunov, no Future)
- Ablation Future (Future only)
- Ablation Lyap (Lyapunov only)
- Pensieve (content/future blind PPO baseline)

Outputs are organized by model under:
  results/models/master_v3/<model_name>/
  results/logs/master_v3/<model_name>/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.environment.abr_multi_env_v2 import ABREnv

PATHS = get_paths()


def linear_schedule(initial_value: float, final_value: float = 1e-5):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func


class Config:
    TRAIN_VIDEOS = ["bigbuckbunny", "crowd_run", "tearsofsteel_short"]
    TEST_VIDEOS = ["sintel"]
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
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TOTAL_TIMESTEPS_DEFAULT = 5_000_000
    TOTAL_TIMESTEPS_PENSIEVE = 2_500_000
    SAVE_FREQ = 50_000
    EVAL_FREQ = 20_000


MODEL_SPECS: Dict[str, Dict] = {
    "proposed": {
        "folder": "proposed",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "timesteps": Config.TOTAL_TIMESTEPS_DEFAULT,
    },
    "ablation_base": {
        "folder": "ablation_base",
        "use_lyapunov": False,
        "use_future": False,
        "blind_features": False,
        "timesteps": Config.TOTAL_TIMESTEPS_DEFAULT,
    },
    "ablation_future": {
        "folder": "ablation_future",
        "use_lyapunov": False,
        "use_future": True,
        "blind_features": False,
        "timesteps": Config.TOTAL_TIMESTEPS_DEFAULT,
    },
    "ablation_lyap": {
        "folder": "ablation_lyap",
        "use_lyapunov": True,
        "use_future": False,
        "blind_features": False,
        "timesteps": Config.TOTAL_TIMESTEPS_DEFAULT,
    },
    "pensieve": {
        "folder": "pensieve",
        "use_lyapunov": False,
        "use_future": False,
        "blind_features": True,
        "timesteps": Config.TOTAL_TIMESTEPS_PENSIEVE,
    },
}


class ContentBlindWrapper(ObservationWrapper):
    """
    Masks explicit content/future signals in ABREnv V2 observation.
    ABREnv V2 layout:
      0:12 throughput history
      12 buffer
      13 buffer trend
      14 last bitrate
      15:17 SI/TI
      17:23 VMAF lookahead
      23:29 next chunk sizes
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        modified = obs.copy()
        modified[15:] = 0.0
        return modified


class EntropyWatchdog(BaseCallback):
    def __init__(self, min_entropy: float = 0.5, check_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.min_entropy = min_entropy
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        actions = self.locals.get("actions", np.array([]))
        if len(actions) == 0:
            return True
        flat = np.array(actions).flatten().astype(int)
        dist = np.bincount(flat, minlength=6) / len(flat)
        entropy = -np.sum((dist + 1e-10) * np.log(dist + 1e-10))
        if entropy < self.min_entropy:
            print(
                f"\n[WARN] Low action entropy at {self.num_timesteps}: {entropy:.3f} | "
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
        with open(self.action_log, "w", encoding="utf-8") as f:
            f.write("step,mean_action,action_distribution,action_variance,action_entropy\n")
        with open(self.episode_log, "w", encoding="utf-8") as f:
            f.write("step,episode_count,avg_reward,avg_vmaf,avg_rebuffer_rate,episode_length\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            actions = self.locals.get("actions", [])
            if len(actions) > 0:
                flat = np.array(actions).flatten()
                dist = np.bincount(flat.astype(int), minlength=6) / len(flat)
                entropy = -np.sum((dist + 1e-10) * np.log(dist + 1e-10))
                with open(self.action_log, "a", encoding="utf-8") as f:
                    dist_str = ";".join(f"{p:.3f}" for p in dist)
                    f.write(
                        f"{self.num_timesteps},{np.mean(flat):.2f},"
                        f"{dist_str},{np.var(flat):.2f},{entropy:.3f}\n"
                    )

        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                vmaf = info.get("avg_quality", 0.0)
                rebuf = (
                    (info.get("total_rebuffer", 0.0) / (ep["l"] * 4.0)) * 100.0
                    if ep["l"] > 0
                    else 0.0
                )
                with open(self.episode_log, "a", encoding="utf-8") as f:
                    f.write(
                        f"{self.num_timesteps},{self.n_calls},"
                        f"{ep['r']:.2f},{vmaf:.2f},{rebuf:.2f},{ep['l']}\n"
                    )
        return True


def make_env(rank: int, model_key: str, seed: int = 0, is_eval: bool = False):
    spec = MODEL_SPECS[model_key]

    def _init():
        videos = Config.TEST_VIDEOS if is_eval else Config.TRAIN_VIDEOS
        traces = PATHS["test_traces"] if is_eval else PATHS["train_traces"]
        env = ABREnv(
            video_names=videos,
            trace_dir=str(traces),
            vmaf_dir=str(PATHS["vmaf_scores"]),
            siti_dir=str(PATHS["content_features"]),
            max_chunks=Config.MAX_CHUNKS,
            random_seed=seed + rank,
            use_lyapunov=spec["use_lyapunov"],
            use_future=spec["use_future"],
        )
        if spec["blind_features"]:
            env = ContentBlindWrapper(env)
        return Monitor(env, info_keywords=("avg_quality", "total_rebuffer"))

    return _init


def train_one_model(model_key: str):
    spec = MODEL_SPECS[model_key]
    model_root = PATHS["models"] / "master_v3" / spec["folder"]
    log_root = PATHS["logs"] / "master_v3" / spec["folder"]
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(
        f"Training {model_key.upper()} | lyap={spec['use_lyapunov']} "
        f"| future={spec['use_future']} | blind={spec['blind_features']}"
    )
    print(f"Model dir: {model_root}")
    print(f"Log dir:   {log_root}")
    print("=" * 72)

    train_env = SubprocVecEnv(
        [make_env(i, model_key, seed=0, is_eval=False) for i in range(Config.NUM_ENVS)]
    )
    eval_env = SubprocVecEnv([make_env(0, model_key, seed=1000, is_eval=True)])

    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.Tanh,
    }

    model = PPO(
        "MlpPolicy",
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
        tensorboard_log=str(log_root),
    )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=Config.SAVE_FREQ // Config.NUM_ENVS,
                save_path=str(model_root / "checkpoints"),
                name_prefix=spec["folder"],
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(model_root / "best_model"),
                log_path=str(log_root / "eval"),
                eval_freq=Config.EVAL_FREQ // Config.NUM_ENVS,
                n_eval_episodes=10,
                deterministic=True,
            ),
            EntropyWatchdog(min_entropy=0.5, check_freq=5000),
            TrainingLogger(log_dir=log_root),
        ]
    )

    try:
        model.learn(
            total_timesteps=spec["timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
        model.save(model_root / "final_model")
        print(f"[DONE] {model_key} training completed.")
    except KeyboardInterrupt:
        model.save(model_root / "interrupted_model")
        print(f"[INTERRUPTED] {model_key} checkpoint saved.")
    finally:
        train_env.close()
        eval_env.close()


def parse_model_list(value: str) -> List[str]:
    keys = [v.strip().lower() for v in value.split(",") if v.strip()]
    invalid = [k for k in keys if k not in MODEL_SPECS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown model keys: {invalid}")
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Train all ABR RL models (Proposed/Ablations/Pensieve) in one script."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 5 models sequentially.",
    )
    parser.add_argument(
        "--models",
        type=parse_model_list,
        default=["proposed"],
        help=(
            "Comma separated model keys, e.g. "
            "'proposed,ablation_base,ablation_future,ablation_lyap,pensieve'"
        ),
    )
    args = parser.parse_args()

    if args.all:
        order = ["proposed", "ablation_base", "ablation_future", "ablation_lyap", "pensieve"]
    else:
        order = args.models

    for model_key in order:
        train_one_model(model_key)

    print("\n" + "=" * 72)
    print("All requested trainings finished.")
    print("=" * 72)


if __name__ == "__main__":
    main()
