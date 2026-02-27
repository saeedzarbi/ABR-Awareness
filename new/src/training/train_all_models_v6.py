"""
Unified training script (V6) for all learning-based models:
- Proposed (Lyapunov + Future + Lagrangian CMDP, V6 tuning)
- Ablation Base (no Lyapunov, no Future, no CMDP)
- Ablation Future (Future only)
- Ablation Lyap (Lyapunov only)
- Pensieve (content/future blind PPO baseline)

V6 builds directly on top of the V5 setup with the following changes:
- Uses `ABREnv` from `abr_multi_env_v6` (slightly relaxed rebuffer /
  Lyapunov / buffer-deviation weights).
- Uses `LagrangianRewardWrapperV6` with tuned CMDP targets:
  rebuf_target=0.07, smooth_target=3.5, lambda_rebuf_range=(2, 8), etc.
- Increases entropy coefficient and training steps for Proposed:
  ent_coef = 0.04, timesteps = 8M (to avoid premature collapse to
  ultra-conservative policies).

All other implementation details (network architecture, SB3 configs,
logging, parallel training) are identical to V5.
"""

import argparse
import os
import subprocess
import sys
import time
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
from src.environment.abr_multi_env_v6 import ABREnv
from src.training.constrained_abr_v6 import (
    DualVariableLogger,
    LagrangianRewardWrapperV6,
)

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
    BATCH_SIZE = 512
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SAVE_FREQ = 50_000
    EVAL_FREQ = 20_000


MODEL_SPECS: Dict[str, Dict] = {
    "proposed": {
        "folder": "proposed_v6",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "ent_coef": 0.04,
        "timesteps": 8_000_000,
    },
    "ablation_base": {
        "folder": "ablation_base_v6",
        "use_lyapunov": False,
        "use_future": False,
        "blind_features": False,
        "use_lagrangian": False,
        "ent_coef": 0.03,
        "timesteps": 4_000_000,
    },
    "ablation_future": {
        "folder": "ablation_future_v6",
        "use_lyapunov": False,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": False,
        "ent_coef": 0.03,
        "timesteps": 4_000_000,
    },
    "ablation_lyap": {
        "folder": "ablation_lyap_v6",
        "use_lyapunov": True,
        "use_future": False,
        "blind_features": False,
        "use_lagrangian": False,
        "ent_coef": 0.03,
        "timesteps": 4_000_000,
    },
    "pensieve": {
        "folder": "pensieve_v6",
        "use_lyapunov": False,
        "use_future": False,
        "blind_features": True,
        "use_lagrangian": False,
        "ent_coef": 0.03,
        "timesteps": 3_000_000,
    },
}


class ContentBlindWrapper(ObservationWrapper):
    """Masks content-aware / future signals (indices 15:) for Pensieve."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        modified = obs.copy()
        modified[15:] = 0.0
        return modified


class EntropyWatchdog(BaseCallback):
    def __init__(self, min_entropy: float = 0.4, check_freq: int = 5000, verbose: int = 0):
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
        if spec.get("use_lagrangian") and not is_eval:
            env = LagrangianRewardWrapperV6(env)
        if spec["blind_features"]:
            env = ContentBlindWrapper(env)
        return Monitor(env, info_keywords=("avg_quality", "total_rebuffer"))

    return _init


def train_one_model(model_key: str):
    spec = MODEL_SPECS[model_key]
    model_root = PATHS["models"] / "master_v6" / spec["folder"]
    log_root = PATHS["logs"] / "master_v6" / spec["folder"]
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(
        f"Training {model_key.upper()} (V6) | lyap={spec['use_lyapunov']} "
        f"| future={spec['use_future']} | blind={spec['blind_features']} "
        f"| lagrangian={spec.get('use_lagrangian', False)} "
        f"| ent={spec['ent_coef']} | steps={spec['timesteps']:,}"
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
        ent_coef=spec["ent_coef"],
        vf_coef=Config.VF_COEF,
        max_grad_norm=Config.MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=Config.DEVICE,
        tensorboard_log=str(log_root),
    )

    cb_list = [
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
        EntropyWatchdog(min_entropy=0.4, check_freq=5000),
        TrainingLogger(log_dir=log_root),
    ]

    if spec.get("use_lagrangian"):
        cb_list.append(DualVariableLogger(log_dir=str(log_root), log_freq=5000))

    callbacks = CallbackList(cb_list)

    try:
        model.learn(
            total_timesteps=spec["timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
        model.save(model_root / "final_model")
        print(f"[DONE] {model_key} V6 training completed.")
    except KeyboardInterrupt:
        model.save(model_root / "interrupted_model")
        print(f"[INTERRUPTED] {model_key} V6 checkpoint saved.")
    finally:
        train_env.close()
        eval_env.close()


def parse_model_list(value: str) -> List[str]:
    keys = [v.strip().lower() for v in value.split(",") if v.strip()]
    invalid = [k for k in keys if k not in MODEL_SPECS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown model keys: {invalid}")
    return keys


# ---------------------------------------------------------------------------
#  Parallel training via subprocesses
# ---------------------------------------------------------------------------


def _run_parallel(model_keys: List[str], max_workers: int, gpu_ids: List[int] = None):
    script = str(Path(__file__).resolve())
    remaining = list(model_keys)
    running: Dict[str, subprocess.Popen] = {}
    completed, failed = [], []
    gpu_slot = 0

    print(f"\n[PARALLEL V6] Launching {len(remaining)} models, max {max_workers} concurrent")
    if gpu_ids:
        print(f"[PARALLEL V6] GPU pool: {gpu_ids}")

    while remaining or running:
        while remaining and len(running) < max_workers:
            key = remaining.pop(0)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[gpu_slot % len(gpu_ids)])
                gpu_slot += 1

            log_file = PATHS["logs"] / "master_v6" / f"{key}_parallel.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_file, "w", encoding="utf-8")

            proc = subprocess.Popen(
                [sys.executable, script, "--models", key],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            running[key] = (proc, fh)
            gpu_tag = f" [GPU {env.get('CUDA_VISIBLE_DEVICES', 'default')}]" if gpu_ids else ""
            print(f"[PARALLEL V6] Started {key.upper()} (PID {proc.pid}){gpu_tag}  -> {log_file}")

        for key in list(running):
            proc, fh = running[key]
            ret = proc.poll()
            if ret is not None:
                fh.close()
                del running[key]
                if ret == 0:
                    completed.append(key)
                    print(f"[PARALLEL V6] {key.upper()} finished successfully")
                else:
                    failed.append(key)
                    print(f"[PARALLEL V6] {key.upper()} FAILED (exit code {ret})")

        if running:
            time.sleep(15)

    print("\n" + "=" * 72)
    print(f"[PARALLEL V6] Completed: {completed}")
    if failed:
        print(f"[PARALLEL V6] FAILED:    {failed}")
        sys.exit(1)
    print("All parallel V6 trainings finished.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Train all ABR RL models V6 (CMDP + Lagrangian tuning)."
    )
    parser.add_argument("--all", action="store_true", help="Train all 5 models.")
    parser.add_argument(
        "--models",
        type=parse_model_list,
        default=["proposed"],
        help="Comma separated model keys.",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    args = parser.parse_args()

    if args.all:
        order = ["proposed", "ablation_base", "ablation_future", "ablation_lyap", "pensieve"]
    else:
        order = args.models

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None

    if args.parallel > 1 and len(order) > 1:
        _run_parallel(order, args.parallel, gpu_ids)
    else:
        for model_key in order:
            train_one_model(model_key)
        print("\n" + "=" * 72)
        print("All requested V6 trainings finished.")
        print("=" * 72)


if __name__ == "__main__":
    main()

