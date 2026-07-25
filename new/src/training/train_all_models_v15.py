"""
Unified training script (V15, low-latency operating point).

Identical to train_all_models_v14.py except:
  * uses abr_multi_env_v15 (buffer cap 6 s instead of 30 s), and
  * saves under models/master_v15_lowlat/.

This is a full standalone script (not a monkey-patch of v14) on purpose: PPO uses
SubprocVecEnv, and env-class overrides applied only in the parent process do not
propagate to the pickled env-factory that runs in the worker processes. Keeping a
self-contained module guarantees every worker builds the low-latency environment.

Run (single seed, all arms) on a CPU server:
  cd new
  python src/training/train_all_models_v15.py --all --seeds 0 --num-envs 8
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.environment.abr_multi_env_v15 import ABREnv
from src.training.constrained_abr_v14 import (
    ConstraintDiagnosticsLogger,
    LagrangianRewardWrapperV14,
)
from src.training.safety_shield_v14 import SafetyShieldWrapper, ShieldConfig
from src.training.shield_aware_wrappers_v12 import (
    HysteresisActionWrapper,
    HysteresisConfig,
    ShieldAwarePenaltyWrapper,
)

PATHS = get_paths()

REBUF_TARGET = 0.05
SMOOTH_TARGET = 3.5
MODEL_TAG = "master_v15_lowlat"


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
    DEVICE = "cpu"

    SAVE_FREQ = 50_000
    EVAL_FREQ = 20_000


MODEL_SPECS: Dict[str, Dict] = {
    "proposed": {
        "folder": "proposed_v14", "use_lyapunov": True, "use_future": True,
        "blind_features": False, "use_lagrangian": True, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.04, "timesteps": 2_000_000,
    },
    "proposed_shielded": {
        "folder": "proposed_shielded_v14", "use_lyapunov": True, "use_future": True,
        "blind_features": False, "use_lagrangian": True, "shield": "classic",
        "penalty": False, "hysteresis": False, "ent_coef": 0.04, "timesteps": 2_000_000,
    },
    "proposed_shielded_qoe": {
        "folder": "proposed_shielded_qoe_v14", "use_lyapunov": True, "use_future": True,
        "blind_features": False, "use_lagrangian": True, "shield": "classic",
        "penalty": True, "hysteresis": True, "ent_coef": 0.04, "timesteps": 2_000_000,
    },
    "proposed_shielded_riskgate": {
        "folder": "proposed_shielded_riskgate_v14", "use_lyapunov": True, "use_future": True,
        "blind_features": False, "use_lagrangian": True, "shield": "riskgate",
        "penalty": False, "hysteresis": False, "ent_coef": 0.04, "timesteps": 2_000_000,
    },
    "ablation_base": {
        "folder": "ablation_base_v14", "use_lyapunov": False, "use_future": False,
        "blind_features": False, "use_lagrangian": False, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.03, "timesteps": 1_000_000,
    },
    "ablation_future": {
        "folder": "ablation_future_v14", "use_lyapunov": False, "use_future": True,
        "blind_features": False, "use_lagrangian": False, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.03, "timesteps": 1_000_000,
    },
    "ablation_lyap": {
        "folder": "ablation_lyap_v14", "use_lyapunov": True, "use_future": False,
        "blind_features": False, "use_lagrangian": False, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.03, "timesteps": 1_000_000,
    },
    "pensieve": {
        "folder": "pensieve_v14", "use_lyapunov": False, "use_future": False,
        "blind_features": True, "use_lagrangian": False, "shield": "off",
        "penalty": False, "hysteresis": False, "ent_coef": 0.03, "timesteps": 1_000_000,
    },
}


class ContentBlindWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        modified = obs.copy()
        modified[15:] = 0.0
        return modified


def _shield_cfg(model_key: str) -> ShieldConfig:
    level = os.environ.get("ABR_SHIELD_LEVEL", "light").strip().lower()
    if level not in {"off", "light", "strong"}:
        level = "light"
    mode = MODEL_SPECS[model_key].get("shield", "off")
    if mode == "off":
        return ShieldConfig(level="off")
    only_when_risky = mode == "riskgate" or os.environ.get("ABR_V14_RISK_GATE", "0").strip().lower() in {"1", "true", "yes"}
    risky_ratio = float(os.environ.get("ABR_V14_RISK_RATIO", "1.10"))
    return ShieldConfig(level=level, only_when_risky=only_when_risky, risky_dl_over_buf_ratio=risky_ratio)


def make_env(rank: int, model_key: str, base_seed: int = 0, is_eval: bool = False):
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
            random_seed=(base_seed + 1000 if is_eval else base_seed) + rank,
            use_lyapunov=spec["use_lyapunov"],
            use_future=spec["use_future"],
        )

        if spec.get("hysteresis", False):
            cfg = HysteresisConfig(
                max_step=int(os.environ.get("ABR_HYST_MAX_STEP", "1")),
                min_buf_for_upswitch=float(os.environ.get("ABR_HYST_MIN_BUF", "1.5")),
            )
            env = HysteresisActionWrapper(env, cfg=cfg)

        if spec.get("shield", "off") != "off":
            env = SafetyShieldWrapper(env, cfg=_shield_cfg(model_key))

        if spec.get("use_lagrangian") and not is_eval:
            env = LagrangianRewardWrapperV14(
                env, rebuf_target=REBUF_TARGET, smooth_target=SMOOTH_TARGET
            )

        if spec.get("penalty", False) and not is_eval:
            beta = float(os.environ.get("ABR_SHIELD_BETA", "0.08"))
            gamma = float(os.environ.get("ABR_SHIELD_GAMMA", "0.03"))
            env = ShieldAwarePenaltyWrapper(env, beta_intervene=beta, gamma_deviation=gamma)

        if spec["blind_features"]:
            env = ContentBlindWrapper(env)

        return Monitor(env, info_keywords=("avg_quality", "total_rebuffer"))

    return _init


def train_one_model(model_key: str, seed: int, timesteps_scale: float, num_envs: int):
    spec = MODEL_SPECS[model_key]
    seed_tag = f"seed_{seed}"
    model_root = PATHS["models"] / MODEL_TAG / spec["folder"] / seed_tag
    log_root = PATHS["logs"] / MODEL_TAG / spec["folder"] / seed_tag
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    total_timesteps = max(1, int(spec["timesteps"] * timesteps_scale))

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("\n" + "=" * 72)
    print(
        f"Training {model_key.upper()} (v15 low-latency) seed={seed} | "
        f"lagr={spec.get('use_lagrangian')} | shield={spec.get('shield')} | "
        f"steps={total_timesteps:,} | envs={num_envs} | buffer_cap={ABREnv.BUFFER_MAX}s"
    )
    print(f"Model dir: {model_root}")
    print("=" * 72)

    train_env = SubprocVecEnv(
        [make_env(i, model_key, base_seed=seed, is_eval=False) for i in range(num_envs)]
    )
    eval_env = SubprocVecEnv([make_env(0, model_key, base_seed=seed, is_eval=True)])

    policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": torch.nn.Tanh}

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
        seed=seed,
        device=Config.DEVICE,
        tensorboard_log=str(log_root),
    )

    cb_list = [
        CheckpointCallback(
            save_freq=max(1, Config.SAVE_FREQ // num_envs),
            save_path=str(model_root / "checkpoints"),
            name_prefix=spec["folder"],
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_root / "best_model"),
            log_path=str(log_root / "eval"),
            eval_freq=max(1, Config.EVAL_FREQ // num_envs),
            n_eval_episodes=10,
            deterministic=True,
        ),
    ]
    if spec.get("use_lagrangian"):
        cb_list.append(
            ConstraintDiagnosticsLogger(
                log_dir=str(log_root), log_freq=5000,
                rebuf_target=REBUF_TARGET, smooth_target=SMOOTH_TARGET,
            )
        )

    callbacks = CallbackList(cb_list)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
        model.save(model_root / "final_model")
        print(f"[DONE] {model_key} v15 seed={seed} completed.")
    except KeyboardInterrupt:
        model.save(model_root / "interrupted_model")
        print(f"[INTERRUPTED] {model_key} v15 seed={seed} checkpoint saved.")
    finally:
        train_env.close()
        eval_env.close()


def parse_model_list(value: str) -> List[str]:
    keys = [v.strip().lower() for v in value.split(",") if v.strip()]
    invalid = [k for k in keys if k not in MODEL_SPECS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown model keys: {invalid}")
    return keys


def parse_seed_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def _run_parallel(model_keys: List[str], seeds: List[int], max_workers: int,
                  timesteps_scale: float, num_envs: int):
    script = str(Path(__file__).resolve())
    remaining: List[Tuple[str, int]] = [(m, s) for m in model_keys for s in seeds]
    running: Dict[str, Tuple[subprocess.Popen, object]] = {}
    completed, failed = [], []
    print(f"\n[PARALLEL v15] {len(remaining)} (model,seed) jobs, max {max_workers} concurrent")
    while remaining or running:
        while remaining and len(running) < max_workers:
            key, seed = remaining.pop(0)
            tag = f"{key}_seed{seed}"
            log_file = PATHS["logs"] / MODEL_TAG / f"{tag}_parallel.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_file, "w", encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable, script, "--models", key, "--seeds", str(seed),
                 "--timesteps-scale", str(timesteps_scale), "--num-envs", str(num_envs)],
                env=os.environ.copy(), stdout=fh, stderr=subprocess.STDOUT,
            )
            running[tag] = (proc, fh)
            print(f"[PARALLEL v15] Started {tag} (PID {proc.pid}) -> {log_file}")
        for tag in list(running):
            proc, fh = running[tag]
            ret = proc.poll()
            if ret is not None:
                fh.close()
                del running[tag]
                (completed if ret == 0 else failed).append(tag)
                print(f"[PARALLEL v15] {tag} {'OK' if ret == 0 else f'FAILED ({ret})'}")
        if running:
            time.sleep(15)
    print(f"\n[PARALLEL v15] Completed: {completed}")
    if failed:
        print(f"[PARALLEL v15] FAILED: {failed}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train all ABR RL models (v15, low-latency).")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--models", type=parse_model_list, default=["proposed_shielded_qoe"])
    parser.add_argument("--seeds", type=parse_seed_list, default=[0])
    parser.add_argument("--timesteps-scale", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=Config.NUM_ENVS)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))

    if args.all:
        order = [
            "proposed", "proposed_shielded", "proposed_shielded_qoe",
            "proposed_shielded_riskgate", "ablation_base", "ablation_future",
            "ablation_lyap", "pensieve",
        ]
    else:
        order = args.models

    n_jobs = len(order) * len(args.seeds)
    if args.parallel > 1 and n_jobs > 1:
        _run_parallel(order, args.seeds, args.parallel, args.timesteps_scale, args.num_envs)
    else:
        for k in order:
            for s in args.seeds:
                train_one_model(k, seed=s, timesteps_scale=args.timesteps_scale, num_envs=args.num_envs)
        print("\nAll requested v15 (low-latency) trainings finished.")


if __name__ == "__main__":
    main()
