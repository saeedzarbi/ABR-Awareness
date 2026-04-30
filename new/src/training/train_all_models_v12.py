"""
Unified training script (v12) for all learning-based models (final/paper-ready).

V12 design goals:
- Safety-centric but QoE-preserving via method-level shield-in-the-loop.
- Mild hysteresis + mild shield-aware shaping (to avoid tuning loops).
- Train everything again (all learning-based models) with optional parallelism.

Run (single GPU):
  python new/src/training/train_all_models_v12.py --all

Run (parallel, multi-GPU):
  python new/src/training/train_all_models_v12.py --all --parallel 2 --gpu-ids 0,1
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.paths import get_paths
from src.environment.abr_multi_env_v12 import ABREnv
from src.training.constrained_abr_v12 import DualVariableLogger, LagrangianRewardWrapperV12
from src.training.safety_shield_v12 import SafetyShieldWrapper, ShieldConfig
from src.training.shield_aware_wrappers_v12 import (
    HysteresisActionWrapper,
    HysteresisConfig,
    ShieldAwarePenaltyWrapper,
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
    # Main method (unshielded policy, evaluated under policy-only guard scope if desired)
    "proposed": {
        "folder": "proposed_v12",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "shield": "off",  # off | classic | riskgate
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.04,
        "timesteps": 8_000_000,
    },
    # Safety-centric baseline: always-on shield
    "proposed_shielded": {
        "folder": "proposed_shielded_v12",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "shield": "classic",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.04,
        "timesteps": 8_000_000,
    },
    # Safety + QoE: mild hysteresis + mild shield-aware shaping
    "proposed_shielded_qoe": {
        "folder": "proposed_shielded_qoe_v12",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "shield": "classic",
        "penalty": True,
        "hysteresis": True,
        "ent_coef": 0.04,
        "timesteps": 8_000_000,
    },
    # Optional: risk-gated shield variant (often higher QoE, slightly lower safety)
    "proposed_shielded_riskgate": {
        "folder": "proposed_shielded_riskgate_v12",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "shield": "riskgate",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.04,
        "timesteps": 8_000_000,
    },
    "ablation_base": {
        "folder": "ablation_base_v12",
        "use_lyapunov": False,
        "use_future": False,
        "blind_features": False,
        "use_lagrangian": False,
        "shield": "off",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.03,
        "timesteps": 4_000_000,
    },
    "ablation_future": {
        "folder": "ablation_future_v12",
        "use_lyapunov": False,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": False,
        "shield": "off",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.03,
        "timesteps": 4_000_000,
    },
    "ablation_lyap": {
        "folder": "ablation_lyap_v12",
        "use_lyapunov": True,
        "use_future": False,
        "blind_features": False,
        "use_lagrangian": False,
        "shield": "off",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.03,
        "timesteps": 4_000_000,
    },
    "pensieve": {
        "folder": "pensieve_v12",
        "use_lyapunov": False,
        "use_future": False,
        "blind_features": True,
        "use_lagrangian": False,
        "shield": "off",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.03,
        "timesteps": 3_000_000,
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

    # Risk gate is enabled for "riskgate" variant (and optionally for classic by env var)
    only_when_risky = mode == "riskgate" or os.environ.get("ABR_V12_RISK_GATE", "0").strip().lower() in {"1", "true", "yes"}
    risky_ratio = float(os.environ.get("ABR_V12_RISK_RATIO", "1.10"))
    return ShieldConfig(level=level, only_when_risky=only_when_risky, risky_dl_over_buf_ratio=risky_ratio)


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

        # Mild hysteresis (before shield)
        if spec.get("hysteresis", False):
            cfg = HysteresisConfig(
                max_step=int(os.environ.get("ABR_HYST_MAX_STEP", "1")),
                min_buf_for_upswitch=float(os.environ.get("ABR_HYST_MIN_BUF", "1.5")),
            )
            env = HysteresisActionWrapper(env, cfg=cfg)

        # Shield in the loop
        if spec.get("shield", "off") != "off":
            env = SafetyShieldWrapper(env, cfg=_shield_cfg(model_key))

        # Constrained training (train env only)
        if spec.get("use_lagrangian") and not is_eval:
            env = LagrangianRewardWrapperV12(env)

        # Mild shield-aware penalty (train only)
        if spec.get("penalty", False) and not is_eval:
            beta = float(os.environ.get("ABR_SHIELD_BETA", "0.08"))
            gamma = float(os.environ.get("ABR_SHIELD_GAMMA", "0.03"))
            env = ShieldAwarePenaltyWrapper(env, beta_intervene=beta, gamma_deviation=gamma)

        if spec["blind_features"]:
            env = ContentBlindWrapper(env)

        return Monitor(env, info_keywords=("avg_quality", "total_rebuffer"))

    return _init


def train_one_model(model_key: str):
    spec = MODEL_SPECS[model_key]
    model_root = PATHS["models"] / "master_v12" / spec["folder"]
    log_root = PATHS["logs"] / "master_v12" / spec["folder"]
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(
        f"Training {model_key.upper()} (v12) | lyap={spec['use_lyapunov']} "
        f"| future={spec['use_future']} | blind={spec['blind_features']} "
        f"| lagrangian={spec.get('use_lagrangian', False)} "
        f"| shield={spec.get('shield', 'off')} "
        f"| penalty={spec.get('penalty', False)} "
        f"| hysteresis={spec.get('hysteresis', False)} "
        f"| ent={spec['ent_coef']} | steps={spec['timesteps']:,}"
    )
    print(f"Model dir: {model_root}")
    print(f"Log dir:   {log_root}")
    print("=" * 72)

    train_env = SubprocVecEnv([make_env(i, model_key, seed=0, is_eval=False) for i in range(Config.NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env(0, model_key, seed=1000, is_eval=True)])

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
    ]
    if spec.get("use_lagrangian"):
        cb_list.append(DualVariableLogger(log_dir=str(log_root), log_freq=5000))

    callbacks = CallbackList(cb_list)
    try:
        model.learn(total_timesteps=spec["timesteps"], callback=callbacks, progress_bar=True)
        model.save(model_root / "final_model")
        print(f"[DONE] {model_key} v12 training completed.")
    except KeyboardInterrupt:
        model.save(model_root / "interrupted_model")
        print(f"[INTERRUPTED] {model_key} v12 checkpoint saved.")
    finally:
        train_env.close()
        eval_env.close()


def parse_model_list(value: str) -> List[str]:
    keys = [v.strip().lower() for v in value.split(",") if v.strip()]
    invalid = [k for k in keys if k not in MODEL_SPECS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown model keys: {invalid}")
    return keys


def _run_parallel(model_keys: List[str], max_workers: int, gpu_ids: Optional[List[int]] = None):
    script = str(Path(__file__).resolve())
    remaining = list(model_keys)
    running: Dict[str, Tuple[subprocess.Popen, object]] = {}
    completed, failed = [], []
    gpu_slot = 0

    print(f"\n[PARALLEL v12] Launching {len(remaining)} models, max {max_workers} concurrent")
    if gpu_ids:
        print(f"[PARALLEL v12] GPU pool: {gpu_ids}")

    while remaining or running:
        while remaining and len(running) < max_workers:
            key = remaining.pop(0)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[gpu_slot % len(gpu_ids)])
                gpu_slot += 1

            log_file = PATHS["logs"] / "master_v12" / f"{key}_parallel.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_file, "w", encoding="utf-8")

            proc = subprocess.Popen(
                [sys.executable, script, "--models", key],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            running[key] = (proc, fh)
            print(f"[PARALLEL v12] Started {key.upper()} (PID {proc.pid}) -> {log_file}")

        for key in list(running):
            proc, fh = running[key]
            ret = proc.poll()
            if ret is not None:
                fh.close()
                del running[key]
                if ret == 0:
                    completed.append(key)
                    print(f"[PARALLEL v12] {key.upper()} finished successfully")
                else:
                    failed.append(key)
                    print(f"[PARALLEL v12] {key.upper()} FAILED (exit code {ret})")

        if running:
            time.sleep(15)

    print("\n" + "=" * 72)
    print(f"[PARALLEL v12] Completed: {completed}")
    if failed:
        print(f"[PARALLEL v12] FAILED:    {failed}")
        sys.exit(1)
    print("All parallel v12 trainings finished.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Train all ABR RL models v12 (final).")
    parser.add_argument("--all", action="store_true", help="Train all v12 learning-based models.")
    parser.add_argument(
        "--models",
        type=parse_model_list,
        default=["proposed_shielded_qoe"],
        help="Comma separated model keys.",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    args = parser.parse_args()

    if args.all:
        order = [
            "proposed",
            "proposed_shielded",
            "proposed_shielded_qoe",
            "proposed_shielded_riskgate",
            "ablation_base",
            "ablation_future",
            "ablation_lyap",
            "pensieve",
        ]
    else:
        order = args.models

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None
    if args.parallel > 1 and len(order) > 1:
        _run_parallel(order, args.parallel, gpu_ids)
    else:
        for k in order:
            train_one_model(k)
        print("\n" + "=" * 72)
        print("All requested v12 trainings finished.")
        print("=" * 72)


if __name__ == "__main__":
    main()

