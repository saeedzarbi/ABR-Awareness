"""
Unified training script (v13) for QoE-recovery experiments.

V13 goal:
- Keep V12 CMDP dynamics/reward constraints.
- Train a high-QoE base policy and an optional QoE-aware guarded policy.
- Avoid default hysteresis because V12 shield-QoE increased switching.

Run:
  python new/src/training/train_all_models_v13.py --all
  python new/src/training/train_all_models_v13.py --models proposed_v13
"""

from __future__ import annotations

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
from src.environment.abr_multi_env_v13 import ABREnv
from src.training.constrained_abr_v13 import DualVariableLogger, LagrangianRewardWrapperV13
from src.training.safety_shield_v13 import ShieldConfigV13, SafetyShieldWrapperV13
from src.training.shield_aware_wrappers_v13 import (
    HysteresisActionWrapperV13,
    HysteresisConfigV13,
    ShieldAwarePenaltyWrapperV13,
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
    NUM_ENVS = int(os.environ.get("ABR_V13_NUM_ENVS", "8"))

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
    "proposed_v13": {
        "folder": "proposed_v13",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "shield": "off",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.045,
        "timesteps": int(os.environ.get("ABR_V13_STEPS_PROPOSED", "8000000")),
    },
    "proposed_v13_guarded": {
        "folder": "proposed_guarded_qoe_v13",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "shield": "qoe",
        "penalty": True,
        "hysteresis": False,
        "ent_coef": 0.045,
        "timesteps": int(os.environ.get("ABR_V13_STEPS_GUARDED", "8000000")),
    },
    "proposed_v13_guarded_hyst": {
        "folder": "proposed_guarded_hyst_v13",
        "use_lyapunov": True,
        "use_future": True,
        "blind_features": False,
        "use_lagrangian": True,
        "shield": "qoe",
        "penalty": True,
        "hysteresis": True,
        "ent_coef": 0.045,
        "timesteps": int(os.environ.get("ABR_V13_STEPS_GUARDED_HYST", "8000000")),
    },
    "pensieve_v13": {
        "folder": "pensieve_v13",
        "use_lyapunov": False,
        "use_future": False,
        "blind_features": True,
        "use_lagrangian": False,
        "shield": "off",
        "penalty": False,
        "hysteresis": False,
        "ent_coef": 0.03,
        "timesteps": int(os.environ.get("ABR_V13_STEPS_PENSIEVE", "3000000")),
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


def _shield_cfg() -> ShieldConfigV13:
    level = os.environ.get("ABR_V13_SHIELD_LEVEL", "qoe").strip().lower()
    if level not in {"off", "qoe", "light", "strong"}:
        level = "qoe"
    return ShieldConfigV13(
        level=level,
        catastrophic_ratio=float(os.environ.get("ABR_V13_CATASTROPHIC_RATIO", "2.50")),
        safe_margin_light=float(os.environ.get("ABR_V13_SAFE_MARGIN_LIGHT", "0.15")),
        safe_margin_strong=float(os.environ.get("ABR_V13_SAFE_MARGIN_STRONG", "0.75")),
        safety_tp_scale=float(os.environ.get("ABR_V13_TP_SCALE", "0.97")),
        critical_buffer_s=float(os.environ.get("ABR_V13_CRITICAL_BUFFER", "0.20")),
        only_when_risky=os.environ.get("ABR_V13_RISK_GATE", "1").strip().lower() in {"1", "true", "yes"},
        risky_dl_over_buf_ratio=float(os.environ.get("ABR_V13_RISK_RATIO", "1.35")),
        max_predicted_stall_s=float(os.environ.get("ABR_V13_ALLOWED_STALL", "0.25")),
        max_downgrade_steps=int(os.environ.get("ABR_V13_MAX_DOWNGRADE", "2")),
    )


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

        if spec.get("shield", "off") != "off":
            env = SafetyShieldWrapperV13(env, cfg=_shield_cfg())

        if spec.get("use_lagrangian") and not is_eval:
            env = LagrangianRewardWrapperV13(env)

        if spec.get("penalty", False) and not is_eval:
            beta = float(os.environ.get("ABR_V13_SHIELD_BETA", "0.05"))
            gamma = float(os.environ.get("ABR_V13_SHIELD_GAMMA", "0.06"))
            env = ShieldAwarePenaltyWrapperV13(env, beta_intervene=beta, gamma_deviation=gamma)

        if spec.get("hysteresis", False):
            cfg = HysteresisConfigV13(
                max_step=int(os.environ.get("ABR_V13_HYST_MAX_STEP", "2")),
                min_buf_for_upswitch=float(os.environ.get("ABR_V13_HYST_MIN_BUF", "1.0")),
            )
            env = HysteresisActionWrapperV13(env, cfg=cfg)

        if spec["blind_features"]:
            env = ContentBlindWrapper(env)

        return Monitor(env, info_keywords=("avg_quality", "total_rebuffer"))

    return _init


def train_one_model(model_key: str):
    spec = MODEL_SPECS[model_key]
    model_root = PATHS["models"] / "master_v13" / spec["folder"]
    log_root = PATHS["logs"] / "master_v13" / spec["folder"]
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(
        f"Training {model_key.upper()} (v13) | lyap={spec['use_lyapunov']} "
        f"| future={spec['use_future']} | blind={spec['blind_features']} "
        f"| lagrangian={spec.get('use_lagrangian', False)} "
        f"| shield={spec.get('shield', 'off')} | penalty={spec.get('penalty', False)} "
        f"| hysteresis={spec.get('hysteresis', False)} | ent={spec['ent_coef']} "
        f"| steps={spec['timesteps']:,}"
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

    callbacks = [
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
        callbacks.append(DualVariableLogger(log_dir=str(log_root), log_freq=5000))

    try:
        model.learn(total_timesteps=spec["timesteps"], callback=CallbackList(callbacks), progress_bar=True)
        model.save(model_root / "final_model")
        print(f"[DONE] {model_key} v13 training completed.")
    except KeyboardInterrupt:
        model.save(model_root / "interrupted_model")
        print(f"[INTERRUPTED] {model_key} v13 checkpoint saved.")
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

    print(f"\n[PARALLEL v13] Launching {len(remaining)} models, max {max_workers} concurrent")
    if gpu_ids:
        print(f"[PARALLEL v13] GPU pool: {gpu_ids}")

    while remaining or running:
        while remaining and len(running) < max_workers:
            key = remaining.pop(0)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[gpu_slot % len(gpu_ids)])
                gpu_slot += 1

            log_file = PATHS["logs"] / "master_v13" / f"{key}_parallel.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log_file, "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, script, "--models", key], env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[key] = (proc, fh)
            print(f"[PARALLEL v13] Started {key.upper()} (PID {proc.pid}) -> {log_file}")

        for key in list(running):
            proc, fh = running[key]
            ret = proc.poll()
            if ret is not None:
                fh.close()
                del running[key]
                if ret == 0:
                    completed.append(key)
                    print(f"[PARALLEL v13] {key.upper()} finished successfully")
                else:
                    failed.append(key)
                    print(f"[PARALLEL v13] {key.upper()} FAILED (exit code {ret})")

        if running:
            time.sleep(15)

    print("\n" + "=" * 72)
    print(f"[PARALLEL v13] Completed: {completed}")
    if failed:
        print(f"[PARALLEL v13] FAILED:    {failed}")
        sys.exit(1)
    print("All parallel v13 trainings finished.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Train ABR RL models v13.")
    parser.add_argument("--all", action="store_true", help="Train all v13 learning-based models.")
    parser.add_argument("--models", type=parse_model_list, default=["proposed_v13"], help="Comma separated model keys.")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-ids", type=str, default=None)
    args = parser.parse_args()

    order = list(MODEL_SPECS) if args.all else args.models
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None
    if args.parallel > 1 and len(order) > 1:
        _run_parallel(order, args.parallel, gpu_ids)
    else:
        for key in order:
            train_one_model(key)
        print("\n" + "=" * 72)
        print("All requested v13 trainings finished.")
        print("=" * 72)


if __name__ == "__main__":
    main()
