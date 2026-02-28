"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V6 tuning).

V6 keeps the same CMDP formulation as V5:

    maximize   E[ Σ_t  VMAF(a_t) ]
    subject to
        (C1)  E[ total_rebuffer / T ]          ≤  δ_rebuf
        (C2)  E[ Σ |VMAF_t − VMAF_{t-1}| / K ] ≤  δ_smooth

V6.1 fix (post-diagnosis): Previous V6 over-relaxed rebuffer, causing
lambda_rebuf to drift down and the policy to prefer 1200kbps almost always,
leading to excessive rebuffering. We tighten again:

    - rebuf_target      : 0.07  →  0.05  (stricter rebuffer budget)
    - lambda_rebuf_range : (2.0, 8.0) → (4.3, 12.0)
      * min 4.3 = eval weight, so we never penalize rebuffer less than at eval
      * max 12.0 so dual can grow when rebuffer is high
    - Base class dual update asymmetry is also fixed (constrained_abr.py).
"""

from pathlib import Path

import gymnasium as gym

from .constrained_abr import (
    DualVariableLogger,
    LagrangianRewardWrapper as _BaseLagrangianRewardWrapper,
)


class LagrangianRewardWrapperV6(_BaseLagrangianRewardWrapper):
    """
    V6-tuned Lagrangian reward wrapper.

    This subclass only changes the default hyperparameters passed to
    the base `LagrangianRewardWrapper`. The underlying dual update
    logic and reward structure are identical.
    """

    def __init__(
        self,
        env: gym.Env,
        rebuf_target: float = 0.05,
        smooth_target: float = 3.5,
        dual_lr_rebuf: float = 0.005,
        dual_lr_smooth: float = 0.003,
        lambda_rebuf_init: float = 4.3,
        lambda_smooth_init: float = 0.7,
        lambda_rebuf_range: tuple = (4.3, 12.0),
        lambda_smooth_range: tuple = (0.3, 2.5),
        buffer_dev_weight: float = 0.03,
        lyapunov_weight: float = 0.4,
        warmup_episodes: int = 80,
    ):
        super().__init__(
            env=env,
            rebuf_target=rebuf_target,
            smooth_target=smooth_target,
            dual_lr_rebuf=dual_lr_rebuf,
            dual_lr_smooth=dual_lr_smooth,
            lambda_rebuf_init=lambda_rebuf_init,
            lambda_smooth_init=lambda_smooth_init,
            lambda_rebuf_range=lambda_rebuf_range,
            lambda_smooth_range=lambda_smooth_range,
            buffer_dev_weight=buffer_dev_weight,
            lyapunov_weight=lyapunov_weight,
            warmup_episodes=warmup_episodes,
        )


__all__ = ["LagrangianRewardWrapperV6", "DualVariableLogger", "Path"]

