"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V8).

V8 is introduced as a clean, versioned training entrypoint. Currently it reuses
the tuned defaults from V6 (so behavior matches your existing setup), but with
a stable import path for v8 scripts.

If you modify CMDP targets / dual learning rates / lambda ranges for V8, do it
here and keep v8 scripts importing this module.
"""

import gymnasium as gym

from .constrained_abr import DualVariableLogger, LagrangianRewardWrapper as _BaseLagrangianRewardWrapper

class LagrangianRewardWrapperV8(_BaseLagrangianRewardWrapper):
    """
    V8-tuned Lagrangian reward wrapper.

    v8 observation from your logs:
    - lambda_rebuf rapidly hits its max (12.0) while evaluation still shows
      very high rebuffer in raw mode. That is a clear sign the dual variable
      is saturating and cannot enforce the constraint.

    This wrapper increases the feasible range for lambda_rebuf and makes the
    dual update more responsive so the learned policy itself becomes safer
    (not just relying on the inference-time guard).
    """

    def __init__(
        self,
        env: gym.Env,
        # Targets (ratio = rebuffer_seconds / video_seconds)
        rebuf_target: float = 0.05,
        smooth_target: float = 3.5,
        # Dual learning rates (more responsive than V6)
        dual_lr_rebuf: float = 0.010,
        dual_lr_smooth: float = 0.003,
        # Init lambdas
        lambda_rebuf_init: float = 4.3,
        lambda_smooth_init: float = 0.7,
        # Key change: raise max lambda_rebuf to avoid saturation
        lambda_rebuf_range: tuple = (4.3, 30.0),
        lambda_smooth_range: tuple = (0.3, 2.5),
        # Keep env-aligned shaping terms
        buffer_dev_weight: float = 0.03,
        lyapunov_weight: float = 0.4,
        warmup_episodes: int = 60,
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


__all__ = ["LagrangianRewardWrapperV8", "DualVariableLogger"]

