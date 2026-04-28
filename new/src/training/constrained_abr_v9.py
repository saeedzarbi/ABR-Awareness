"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V9).

V9 is designed as the "paper-ready" configuration:
- Stronger dual control to avoid lambda saturation.
- Targets chosen to produce low rebuffer in policy-only evaluation
  (so improvements don't rely on an external guard).
"""

import gymnasium as gym

from .constrained_abr import DualVariableLogger, LagrangianRewardWrapper as _BaseLagrangianRewardWrapper


class LagrangianRewardWrapperV9(_BaseLagrangianRewardWrapper):
    """
    V9-tuned Lagrangian wrapper.

    Key changes vs earlier versions:
    - Raise max lambda_rebuf (prevents saturation at high rebuffer).
    - Slightly higher dual_lr_rebuf for faster reaction.
    - Keep lambda_rebuf_min at eval weight (4.3) for comparability.
    """

    def __init__(
        self,
        env: gym.Env,
        rebuf_target: float = 0.05,
        smooth_target: float = 3.5,
        dual_lr_rebuf: float = 0.012,
        dual_lr_smooth: float = 0.003,
        lambda_rebuf_init: float = 6.0,
        lambda_smooth_init: float = 0.7,
        lambda_rebuf_range: tuple = (4.3, 40.0),
        lambda_smooth_range: tuple = (0.3, 2.5),
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


__all__ = ["LagrangianRewardWrapperV9", "DualVariableLogger"]

