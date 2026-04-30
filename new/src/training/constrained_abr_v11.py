"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V11).

V11 reuses the stabilized V9/V10 wrapper defaults. Any V11-specific changes
should be made here to keep experiments reproducible.
"""

from .constrained_abr_v10 import DualVariableLogger, LagrangianRewardWrapperV10


class LagrangianRewardWrapperV11(LagrangianRewardWrapperV10):
    """V11 wrapper (currently identical to V10)."""


__all__ = ["DualVariableLogger", "LagrangianRewardWrapperV11"]

