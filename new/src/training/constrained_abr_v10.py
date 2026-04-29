"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V10).

V10 reuses the stabilized V9 wrapper defaults. Any V10-specific changes
should be made here to keep experiments reproducible.
"""

from pathlib import Path

from .constrained_abr_v9 import DualVariableLogger, LagrangianRewardWrapperV9


class LagrangianRewardWrapperV10(LagrangianRewardWrapperV9):
    """V10 wrapper (currently identical to V9)."""


__all__ = ["DualVariableLogger", "LagrangianRewardWrapperV10", "Path"]

