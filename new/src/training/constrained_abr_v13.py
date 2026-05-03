"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V13).

The Lagrangian wrapper is kept identical to V12. V13 changes are isolated to
runtime guarding and experiment composition.
"""

from .constrained_abr_v12 import DualVariableLogger, LagrangianRewardWrapperV12


class LagrangianRewardWrapperV13(LagrangianRewardWrapperV12):
    """V13 wrapper (currently identical to V12)."""


__all__ = ["DualVariableLogger", "LagrangianRewardWrapperV13"]
