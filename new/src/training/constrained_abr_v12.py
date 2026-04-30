"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V12).
V12 reuses the stabilized V9/V10/V11 wrapper defaults. Any V12-specific changes
should be made here to keep experiments reproducible.
"""
from .constrained_abr_v11 import DualVariableLogger, LagrangianRewardWrapperV11
class LagrangianRewardWrapperV12(LagrangianRewardWrapperV11):
    """V12 wrapper (currently identical to V11)."""
__all__ = ["DualVariableLogger", "LagrangianRewardWrapperV12"]