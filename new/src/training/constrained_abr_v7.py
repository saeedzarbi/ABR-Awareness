"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V7).

V7 was referenced by the training script, but only the V6 implementation
was present in the repository. This module provides a stable v7 import path.

Currently, V7 defaults are identical to V6. If you update the CMDP targets
or dual update rules for a true v7, implement them here.
"""

from .constrained_abr_v6 import DualVariableLogger
from .constrained_abr_v6 import LagrangianRewardWrapperV6 as LagrangianRewardWrapperV7

__all__ = ["LagrangianRewardWrapperV7", "DualVariableLogger"]

