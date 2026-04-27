"""
Constrained ABR via Lagrangian Primal-Dual Optimization (V8).

V8 is introduced as a clean, versioned training entrypoint. Currently it reuses
the tuned defaults from V6 (so behavior matches your existing setup), but with
a stable import path for v8 scripts.

If you modify CMDP targets / dual learning rates / lambda ranges for V8, do it
here and keep v8 scripts importing this module.
"""

from .constrained_abr_v6 import DualVariableLogger
from .constrained_abr_v6 import LagrangianRewardWrapperV6 as LagrangianRewardWrapperV8

__all__ = ["LagrangianRewardWrapperV8", "DualVariableLogger"]

