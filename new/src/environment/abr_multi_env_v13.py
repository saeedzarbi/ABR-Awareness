"""
Multi-Video ABR Environment V13.

V13 intentionally keeps the V12/V11 plant dynamics unchanged. The experiment
delta lives in the runtime guard and train/eval composition so result changes
can be attributed to the QoE-oriented shielding design.
"""

from .abr_multi_env_v12 import ABREnv as _ABREnvV12


class ABREnv(_ABREnvV12):
    """V13 environment (dynamics identical to V12)."""


__all__ = ["ABREnv"]
