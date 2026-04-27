"""
Multi-Video ABR Environment V8.

This repository previously only contained `abr_multi_env_v6.py`. V8 is introduced
as an explicit versioned entrypoint so training/evaluation scripts can be pinned
to a stable env import path.

For now, V8 is identical to V6 (alias). If you introduce true V8 reward weights,
dynamics, or observation changes, implement them here.
"""

from .abr_multi_env_v6 import ABREnv as _ABREnvV6


class ABREnv(_ABREnvV6):
    """V8 environment (currently identical to V6)."""


__all__ = ["ABREnv"]

