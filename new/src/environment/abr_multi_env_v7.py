"""
Multi-Video ABR Environment V7

This project originally referenced a v7 environment, but only v6 existed
in the repository. To make training/evaluation versioning explicit and
reproducible, v7 is defined as a thin, named alias of v6.

If you later introduce true v7 dynamics/weights, update this file (and keep
`train_all_models_v7.py` / `evaluate_all_models_v7.py` importing v7).
"""

from .abr_multi_env_v6 import ABREnv as _ABREnvV6


class ABREnv(_ABREnvV6):
    """V7 environment alias (currently identical to V6)."""


__all__ = ["ABREnv"]

