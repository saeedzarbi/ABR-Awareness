"""
Low-latency + per-chunk non-monotone VMAF environment, V17.

This is the ONLY configuration under which the VMAF-aware shield can plausibly
beat the index shield with a measurable, non-zero margin, because it combines the
two conditions that all previous rounds showed are BOTH necessary:

  * a small buffer (low-latency), so the shield actually intervenes
    (v14/v16 broadband: intervention rate ~0% -> shield inert), and
  * a non-monotone per-chunk (multi-resolution) VMAF ladder, so the ranking rule
    can diverge from highest-index selection
    (v15 low-latency but monotone ladder: vmaf_aware == index, gain exactly 0).

V17 = V16 (per-chunk multi-resolution VMAF ladder) + V15 operating point
(small buffer, robust trace loader). Buffer/rebuffering dynamics stay byte-driven,
so only the VMAF mapping and buffer geometry change relative to the baselines.

Prerequisite: data/vmaf_scores/vmaf_perchunk_multires.csv (build_multires_vmaf.py).
"""

from __future__ import annotations

from .abr_multi_env_v15 import ABREnv as _ABREnvV15
from .abr_multi_env_v16 import ABREnv as _ABREnvV16


class ABREnv(_ABREnvV16):
    """Per-chunk multi-resolution VMAF ladder at the low-latency operating point."""

    # Low-latency operating point (identical to V15).
    BUFFER_MAX = 6.0
    BUFFER_TARGET = 3.0
    B_REF = 2.0

    # Reuse V15's robust trace loader (skips sidecar JSON like trace_stats.json).
    _load_traces = _ABREnvV15._load_traces


__all__ = ["ABREnv"]
