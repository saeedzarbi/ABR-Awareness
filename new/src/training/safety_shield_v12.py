"""
Safety shield (V12): deterministic action projection with stable defaults.

Design goals:
- Keep safety strong and predictable (low rebuffer).
- Avoid over-triggering and excessive switching.
- Be reproducible and easy to report in a paper.

V12 adopts a risk-gated projection based on download-time vs buffer, which is
more directly aligned with stall risk than (tp, buffer, action) thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class ShieldConfig:
    level: str = "light"  # 'off' | 'light' | 'strong'
    catastrophic_ratio: float = 2.0
    safe_margin_light: float = 0.5
    safe_margin_strong: float = 1.5
    safety_tp_scale: float = 0.90

    # Risk gate (download-time vs buffer)
    only_when_risky: bool = False
    risky_dl_over_buf_ratio: float = 1.10


def safe_adjust_action(env, action: int, cfg: ShieldConfig) -> tuple[int, int]:
    """
    Returns (safe_action, intervened_flag).
    """
    try:
        if cfg.level == "off":
            return int(action), 0

        buf = float(getattr(env, "buffer_level", 0.0))
        cur_idx = int(action)
        cur_idx = max(0, min(cur_idx, len(env.BITRATE_LEVELS) - 1))

        if buf <= 0.3:
            return 0, int(cur_idx != 0)

        trace_tp = getattr(env, "current_trace", None)
        if trace_tp and "throughput_kbps" in trace_tp:
            tp_idx = int(env.chunk_idx * env.CHUNK_DURATION) % len(trace_tp["throughput_kbps"])
            trace_tp_val = float(trace_tp["throughput_kbps"][tp_idx])
        else:
            trace_tp_val = 2000.0

        last_tp = getattr(env, "last_raw_throughput", 2000.0)
        tp_est = min(trace_tp_val, last_tp) * cfg.safety_tp_scale
        tp_est = max(tp_est, env.MIN_NETWORK_THROUGHPUT)

        def dl_time_for(idx: int) -> float:
            br = int(env.BITRATE_LEVELS[idx])
            chunk_bits = env.get_chunk_size_bits(br, env.chunk_idx)
            return min(chunk_bits / (tp_est * 1000.0 + 1e-6), 60.0)

        if cfg.only_when_risky:
            dt_req = dl_time_for(cur_idx)
            if dt_req <= max(buf, 0.1) * cfg.risky_dl_over_buf_ratio:
                return cur_idx, 0

        if cfg.level == "light":
            dt = dl_time_for(cur_idx)
            if dt > buf * cfg.catastrophic_ratio:
                original = cur_idx
                cur_idx = max(0, cur_idx - 1)
                dt = dl_time_for(cur_idx)
                if dt > buf * cfg.catastrophic_ratio:
                    for fallback in range(cur_idx, -1, -1):
                        if dl_time_for(fallback) <= buf - cfg.safe_margin_light:
                            return fallback, 1
                    return 0, 1
                return cur_idx, int(cur_idx != original)
            return cur_idx, 0

        original = cur_idx
        margin = cfg.safe_margin_strong
        for _ in range(cur_idx):
            if dl_time_for(cur_idx) <= buf - margin:
                break
            cur_idx -= 1
        return cur_idx, int(cur_idx != original)
    except Exception:
        return int(action), 0


class SafetyShieldWrapper(gym.Wrapper):
    """Wraps an ABR env and projects the action before stepping."""

    def __init__(self, env: gym.Env, cfg: ShieldConfig):
        super().__init__(env)
        self.cfg = cfg
        self.interventions = 0
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.interventions = 0
        self.steps = 0
        info = dict(info)
        info["shield_intervention_rate"] = 0.0
        return obs, info

    def step(self, action):
        safe_action, intervened = safe_adjust_action(self.env, action, self.cfg)
        self.interventions += int(intervened)
        self.steps += 1

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        info = dict(info)
        info["shielded_action"] = int(safe_action)
        info["shield_intervened"] = int(intervened)
        info["shield_intervention_rate"] = float(self.interventions) / max(float(self.steps), 1.0)
        return obs, reward, terminated, truncated, info


__all__ = ["ShieldConfig", "SafetyShieldWrapper", "safe_adjust_action"]

