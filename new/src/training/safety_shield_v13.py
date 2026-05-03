"""
Safety shield (V13): QoE-oriented deterministic action projection.

V13 is designed to recover QoE lost by the V12 shield while keeping stall risk
bounded. Compared with V12, defaults are less pessimistic and the projection
can allow a small predicted stall budget before downgrading.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class ShieldConfigV13:
    level: str = "qoe"  # 'off' | 'qoe' | 'light' | 'strong'
    catastrophic_ratio: float = 2.50
    safe_margin_light: float = 0.15
    safe_margin_strong: float = 0.75
    safety_tp_scale: float = 0.97
    critical_buffer_s: float = 0.20
    min_guard_action: int = 1

    # Risk gate: leave the raw action untouched unless estimated download time
    # meaningfully exceeds buffered slack.
    only_when_risky: bool = True
    risky_dl_over_buf_ratio: float = 1.35

    # QoE mode accepts tiny predicted stalls instead of forcing a low bitrate.
    max_predicted_stall_s: float = 0.25
    max_downgrade_steps: int = 2

    # Avoid bitrate ping-pong after a guard intervention.
    smooth_recovery: bool = True
    recovery_window: int = 3
    max_recovery_upshift: int = 1
    recovery_buffer_s: float = 6.0


def _trace_throughput_kbps(env) -> float:
    trace_tp = getattr(env, "current_trace", None)
    if trace_tp and "throughput_kbps" in trace_tp:
        tp_idx = int(env.chunk_idx * env.CHUNK_DURATION) % len(trace_tp["throughput_kbps"])
        return float(trace_tp["throughput_kbps"][tp_idx])
    return 2000.0


def safe_adjust_action_v13(env, action: int, cfg: ShieldConfigV13) -> tuple[int, int, dict]:
    """
    Returns (safe_action, intervened_flag, diagnostics).
    """
    try:
        raw_idx = int(action)
        raw_idx = max(0, min(raw_idx, len(env.BITRATE_LEVELS) - 1))
        if cfg.level == "off":
            return raw_idx, 0, {"shield_reason": "off"}

        buf = max(float(getattr(env, "buffer_level", 0.0)), 0.0)
        if buf <= cfg.critical_buffer_s:
            return 0, int(raw_idx != 0), {"shield_reason": "critical_buffer", "shield_est_dl": None}

        min_guard_action = max(0, min(int(cfg.min_guard_action), len(env.BITRATE_LEVELS) - 1))

        last_tp = float(getattr(env, "last_raw_throughput", 2000.0))
        tp_est = min(_trace_throughput_kbps(env), last_tp) * float(cfg.safety_tp_scale)
        tp_est = max(tp_est, float(env.MIN_NETWORK_THROUGHPUT))

        def dl_time_for(idx: int) -> float:
            br = int(env.BITRATE_LEVELS[idx])
            chunk_bits = env.get_chunk_size_bits(br, env.chunk_idx)
            return min(chunk_bits / (tp_est * 1000.0 + 1e-6), 60.0)

        raw_dl = dl_time_for(raw_idx)
        diag = {
            "shield_reason": "pass",
            "shield_est_dl": float(raw_dl),
            "shield_tp_est": float(tp_est),
            "shield_buffer": float(buf),
        }

        if cfg.only_when_risky and raw_dl <= max(buf, 0.1) * cfg.risky_dl_over_buf_ratio:
            return raw_idx, 0, diag

        if cfg.level == "qoe":
            max_stall = max(float(cfg.max_predicted_stall_s), 0.0)
            floor_idx = max(min_guard_action, raw_idx - max(int(cfg.max_downgrade_steps), 0))
            for candidate in range(raw_idx, floor_idx - 1, -1):
                if dl_time_for(candidate) <= buf + max_stall:
                    diag["shield_reason"] = "qoe_budget"
                    return candidate, int(candidate != raw_idx), diag

            # If a bounded downgrade cannot satisfy the small-stall budget, fall
            # back to the highest representation that does.
            for candidate in range(floor_idx - 1, min_guard_action - 1, -1):
                if dl_time_for(candidate) <= buf + max_stall:
                    diag["shield_reason"] = "qoe_fallback"
                    return candidate, int(candidate != raw_idx), diag
            diag["shield_reason"] = "qoe_safest"
            return min_guard_action, int(raw_idx != min_guard_action), diag

        if cfg.level == "light":
            if raw_dl <= buf * cfg.catastrophic_ratio:
                return raw_idx, 0, diag
            for candidate in range(raw_idx - 1, min_guard_action - 1, -1):
                if dl_time_for(candidate) <= max(buf - cfg.safe_margin_light, 0.0):
                    diag["shield_reason"] = "light_project"
                    return candidate, 1, diag
            diag["shield_reason"] = "light_safest"
            return min_guard_action, int(raw_idx != min_guard_action), diag

        original = raw_idx
        cur_idx = raw_idx
        while cur_idx > min_guard_action and dl_time_for(cur_idx) > max(buf - cfg.safe_margin_strong, 0.0):
            cur_idx -= 1
        diag["shield_reason"] = "strong_project"
        return cur_idx, int(cur_idx != original), diag
    except Exception:
        return int(action), 0, {"shield_reason": "error"}


class SafetyShieldWrapperV13(gym.Wrapper):
    """Wraps an ABR env and projects the action before stepping."""

    def __init__(self, env: gym.Env, cfg: ShieldConfigV13):
        super().__init__(env)
        self.cfg = cfg
        self.interventions = 0
        self.steps = 0
        self.last_safe_action = 0
        self.recovery_steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.interventions = 0
        self.steps = 0
        self.last_safe_action = 0
        self.recovery_steps = 0
        info = dict(info)
        info["shield_intervention_rate"] = 0.0
        return obs, info

    def step(self, action):
        safe_action, intervened, diag = safe_adjust_action_v13(self.env, action, self.cfg)

        if self.cfg.smooth_recovery and self.steps > 0:
            buf = float(getattr(self.env, "buffer_level", 0.0))
            max_up = max(int(self.cfg.max_recovery_upshift), 1)
            recovery_cap = min(len(self.env.BITRATE_LEVELS) - 1, int(self.last_safe_action) + max_up)
            in_recovery = self.recovery_steps > 0 and buf < float(self.cfg.recovery_buffer_s)
            if in_recovery and int(safe_action) > recovery_cap:
                safe_action = recovery_cap
                intervened = int(int(safe_action) != int(action))
                diag["shield_reason"] = "smooth_recovery"

        if intervened:
            self.recovery_steps = max(int(self.cfg.recovery_window), 0)
        elif self.recovery_steps > 0:
            self.recovery_steps -= 1

        self.interventions += int(intervened)
        self.steps += 1
        self.last_safe_action = int(safe_action)

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        info = dict(info)
        info.update(diag)
        info["raw_action"] = int(action)
        info["shielded_action"] = int(safe_action)
        info["shield_intervened"] = int(intervened)
        info["shield_intervention_rate"] = float(self.interventions) / max(float(self.steps), 1.0)
        return obs, reward, terminated, truncated, info


__all__ = ["ShieldConfigV13", "SafetyShieldWrapperV13", "safe_adjust_action_v13"]
