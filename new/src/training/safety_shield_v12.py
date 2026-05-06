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

    # ---- VMAF-Aware Soft Projection (V12.1) ----
    # When True the shield uses a per-video VMAF-aware fallback:
    #  (1) soft buffer tolerance instead of hard "buf - margin" threshold,
    #  (2) VMAF-loss budget that prevents dropping bitrate too far when
    #      the per-video VMAF curve is concave (e.g. crowd_run).
    vmaf_aware: bool = False
    # Soft tolerance: candidate j is acceptable if dl_time(j) <= buf * soft_tolerance.
    # 1.0 = stay within current buffer; 1.2 = allow mild risk to keep VMAF.
    soft_tolerance: float = 1.0
    # Maximum VMAF the shield is allowed to give up vs the policy's raw choice.
    # If no soft-safe candidate keeps VMAF within this budget, we pick the
    # candidate with smallest VMAF loss subject to soft safety.
    vmaf_loss_budget: float = 8.0


def _vmaf_for_idx(env, idx: int) -> float:
    """Per-video VMAF lookup; falls back to a monotone proxy if unavailable."""
    try:
        br = int(env.BITRATE_LEVELS[idx])
        scores = getattr(env, "current_vmaf_scores", None)
        if scores:
            return float(scores.get(br, 35.0))
    except Exception:
        pass
    # Monotone fallback: index-proportional score so logic still makes sense.
    return float(idx) * 10.0


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

        # ============================================================
        # VMAF-Aware Soft Projection (V12.1)
        # ------------------------------------------------------------
        # Replaces the linear "step-down with hard margin" fallback by:
        #   1) soft tolerance: dl_time(j) <= buf * soft_tolerance
        #      (instead of dl_time(j) <= buf - margin)
        #   2) VMAF-loss budget: among candidates that pass (1), prefer
        #      those whose per-video VMAF stays within `vmaf_loss_budget`
        #      of the policy's raw choice. If none qualify, pick the one
        #      with the smallest VMAF loss subject to soft safety.
        # This addresses concave per-video VMAF curves where stepping
        # down can be very cheap (sintel: 1850->1200 = -0.12 VMAF) or
        # very expensive (crowd_run: 1850->1200 = -8.09 VMAF).
        # ============================================================
        if cfg.vmaf_aware:
            cur_dt = dl_time_for(cur_idx)
            if cur_dt <= buf * cfg.catastrophic_ratio:
                # Already within catastrophic ratio: no intervention.
                return cur_idx, 0

            cur_vmaf = _vmaf_for_idx(env, cur_idx)
            soft_thresh = buf * cfg.soft_tolerance

            # Build candidates over indices [0, cur_idx]; never upgrade.
            soft_safe = []  # (idx, dt, vmaf, vmaf_loss)
            for j in range(cur_idx + 1):
                dt_j = dl_time_for(j)
                if dt_j <= soft_thresh:
                    vmaf_j = _vmaf_for_idx(env, j)
                    soft_safe.append((j, dt_j, vmaf_j, max(0.0, cur_vmaf - vmaf_j)))

            if soft_safe:
                # Prefer candidates within VMAF-loss budget; among those,
                # pick the one with the highest index (== highest bitrate that
                # is still soft-safe, == best VMAF in monotone ladders).
                in_budget = [c for c in soft_safe if c[3] <= cfg.vmaf_loss_budget]
                pool = in_budget if in_budget else soft_safe
                # Tie-break: highest VMAF, then highest index, then smallest dt.
                best = max(pool, key=lambda x: (x[2], x[0], -x[1]))
                return best[0], int(best[0] != cur_idx)

            # No soft-safe option in [0, cur_idx]: emergency fallback.
            return 0, int(cur_idx != 0)

        # ---- Legacy V12 logic (unchanged) ----
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

