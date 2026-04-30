"""
V11 training wrappers to improve QoE while keeping safety.

- ShieldAwarePenaltyWrapper:
    Penalize shield interventions (and disagreement with shielded action)
    so the learned policy becomes intrinsically safer/less oscillatory.

- HysteresisActionWrapper:
    Method-level hysteresis to reduce bitrate switching without relying on a
    system-wide guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


class ShieldAwarePenaltyWrapper(gym.Wrapper):
    """
    Reward shaping:
      r' = r - beta * 1[shield_intervened] - gamma * |a_raw - a_applied|

    Expects info keys provided by `SafetyShieldWrapper`:
      - shield_intervened
      - shielded_action
    """

    def __init__(self, env: gym.Env, beta: float = 0.25, gamma: float = 0.05):
        super().__init__(env)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        intervened = int(info.get("shield_intervened", 0))
        applied = int(info.get("shielded_action", action))
        raw = int(action)
        shaped = float(reward) - self.beta * float(intervened) - self.gamma * float(abs(raw - applied))
        return obs, shaped, terminated, truncated, info


@dataclass(frozen=True)
class HysteresisConfig:
    max_up_step: int = 1
    max_down_step: int = 2
    min_buffer_for_up: float = 3.0  # seconds


class HysteresisActionWrapper(gym.Wrapper):
    """
    Modifies the input action before stepping (reduces switching).
    """

    def __init__(self, env: gym.Env, cfg: HysteresisConfig):
        super().__init__(env)
        self.cfg = cfg
        self._last_action: int = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_action = 0
        return obs, info

    def step(self, action):
        a = int(action)
        a = max(0, min(a, len(self.env.BITRATE_LEVELS) - 1))

        buf = float(getattr(self.env, "buffer_level", 0.0))

        if a > self._last_action:
            a = min(a, self._last_action + int(self.cfg.max_up_step))
            if buf < float(self.cfg.min_buffer_for_up):
                a = self._last_action
        elif a < self._last_action:
            a = max(a, self._last_action - int(self.cfg.max_down_step))

        self._last_action = int(a)
        return self.env.step(a)


__all__ = ["ShieldAwarePenaltyWrapper", "HysteresisConfig", "HysteresisActionWrapper"]

