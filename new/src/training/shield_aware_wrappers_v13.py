"""
Shield-aware training wrappers (V13).

Defaults are tuned for QoE recovery: avoid the V12 max-step hysteresis on the
main path, but keep optional mild penalties so the policy learns to avoid
actions that the guard must repair.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


class ShieldAwarePenaltyWrapperV13(gym.Wrapper):
    """
    Penalize guard disagreement without making the policy overly conservative.
    """

    def __init__(
        self,
        env: gym.Env,
        beta_intervene: float = 0.05,
        gamma_deviation: float = 0.06,
    ):
        super().__init__(env)
        self.beta = float(beta_intervene)
        self.gamma = float(gamma_deviation)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        intervened = int(info.get("shield_intervened", 0))
        applied = int(info.get("shielded_action", info.get("applied_action", action)))
        agent_a = int(action)
        deviation = abs(applied - agent_a)
        shaped = float(reward) - self.beta * float(intervened) - self.gamma * float(deviation)

        info["reward_raw"] = float(reward)
        info["reward_shaped"] = float(shaped)
        info["shield_deviation"] = int(deviation)
        return obs, shaped, terminated, truncated, info


@dataclass(frozen=True)
class HysteresisConfigV13:
    max_step: int = 2
    min_buf_for_upswitch: float = 1.0


class HysteresisActionWrapperV13(gym.Wrapper):
    """
    Optional, softer hysteresis. Not used by the default v13 QoE path.
    """

    def __init__(self, env: gym.Env, cfg: HysteresisConfigV13):
        super().__init__(env)
        self.cfg = cfg
        self._last_action = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_action = 0
        return obs, info

    def step(self, action):
        a = int(action)
        last = int(self._last_action)
        max_step = max(int(self.cfg.max_step), 1)

        if a > last + max_step:
            a = last + max_step
        elif a < last - max_step:
            a = last - max_step

        buf = float(getattr(self.env, "buffer_level", 0.0))
        if a > last and buf < float(self.cfg.min_buf_for_upswitch):
            a = last

        self._last_action = a
        obs, reward, terminated, truncated, info = self.env.step(a)
        info = dict(info)
        info["applied_action"] = int(a)
        return obs, reward, terminated, truncated, info


__all__ = ["ShieldAwarePenaltyWrapperV13", "HysteresisActionWrapperV13", "HysteresisConfigV13"]
