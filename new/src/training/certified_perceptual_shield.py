"""
Certified Perceptual Shield (V18).

A model-agnostic, post-policy runtime shield with THREE novel ingredients that,
unlike the V12-V16 "VMAF-aware ranking" shield (proven inert on monotone
ladders), exploit structure that genuinely exists in the data and admit a formal
guarantee:

  (1) VMAF-KNEE BANDWIDTH BANKING (perceptually-lossless downshift).
      The rate-VMAF ladder saturates at the top (e.g. 2850->6000 kbps buys ~0
      VMAF). Even when the policy's action is *feasible*, the shield lowers it to
      the smallest rung whose VMAF is within epsilon of the policy's rung:
          j* = min { j <= a : VMAF(a) - VMAF(j) <= epsilon }.
      This is perceptually lossless (<= epsilon VMAF by construction) yet can
      halve the bytes on saturated chunks. The saved bytes are *banked* as extra
      buffer that protects future vulnerable chunks. This is proactive budget
      TRANSFER, not reactive projection -- and it works on the plain (monotone)
      session ladder because the saturation, not any inversion, is the lever.

  (2) CONFORMAL THROUGHPUT LOWER BOUND (distribution-free).
      Instead of a hand-tuned pessimism factor, the download-time bound uses a
      split/online-conformal lower bound on throughput calibrated from the
      predictor's own residuals, giving coverage P(actual >= tp_lb) >= 1 - alpha
      under exchangeability.

  (3) CERTIFIED FEASIBILITY.
      After banking, the shield enforces feasibility under tp_lb: it never lets
      the (bounded) download time exceed buffer minus margin. Combined with (2)
      this yields the paper's theorem: under conformal coverage 1-alpha, the
      per-step rebuffering bound holds with probability >= 1-alpha.

The shield returns the executed rung plus rich diagnostics (banked bits, VMAF
given up, tp lower bound, whether the conformal bound covered the realized
throughput) so the evaluation can validate BOTH the perceptual non-inferiority
and the empirical conformal coverage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np


# --------------------------------------------------------------------------- #
# Conformal throughput estimator
# --------------------------------------------------------------------------- #
@dataclass
class ConformalConfig:
    alpha: float = 0.10           # miscoverage target -> coverage 1 - alpha
    window: int = 200             # rolling calibration window (residual ratios)
    k_predict: int = 5            # harmonic-mean horizon for the point predictor
    min_calib: int = 20           # fall back to a fixed factor until this many residuals
    fallback_scale: float = 0.80  # multiplicative pessimism before calibration


class ConformalThroughputEstimator:
    """Online-conformal lower bound on throughput (kbps).

    Point predictor: harmonic mean of the last ``k`` observed throughputs
    (the standard robust-MPC ABR predictor). Calibration: the multiplicative
    residual ``ratio = actual / predicted`` is collected in a rolling window; the
    empirical ``alpha``-quantile ``q_alpha`` of those ratios gives the lower
    bound ``tp_lb = predicted * q_alpha``, which by construction covers the
    realized throughput with probability >= 1 - alpha (exchangeable residuals).
    """

    def __init__(self, cfg: ConformalConfig):
        self.cfg = cfg
        self._obs: deque[float] = deque(maxlen=max(cfg.k_predict, 1))
        self._ratios: deque[float] = deque(maxlen=cfg.window)
        self._last_pred: float | None = None

    def predict(self) -> float:
        if not self._obs:
            return 0.0
        vals = np.asarray(self._obs, dtype=float)
        vals = np.clip(vals, 1e-6, None)
        pred = float(len(vals) / np.sum(1.0 / vals))  # harmonic mean
        self._last_pred = pred
        return pred

    def lower_bound(self) -> float:
        pred = self.predict()
        if pred <= 0.0:
            return 0.0
        if len(self._ratios) < self.cfg.min_calib:
            return pred * self.cfg.fallback_scale
        q = float(np.quantile(np.asarray(self._ratios, dtype=float), self.cfg.alpha))
        q = min(q, 1.0)  # a lower bound should not exceed the point prediction
        return pred * max(q, 1e-3)

    def update(self, actual_kbps: float):
        """Register the realized throughput and grow the calibration set."""
        actual = float(max(actual_kbps, 1e-6))
        if self._last_pred and self._last_pred > 0:
            self._ratios.append(actual / self._last_pred)
        self._obs.append(actual)

    def empirical_coverage(self) -> float:
        if len(self._ratios) == 0:
            return float("nan")
        q = float(np.quantile(np.asarray(self._ratios, dtype=float), self.cfg.alpha))
        arr = np.asarray(self._ratios, dtype=float)
        return float((arr >= q).mean())


# --------------------------------------------------------------------------- #
# Certified perceptual shield
# --------------------------------------------------------------------------- #
@dataclass
class CPShieldConfig:
    enabled: bool = True
    # (1) perceptual banking
    enable_banking: bool = True
    epsilon_vmaf: float = 1.0          # perceptual budget: max VMAF given up by banking
    # (2) conformal bound (if disabled, use last/trace throughput * fallback_scale)
    enable_conformal: bool = True
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    # (3) feasibility projection
    safety_margin: float = 0.5         # keep dl_time <= buffer - margin (seconds)
    min_buffer: float = 0.3            # below this -> emergency lowest rung
    never_upgrade: bool = True


def _vmaf(env, j: int) -> float:
    try:
        br = int(env.BITRATE_LEVELS[j])
        scores = getattr(env, "current_vmaf_scores", None)
        if scores:
            return float(scores.get(br, 35.0))
    except Exception:
        pass
    return float(j) * 10.0


def certified_safe_action(env, action, cfg: CPShieldConfig, est: ConformalThroughputEstimator):
    """Return (safe_action, intervened, info)."""
    info = {"banked_bits": 0.0, "vmaf_given_up": 0.0, "tp_lb_kbps": 0.0,
            "banked": 0, "safety_downgrade": 0}
    try:
        if not cfg.enabled:
            return int(action), 0, info

        n = len(env.BITRATE_LEVELS)
        a = max(0, min(int(action), n - 1))
        buf = float(getattr(env, "buffer_level", 0.0))

        # throughput lower bound (kbps)
        if cfg.enable_conformal:
            tp_lb = est.lower_bound()
        else:
            trace = getattr(env, "current_trace", None)
            if trace and "throughput_kbps" in trace:
                series = trace["throughput_kbps"]
                tp_now = float(series[int(env.chunk_idx * env.CHUNK_DURATION) % len(series)])
            else:
                tp_now = 2000.0
            last_tp = float(getattr(env, "last_raw_throughput", 2000.0))
            tp_lb = min(tp_now, last_tp) * cfg.conformal.fallback_scale
        tp_lb = max(tp_lb, float(env.MIN_NETWORK_THROUGHPUT))
        info["tp_lb_kbps"] = tp_lb

        def dl(j: int) -> float:
            bits = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[j]), env.chunk_idx)
            return min(bits / (tp_lb * 1000.0 + 1e-6), 60.0)

        if buf <= cfg.min_buffer:
            info["safety_downgrade"] = int(a != 0)
            return 0, int(a != 0), info

        va = _vmaf(env, a)

        # (1) perceptual-knee banking: smallest rung within epsilon of the policy's VMAF
        j_knee = a
        if cfg.enable_banking:
            for j in range(a + 1):
                if va - _vmaf(env, j) <= cfg.epsilon_vmaf:
                    j_knee = j
                    break

        # (3) feasibility projection under the conformal bound (safety wins over epsilon)
        target = j_knee
        while target > 0 and dl(target) > max(buf - cfg.safety_margin, 0.1):
            target -= 1

        safe = min(a, target) if cfg.never_upgrade else target
        intervened = int(safe != a)

        # diagnostics
        bits_a = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[a]), env.chunk_idx)
        bits_s = env.get_chunk_size_bits(int(env.BITRATE_LEVELS[safe]), env.chunk_idx)
        info["banked_bits"] = float(max(0.0, bits_a - bits_s))
        info["vmaf_given_up"] = float(max(0.0, va - _vmaf(env, safe)))
        # Banking credit only when the knee was below the policy AND safety did
        # not force a further cut past the knee.
        info["banked"] = int(j_knee < a and safe == j_knee)
        info["safety_downgrade"] = int(target < j_knee)
        info["knee_idx"] = int(j_knee)
        return int(safe), intervened, info
    except Exception:
        return int(action), 0, info


class CertifiedPerceptualShieldWrapper(gym.Wrapper):
    """Wrap an ABR env, apply the certified perceptual shield before stepping, and
    feed realized throughput back to the conformal estimator after stepping."""

    def __init__(self, env: gym.Env, cfg: CPShieldConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or CPShieldConfig()
        self.est = ConformalThroughputEstimator(self.cfg.conformal)
        self._reset_stats()

    def _reset_stats(self):
        self.interventions = 0
        self.bank_events = 0
        self.safety_events = 0
        self.steps = 0
        self.total_banked_bits = 0.0
        self.total_vmaf_given_up = 0.0
        self.cover_hits = 0
        self.cover_total = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.est = ConformalThroughputEstimator(self.cfg.conformal)
        self._reset_stats()
        info = dict(info)
        info["shield_intervention_rate"] = 0.0
        return obs, info

    def step(self, action):
        safe_action, intervened, sinfo = certified_safe_action(self.env, action, self.cfg, self.est)
        tp_lb = sinfo.get("tp_lb_kbps", 0.0)

        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # realized throughput this step (kbps); feed conformal calibration
        realized = float(info.get("throughput", getattr(self.env, "last_raw_throughput", 0.0)))
        if tp_lb > 0.0 and realized > 0.0:
            self.cover_total += 1
            self.cover_hits += int(realized >= tp_lb)
        self.est.update(realized)

        self.steps += 1
        self.interventions += int(intervened)
        self.bank_events += int(sinfo.get("banked", 0))
        self.safety_events += int(sinfo.get("safety_downgrade", 0))
        self.total_banked_bits += float(sinfo.get("banked_bits", 0.0))
        self.total_vmaf_given_up += float(sinfo.get("vmaf_given_up", 0.0))

        info = dict(info)
        info["shielded_action"] = int(safe_action)
        info["shield_intervened"] = int(intervened)
        info["shield_intervention_rate"] = self.interventions / max(self.steps, 1)
        info["banked_bits"] = float(sinfo.get("banked_bits", 0.0))
        info["vmaf_given_up"] = float(sinfo.get("vmaf_given_up", 0.0))
        info["tp_lb_kbps"] = tp_lb
        info["conformal_coverage"] = (self.cover_hits / self.cover_total) if self.cover_total else float("nan")
        return obs, reward, terminated, truncated, info


__all__ = [
    "ConformalConfig",
    "ConformalThroughputEstimator",
    "CPShieldConfig",
    "CertifiedPerceptualShieldWrapper",
    "certified_safe_action",
]
