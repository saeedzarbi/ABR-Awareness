import numpy as np

def compute_composite_reward(info, last_bitrate=None,
                             alpha=1.2,   # ↑ تأکید بیشتر روی کیفیت VMAF
                             beta=0.5,    # وزن bitrate
                             gamma=4.0,   # ↑ افزایش جریمه‌ی rebuffer
                             delta=0.6,   # ↑ کاهش نوسانات (switch penalty)
                             buffer_bonus=0.3):
    """
    ✅ Composite QoE Reward (Balanced for Quality, Stability, and Smoothness)

    ترکیب شده از:
    - کیفیت ادراکی (VMAF)
    - bitrate
    - rebuffer penalty (log-scale)
    - smoothness penalty
    - buffer stability bonus
    """

    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)
    buffer = info.get("buffer", 0.0)

    # Normalize features
    bitrate_mbps = bitrate / 1000.0
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    # Log-scaled rebuffer penalty
    rebuffer_penalty = np.log1p(rebuffer) * gamma

    # Smoothness penalty (switching between bitrates)
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    # Buffer stability bonus
    buffer_reward = buffer_bonus * np.tanh(buffer / 10.0)

    # Weighted reward combination
    raw_reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps +
        buffer_reward -
        rebuffer_penalty -
        delta * smooth_penalty
    )

    # Normalize and clip to stable PPO range
    reward = np.tanh(raw_reward / 8.0) * 10.0
    return float(np.clip(reward, -10.0, 10.0))
