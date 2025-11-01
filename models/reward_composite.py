import numpy as np

def compute_composite_reward(info, last_bitrate=None,
                             alpha=0.7, beta=0.4, gamma=3.0, delta=0.6, scale=10.0):
    """
    Balanced Composite QoE Reward (scaled)
    Includes proportional normalization to keep reward range around [-1, +1].
    """
    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)

    # Normalize features
    bitrate_mbps = bitrate / 1000.0
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    # Log-scaled rebuffer penalty
    rebuffer_penalty = np.log1p(rebuffer) * gamma

    # Smoothness penalty
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    # Weighted composite reward
    raw_reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps -
        rebuffer_penalty -
        delta * smooth_penalty
    )

    # ⚙️ Normalize reward by scale factor to keep magnitude around ±1
    reward = np.tanh(raw_reward / scale) * scale

    return float(np.clip(reward, -10.0, 10.0))
