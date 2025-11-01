import numpy as np

def compute_composite_reward(info, last_bitrate=None,
                             alpha=1.0, beta=0.5, gamma=1.8, delta=0.4, buffer_bonus=0.3):
    """
    ✅ Balanced Composite QoE Reward (normalized and perceptually weighted)

    Combines perceptual (VMAF), bitrate, and rebuffer/smoothness penalties with
    mild buffer stability bonus. Designed for stable PPO convergence and realistic QoE.

    Parameters
    ----------
    info : dict
        Step information from the environment containing:
        - bitrate (kbps)
        - rebuffer_time (s)
        - vmaf (0–100)
        - buffer (optional, s)
    last_bitrate : float, optional
        Bitrate from previous chunk for smoothness penalty.
    alpha : float
        Weight for perceptual quality (VMAF).
    beta : float
        Weight for bitrate contribution.
    gamma : float
        Weight for rebuffering penalty (logarithmic).
    delta : float
        Weight for bitrate switching penalty.
    buffer_bonus : float
        Positive bonus factor for maintaining stable buffer.

    Returns
    -------
    float : final reward (scaled to range approximately [-10, +10])
    """

    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)
    buffer = info.get("buffer", 0.0)

    # Normalize features
    bitrate_mbps = bitrate / 1000.0
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    # Log-scaled rebuffer penalty (less aggressive)
    rebuffer_penalty = np.log1p(rebuffer) * gamma

    # Smoothness penalty
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    # Small buffer stability bonus
    buffer_reward = buffer_bonus * np.tanh(buffer / 10.0)

    # Weighted reward
    raw_reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps +
        buffer_reward -
        rebuffer_penalty -
        delta * smooth_penalty
    )

    # Normalize to stable PPO range
    reward = np.tanh(raw_reward / 5.0) * 10.0

    return float(np.clip(reward, -10.0, 10.0))
