import numpy as np

def compute_composite_reward(info, last_bitrate=None,
                             alpha=0.7, beta=0.4, gamma=3.5, delta=0.6):
    """
    Composite QoE Reward Function
    Combines perceptual quality (VMAF), bitrate, rebuffering, and smoothness.

    reward = α * VMAF_norm + β * bitrate(Mbps) - γ * rebuffer(s) - δ * |Δbitrate|

    Parameters
    ----------
    info : dict
        Environment info dict containing keys:
        - 'vmaf': perceptual quality score [0–100]
        - 'bitrate': current segment bitrate (kbps)
        - 'rebuffer_time': rebuffer duration (seconds)
    last_bitrate : float, optional
        Previous segment bitrate (kbps)
    alpha, beta, gamma, delta : float
        Weights for VMAF quality, bitrate, rebuffer penalty, and smoothness penalty.

    Returns
    -------
    float
        Clipped reward value in range [-10, +10]
    """

    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)

    # Normalize bitrate to Mbps
    bitrate_mbps = bitrate / 1000.0

    # Normalize VMAF to 0–1 range
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    # Smoothness penalty (change in bitrate between segments)
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    # Weighted composite reward
    reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps -
        gamma * rebuffer -
        delta * smooth_penalty
    )

    # Safety: limit to a reasonable range to avoid unstable gradients
    reward = float(np.clip(reward, -10.0, 10.0))

    return reward
