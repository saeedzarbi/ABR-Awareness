import numpy as np

def compute_composite_reward(info, last_bitrate=None,
                             alpha=0.7, beta=0.4, gamma=3.5, delta=0.6):
    """
    Composite QoE Reward (log-scaled rebuffer penalty)
    """
    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)

    # Normalize bitrate and VMAF
    bitrate_mbps = bitrate / 1000.0
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    # Log-scaled rebuffer penalty (less harsh for small stalls)
    rebuffer_penalty = np.log1p(rebuffer) * gamma  # log(1+x) softens effect

    # Smoothness penalty
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    # Weighted composite reward
    reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps -
        rebuffer_penalty -
        delta * smooth_penalty
    )

    return float(np.clip(reward, -10.0, 10.0))
