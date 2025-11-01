"""
Composite QoE Reward Function
-----------------------------
Combines bitrate, rebuffering, smoothness, and perceptual quality (VMAF)
into a single scalar reward.

Used in: train_composite.py
"""

import numpy as np

def compute_composite_reward(info, last_bitrate=None, alpha=0.7, beta=0.3, gamma=4.0, delta=0.5):
    """
    Compute the composite QoE-based reward.
    
    Parameters
    ----------
    info : dict
        Environment info after taking an action. Should include:
        - 'bitrate': bitrate used (in kbps)
        - 'rebuffer_time': stall duration (in seconds)
        - 'vmaf': predicted or measured VMAF score (0–100)
    last_bitrate : float, optional
        Bitrate from previous step (kbps) for smoothness penalty.
    alpha, beta, gamma, delta : float
        Weights for each component:
        α: VMAF contribution
        β: bitrate contribution
        γ: rebuffer penalty
        δ: smoothness penalty
    
    Returns
    -------
    float
        The total reward value.
    """
    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)

    # Normalize values
    bitrate_mbps = bitrate / 1000.0
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    # Smoothness penalty
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    # Weighted sum
    reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps -
        gamma * rebuffer -
        delta * smooth_penalty
    )

    # Clip to stable range
    return float(np.clip(reward, -10.0, 10.0))
