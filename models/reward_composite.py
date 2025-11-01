"""
Composite QoE Reward Function
-----------------------------
Combines bitrate, rebuffering, smoothness, and perceptual quality (VMAF)
into a single scalar reward.

Used in: train_composite.py
"""

import numpy as np


def compute_composite_reward(info, last_bitrate=None, alpha=0.7, beta=0.3, gamma=5.0, delta=0.7):
    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)

    bitrate_mbps = bitrate / 1000.0
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps -
        gamma * rebuffer -
        delta * smooth_penalty
    )

    return float(np.clip(reward, -10.0, 10.0))
