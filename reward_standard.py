def compute_standard_reward(info, last_bitrate):
    """
    Reward = bitrate_kbps - 4.3 * rebuffer - 1.0 * smoothness
    (scaled to roughly match the VMAF reward scale)
    """
    rebuffer = float(info.get("rebuffer_time", 0.0))
    bitrate = float(info.get("bitrate", 0.0))

    rebuf_penalty = 4.3 * rebuffer
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = 1.0 * abs(bitrate - last_bitrate) / 1000.0

    return (bitrate / 1000.0) - rebuf_penalty - smooth_penalty
