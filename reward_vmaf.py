def compute_vmaf_reward(info, last_bitrate):
    """
    Reward = (VMAF_scaled) - 4.3 * rebuffer_time - 1.0 * smoothness_penalty
    VMAF_scaled: vmaf/100 * 6 (to match bitrate scale)
    """
    vmaf = float(info.get("vmaf", 0.0))
    rebuffer = float(info.get("rebuffer_time", 0.0))
    bitrate = float(info.get("bitrate", 0.0))

    quality = (vmaf / 100.0) * 6.0
    rebuf_penalty = 4.3 * rebuffer
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = 1.0 * abs(bitrate - last_bitrate) / 1000.0

    return quality - rebuf_penalty - smooth_penalty
