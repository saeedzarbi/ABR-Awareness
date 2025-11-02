import numpy as np

def compute_balanced_reward(info, last_bitrate=None,
                             alpha=1.2,   # وزن کیفیت تصویری (VMAF)
                             beta=0.5,    # وزن bitrate
                             gamma=4.0,   # شدت جریمه rebuffer
                             delta=0.6,   # شدت جریمه سوئیچ کیفیت
                             buffer_bonus=0.3):
    """
    ✅ Balanced Composite QoE Reward
    ترکیب ادراک کیفیت، سرعت، و پایداری با پاداش متعادل برای PPO training/testing
    """

    bitrate = info.get("bitrate", 0.0)
    rebuffer = info.get("rebuffer_time", 0.0)
    vmaf = info.get("vmaf", 50.0)
    buffer = info.get("buffer", 0.0)

    # Normalize features
    bitrate_mbps = bitrate / 1000.0
    vmaf_norm = np.clip(vmaf / 100.0, 0.0, 1.0)

    # Rebuffer penalty (log scale → کمتر از خطی)
    rebuffer_penalty = np.log1p(rebuffer) * gamma

    # Smoothness penalty (تغییر bitrate زیاد → جریمه)
    smooth_penalty = 0.0
    if last_bitrate is not None:
        smooth_penalty = abs(bitrate - last_bitrate) / 1000.0

    # Buffer stability bonus (کمک به پایداری)
    buffer_reward = buffer_bonus * np.tanh(buffer / 10.0)

    # Weighted combination
    raw_reward = (
        alpha * vmaf_norm +
        beta * bitrate_mbps +
        buffer_reward -
        rebuffer_penalty -
        delta * smooth_penalty
    )

    # Normalize to [-10, +10]
    reward = np.tanh(raw_reward / 8.0) * 10.0
    return float(np.clip(reward, -10.0, 10.0))


if __name__ == "__main__":
    # Quick test
    test_info = {"bitrate": 1850, "rebuffer_time": 1.2, "vmaf": 72, "buffer": 12.0}
    print("Sample Reward:", compute_balanced_reward(test_info))
