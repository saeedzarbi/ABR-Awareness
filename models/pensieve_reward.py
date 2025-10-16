"""
Pensieve QoE Reward Model
Based on: https://github.com/hongzimao/pensieve

QoE = Σ quality(n) - μ × rebuffer(n) - λ × smoothness(n)

Pensieve standard weights:
- μ (rebuffering penalty): 4.3
- λ (smoothness penalty): 1.0
"""


class PensieveReward:
    """
    Compute QoE reward exactly as Pensieve paper
    """
    
    def __init__(
        self,
        rebuffer_penalty=4.3,      # Pensieve standard
        smoothness_penalty=1.0,    # Pensieve standard
        bitrate_levels=[300, 750, 1850, 2850, 4300, 6000]
    ):
        self.rebuffer_penalty = rebuffer_penalty
        self.smoothness_penalty = smoothness_penalty
        self.bitrate_levels = bitrate_levels
        
        # For normalization
        self.M_IN_K = 1000.0  # Kbps to Mbps
    
    def compute_reward(self, bitrate, rebuffer_time, last_bitrate):
        """
        Compute Pensieve QoE reward (bitrate-based)
        
        Args:
            bitrate: selected bitrate (kbps)
            rebuffer_time: rebuffering time (seconds)
            last_bitrate: previous bitrate (kbps), 0 if first chunk
        
        Returns:
            reward: float
        """
        
        # 1. Quality reward (bitrate in Mbps)
        quality_reward = bitrate / self.M_IN_K
        
        # 2. Rebuffering penalty
        rebuffer_penalty_val = self.rebuffer_penalty * rebuffer_time
        
        # 3. Smoothness penalty (bitrate change in Mbps)
        if last_bitrate > 0:
            smoothness_penalty_val = self.smoothness_penalty * abs(bitrate - last_bitrate) / self.M_IN_K
        else:
            smoothness_penalty_val = 0.0
        
        # Total QoE
        reward = quality_reward - rebuffer_penalty_val - smoothness_penalty_val
        
        return float(reward)
    
    def compute_reward(self, action, rebuffer_time, vmaf_score):
        """
        Compute reward using Pensieve QoE model with VMAF.
        This version receives vmaf_score as an argument.
        """
        # Get bitrates (kbps)
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        # Compute Pensieve reward with VMAF
        reward = self.reward_func.compute_reward_vmaf(
            vmaf_score=vmaf_score,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate,
            current_bitrate=current_bitrate
        )
        
        # Debug extreme reward cases
        if reward < -100.0:
            logger.info(f"REWARD_DBG vmaf={vmaf_score:.1f} bitrate={current_bitrate}kbps "
                        f"rebuffer={rebuffer_time:.2f}s last_br={last_bitrate}kbps "
                        f"reward={reward:.2f}")
        
        return float(reward)


# ============================================
# Test
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Pensieve Reward Function")
    print("=" * 60)
    
    reward_func = PensieveReward()
    
    # Test cases
    test_cases = [
        # (bitrate, rebuffer_time, last_bitrate, description)
        (300, 0.0, 0, "Start with lowest bitrate, no rebuffer"),
        (750, 0.0, 300, "Increase bitrate, no rebuffer"),
        (1850, 1.0, 750, "Increase bitrate but rebuffer 1s"),
        (300, 0.0, 1850, "Drop to lowest, no rebuffer"),
        (6000, 5.0, 300, "Jump to highest with 5s rebuffer"),
    ]
    
    print("\nBitrate-based rewards (Pensieve original):")
    print("-" * 60)
    
    cumulative_reward = 0
    for bitrate, rebuffer, last_br, desc in test_cases:
        reward = reward_func.compute_reward(bitrate, rebuffer, last_br)
        cumulative_reward += reward
        
        print(f"  {desc}")
        print(f"    Bitrate: {bitrate:4d} kbps, Rebuffer: {rebuffer:.1f}s")
        print(f"    Reward: {reward:+7.3f}, Cumulative: {cumulative_reward:+7.3f}")
        print()
    
    # Test VMAF-based
    print("\nVMAF-based rewards (our extension):")
    print("-" * 60)
    
    vmaf_test_cases = [
        # (vmaf, bitrate, rebuffer, last_bitrate, description)
        (30, 300, 0.0, 0, "Low VMAF (30), 300 kbps, no rebuffer"),
        (50, 750, 0.0, 300, "Medium VMAF (50), 750 kbps"),
        (65, 1850, 1.0, 750, "Good VMAF (65), 1850 kbps, 1s rebuffer"),
        (30, 300, 0.0, 1850, "Low VMAF (30), drop to 300"),
        (87, 6000, 5.0, 300, "High VMAF (87), 6000 kbps, 5s rebuffer"),
    ]
    
    cumulative_reward = 0
    for vmaf, bitrate, rebuffer, last_br, desc in vmaf_test_cases:
        reward = reward_func.compute_reward_vmaf(vmaf, rebuffer, last_br, bitrate)
        cumulative_reward += reward
        
        print(f"  {desc}")
        print(f"    VMAF: {vmaf:2d}, Bitrate: {bitrate:4d} kbps, Rebuffer: {rebuffer:.1f}s")
        print(f"    Reward: {reward:+7.3f}, Cumulative: {cumulative_reward:+7.3f}")
        print()
    
    # Comparison
    print("=" * 60)
    print("Reward Formula:")
    print("=" * 60)
    print("Bitrate-based: quality - 4.3×rebuffer - 1.0×smoothness")
    print("  quality = bitrate_Mbps (0.3 to 6.0)")
    print()
    print("VMAF-based: quality - 4.3×rebuffer - 1.0×smoothness")
    print("  quality = (VMAF/100) × 6.0 (scaled to match bitrate range)")
    print()
    print("Example comparisons:")
    print("  1850 kbps, VMAF 53:")
    print("    Bitrate-based quality: 1.85")
    print("    VMAF-based quality:    3.18  (= 53/100 × 6.0)")
    print()
    print("  6000 kbps, VMAF 81:")
    print("    Bitrate-based quality: 6.00")
    print("    VMAF-based quality:    4.86  (= 81/100 × 6.0)")
    print("=" * 60)