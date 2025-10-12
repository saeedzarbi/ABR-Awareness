"""
Pensieve QoE Reward Model
Based on: https://github.com/hongzimao/pensieve

QoE = Σ quality(n) - μ × rebuffer(n) - λ × smoothness(n)

Original Pensieve weights:
- μ (rebuffering penalty): 4.3
- λ (smoothness penalty): 1.0
"""


class PensieveReward:
    """
    Compute QoE reward exactly as Pensieve paper
    """
    
    def __init__(
        self,
        rebuffer_penalty=4.3,  # Pensieve constant
        smoothness_penalty=1.0,  # Pensieve constant
        bitrate_levels=[300, 750, 1850, 2850, 4300, 6000]
    ):
        self.rebuffer_penalty = rebuffer_penalty
        self.smoothness_penalty = smoothness_penalty
        self.bitrate_levels = bitrate_levels
        
        # For normalization
        self.M_IN_K = 1000.0  # Mbps to Kbps
    
    def compute_reward(self, bitrate, rebuffer_time, last_bitrate):
        """
        Compute Pensieve QoE reward
        
        Args:
            bitrate: selected bitrate (kbps)
            rebuffer_time: rebuffering time (seconds)
            last_bitrate: previous bitrate (kbps), 0 if first chunk
        
        Returns:
            reward: float
        """
        
        # 1. Quality reward (linear in bitrate, normalized to Mbps)
        # Pensieve uses bitrate directly (in Mbps)
        quality_reward = bitrate / self.M_IN_K  # Convert to Mbps
        
        # 2. Rebuffering penalty
        rebuffer_penalty_val = self.rebuffer_penalty * rebuffer_time
        
        # 3. Smoothness penalty (absolute bitrate change, normalized)
        if last_bitrate > 0:
            smoothness_penalty_val = self.smoothness_penalty * abs(bitrate - last_bitrate) / self.M_IN_K
        else:
            smoothness_penalty_val = 0.0
        
        # Total QoE
        reward = quality_reward - rebuffer_penalty_val - smoothness_penalty_val
        
        return float(reward)
    
    def compute_reward_vmaf(self, vmaf_score, rebuffer_time, last_bitrate, current_bitrate):
        """
        VMAF-based version (our extension)
        
        Args:
            vmaf_score: VMAF score (0-100)
            rebuffer_time: rebuffering time (seconds)
            last_bitrate: previous bitrate (kbps)
            current_bitrate: current bitrate (kbps)
        
        Returns:
            reward: float
        """
        
        # 1. Quality reward (VMAF-based instead of bitrate)
        # Normalize VMAF to similar scale as bitrate (0-100 -> 0-6 Mbps equivalent)
        quality_reward = vmaf_score / 100.0 * 6.0  # Scale to max bitrate
        
        # 2. Rebuffering penalty
        rebuffer_penalty_val = self.rebuffer_penalty * rebuffer_time
        
        # 3. Smoothness penalty
        if last_bitrate > 0:
            smoothness_penalty_val = self.smoothness_penalty * abs(current_bitrate - last_bitrate) / self.M_IN_K
        else:
            smoothness_penalty_val = 0.0
        
        # Total QoE
        reward = quality_reward - rebuffer_penalty_val - smoothness_penalty_val
        
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
        (30, 300, 0.0, 0, "Low VMAF, no rebuffer"),
        (50, 750, 0.0, 300, "Medium VMAF, no rebuffer"),
        (65, 1850, 1.0, 750, "Good VMAF but rebuffer"),
        (30, 300, 0.0, 1850, "Low VMAF after drop"),
        (87, 6000, 5.0, 300, "Highest VMAF with rebuffer"),
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
    print("Key Insights:")
    print("=" * 60)
    print("1. Rebuffering dominates: 1s rebuffer = -4.3 reward")
    print("2. Smoothness matters: Large bitrate changes penalized")
    print("3. VMAF-based can give different ordering than bitrate-based")
    print("=" * 60)
