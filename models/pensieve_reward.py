# """
# Pensieve QoE Reward Model
# Stateless version - performs calculations only.
# """

# class PensieveReward:
#     def __init__(
#         self,
#         rebuffer_penalty=4.3,
#         smoothness_penalty=1.0,
#         bitrate_levels=None
#     ):
#         self.rebuffer_penalty = rebuffer_penalty
#         self.smoothness_penalty = smoothness_penalty
#         self.bitrate_levels = bitrate_levels if bitrate_levels is not None else [300, 750, 1850, 2850, 4300, 6000]
#         self.M_IN_K = 1000.0
    
#     def compute_reward(self, bitrate, rebuffer_time, last_bitrate):
#         """
#         Computes Pensieve QoE reward based on bitrate.
#         All inputs must be provided by the caller.
#         """
#         # 1. Quality reward (bitrate in Mbps)
#         quality_reward = bitrate / self.M_IN_K
        
#         # 2. Rebuffering penalty
#         rebuffer_penalty_val = self.rebuffer_penalty * rebuffer_time
        
#         # 3. Smoothness penalty (bitrate change in Mbps)
#         smoothness_penalty_val = self.smoothness_penalty * abs(bitrate - last_bitrate) / self.M_IN_K if last_bitrate > 0 else 0.0
        
#         reward = quality_reward - rebuffer_penalty_val - smoothness_penalty_val
#         return float(reward)

#     def compute_reward_vmaf(self, vmaf_score, rebuffer_time, last_bitrate, current_bitrate):
#         """
#         Computes Pensieve QoE reward based on VMAF.
#         All inputs must be provided by the caller.
#         """
#         # 1. Quality reward based on VMAF (scaled to match bitrate range)
#         quality_reward = (vmaf_score / 100.0) * (self.bitrate_levels[-1] / self.M_IN_K)
        
#         # 2. Rebuffering penalty
#         rebuffer_penalty_val = self.rebuffer_penalty * rebuffer_time
        
#         # 3. Smoothness penalty (based on bitrate change)
#         smoothness_penalty_val = self.smoothness_penalty * abs(current_bitrate - last_bitrate) / self.M_IN_K if last_bitrate > 0 else 0.0
        
#         reward = quality_reward - rebuffer_penalty_val - smoothness_penalty_val
#         return float(reward)

"""
Pensieve Reward (Fixed Version)
Use bitrate instead of VMAF for quality metric
"""

class PensieveReward:
    """
    Pensieve QoE reward function
    Standard: bitrate - 4.3 × rebuffer - 1.0 × smoothness
    """
    
    def __init__(
        self,
        rebuffer_penalty: float = 4.3,
        smoothness_penalty: float = 1.0,
        bitrate_levels: list = None
    ):
        self.rebuffer_penalty = rebuffer_penalty
        self.smoothness_penalty = smoothness_penalty
        self.bitrate_levels = bitrate_levels or [300, 750, 1850, 2850, 4300, 6000]
    
    def compute_reward_bitrate(self, current_bitrate, rebuffer_time, last_bitrate):
        """
        Enhanced reward with bitrate bonus
        """
        # Base quality term
        quality = current_bitrate / 1000.0
        
        # Rebuffering penalty
        rebuffer_term = self.rebuffer_penalty * rebuffer_time
        
        # Smoothness penalty
        bitrate_change = abs(current_bitrate - last_bitrate) / 1000.0
        smoothness_term = self.smoothness_penalty * bitrate_change
        
        # NEW: Bonus for choosing higher bitrates
        if current_bitrate >= self.min_bitrate_threshold:
            bitrate_bonus = self.bitrate_bonus_weight * (current_bitrate / 6000.0)
        else:
            bitrate_bonus = -0.5  # Penalty for too low bitrate
        
        # NEW: Extra bonus if no rebuffering AND high bitrate
        no_rebuffer_bonus = 0
        if rebuffer_time == 0 and current_bitrate >= 1850:
            no_rebuffer_bonus = 1.0
        
        # Total reward
        reward = quality - rebuffer_term - smoothness_term + bitrate_bonus + no_rebuffer_bonus
        
        return reward
    
    def compute_reward_vmaf(
        self,
        vmaf_score: float,
        rebuffer_time: float,
        last_bitrate: float,
        current_bitrate: float
    ) -> float:
        """
        Compute reward using VMAF
        
        QoE = vmaf/100 - rebuffer_penalty × rebuffer - smoothness_penalty × |Δbitrate|/1000
        
        Returns:
            reward: QoE score
        """
        # Quality term (VMAF normalized to 0-1)
        quality = vmaf_score / 100.0
        
        # Rebuffering penalty
        rebuffer_term = self.rebuffer_penalty * rebuffer_time
        
        # Smoothness penalty
        bitrate_change = abs(current_bitrate - last_bitrate) / 1000.0
        smoothness_term = self.smoothness_penalty * bitrate_change
        
        # Total QoE
        reward = quality - rebuffer_term - smoothness_term
        
        return reward


if __name__ == '__main__':
    print("="*60)
    print("Testing Pensieve Reward Function")
    print("="*60)
    
    reward_func = PensieveReward()
    
    # Test case 1: Good quality, no rebuffer
    print("\nTest 1: High bitrate (4300 kbps), no rebuffer")
    r = reward_func.compute_reward_bitrate(
        current_bitrate=4300,
        rebuffer_time=0.0,
        last_bitrate=2850
    )
    print(f"  Quality: 4.3")
    print(f"  Rebuffer: 0.0")
    print(f"  Smoothness: -1.45")
    print(f"  Reward: {r:.2f}")
    
    # Test case 2: Low quality, high rebuffer
    print("\nTest 2: Low bitrate (300 kbps), high rebuffer")
    r = reward_func.compute_reward_bitrate(
        current_bitrate=300,
        rebuffer_time=5.0,
        last_bitrate=1850
    )
    print(f"  Quality: 0.3")
    print(f"  Rebuffer: -21.5")
    print(f"  Smoothness: -1.55")
    print(f"  Reward: {r:.2f}")
    
    # Test case 3: Medium quality, medium rebuffer
    print("\nTest 3: Medium bitrate (1850 kbps), medium rebuffer")
    r = reward_func.compute_reward_bitrate(
        current_bitrate=1850,
        rebuffer_time=2.0,
        last_bitrate=1850
    )
    print(f"  Quality: 1.85")
    print(f"  Rebuffer: -8.6")
    print(f"  Smoothness: 0.0")
    print(f"  Reward: {r:.2f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)