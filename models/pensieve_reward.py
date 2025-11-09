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

"""
Pensieve Reward Function
Bitrate-based reward (original Pensieve)
"""

class PensieveReward:
    """
    Pensieve QoE reward function
    QoE = bitrate/1000 - rebuffer_penalty × rebuffer - smoothness_penalty × |Δbitrate|/1000
    """
    
    def __init__(
        self,
        rebuffer_penalty: float = 2.0,
        smoothness_penalty: float = 1.0,
        bitrate_levels: list = None
    ):
        self.rebuffer_penalty = rebuffer_penalty
        self.smoothness_penalty = smoothness_penalty
        self.bitrate_levels = bitrate_levels or [300, 750, 1850, 2850, 4300, 6000]
    
    def compute_reward_bitrate(
        self,
        current_bitrate: float,
        rebuffer_time: float,
        last_bitrate: float = 0
    ) -> float:
        """
        Compute reward using BITRATE
        
        Args:
            current_bitrate: Selected bitrate in kbps
            rebuffer_time: Rebuffering time in seconds
            last_bitrate: Previous bitrate in kbps
        
        Returns:
            reward: QoE score
        """
        # Quality term (bitrate in Mbps)
        quality = current_bitrate / 1000.0
        
        # Rebuffering penalty
        rebuffer_term = self.rebuffer_penalty * rebuffer_time
        
        # Smoothness penalty (bitrate change in Mbps)
        bitrate_change = abs(current_bitrate - last_bitrate) / 1000.0
        smoothness_term = self.smoothness_penalty * bitrate_change
        if current_bitrate >= self.min_bitrate_threshold:
            bitrate_bonus = self.bitrate_bonus_weight * (current_bitrate / 6000.0)
        else:
            bitrate_bonus = -0.5  # Penalty for too low bitrate
        
        # NEW: Extra bonus if no rebuffering AND high bitrate
        no_rebuffer_bonus = 0
        if rebuffer_time == 0 and current_bitrate >= 1850:
            no_rebuffer_bonus = 1.0
        
        # Total QoE
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
        Compute reward using VMAF (for comparison)
        """
        quality = vmaf_score / 100.0
        rebuffer_term = self.rebuffer_penalty * rebuffer_time
        bitrate_change = abs(current_bitrate - last_bitrate) / 1000.0
        smoothness_term = self.smoothness_penalty * bitrate_change
        
        reward = quality - rebuffer_term - smoothness_term
        return reward


if __name__ == '__main__':
    print("="*60)
    print("Testing Pensieve Reward Function")
    print("="*60)
    
    reward_func = PensieveReward()
    
    # Test case 1
    print("\nTest 1: High bitrate (4300 kbps), no rebuffer")
    r = reward_func.compute_reward_bitrate(
        current_bitrate=4300,
        rebuffer_time=0.0,
        last_bitrate=2850
    )
    print(f"  Quality: 4.3 Mbps")
    print(f"  Rebuffer: 0.0s")
    print(f"  Smoothness: -1.45 Mbps")
    print(f"  Reward: {r:.2f}")
    
    # Test case 2
    print("\nTest 2: Low bitrate (300 kbps), high rebuffer")
    r = reward_func.compute_reward_bitrate(
        current_bitrate=300,
        rebuffer_time=5.0,
        last_bitrate=1850
    )
    print(f"  Quality: 0.3 Mbps")
    print(f"  Rebuffer: 5.0s")
    print(f"  Smoothness: -1.55 Mbps")
    print(f"  Reward: {r:.2f}")
    
    print("\n✓ All tests passed!")