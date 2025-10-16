"""
Pensieve QoE Reward Model
Stateless version - performs calculations only.
"""

class PensieveReward:
    def __init__(
        self,
        rebuffer_penalty=4.3,
        smoothness_penalty=1.0,
        bitrate_levels=None
    ):
        self.rebuffer_penalty = rebuffer_penalty
        self.smoothness_penalty = smoothness_penalty
        self.bitrate_levels = bitrate_levels if bitrate_levels is not None else [300, 750, 1850, 2850, 4300, 6000]
        self.M_IN_K = 1000.0
    
    def compute_reward(self, bitrate, rebuffer_time, last_bitrate):
        """
        Computes Pensieve QoE reward based on bitrate.
        All inputs must be provided by the caller.
        """
        # 1. Quality reward (bitrate in Mbps)
        quality_reward = bitrate / self.M_IN_K
        
        # 2. Rebuffering penalty
        rebuffer_penalty_val = self.rebuffer_penalty * rebuffer_time
        
        # 3. Smoothness penalty (bitrate change in Mbps)
        smoothness_penalty_val = self.smoothness_penalty * abs(bitrate - last_bitrate) / self.M_IN_K if last_bitrate > 0 else 0.0
        
        reward = quality_reward - rebuffer_penalty_val - smoothness_penalty_val
        return float(reward)

    def compute_reward_vmaf(self, vmaf_score, rebuffer_time, last_bitrate, current_bitrate):
        """
        Computes Pensieve QoE reward based on VMAF.
        All inputs must be provided by the caller.
        """
        # 1. Quality reward based on VMAF (scaled to match bitrate range)
        quality_reward = (vmaf_score / 100.0) * (self.bitrate_levels[-1] / self.M_IN_K)
        
        # 2. Rebuffering penalty
        rebuffer_penalty_val = self.rebuffer_penalty * rebuffer_time
        
        # 3. Smoothness penalty (based on bitrate change)
        smoothness_penalty_val = self.smoothness_penalty * abs(current_bitrate - last_bitrate) / self.M_IN_K if last_bitrate > 0 else 0.0
        
        reward = quality_reward - rebuffer_penalty_val - smoothness_penalty_val
        return float(reward)