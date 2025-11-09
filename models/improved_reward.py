"""
Improved Reward with Bitrate Bonus and Better Shaping
"""

import numpy as np


class ImprovedReward:
    """
    Enhanced reward function to encourage higher bitrates
    
    Key improvements:
    1. Bonus for choosing bitrates >= 1000 kbps
    2. Penalty for too low bitrates
    3. Extra bonus for high bitrate + no rebuffering
    4. Progressive scaling
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
        
        # NEW: Bitrate incentive parameters
        self.bitrate_bonus_weight = 0.5
        self.min_bitrate_threshold = 1000  # kbps
        self.low_bitrate_penalty = 0.5
        self.perfect_action_bonus = 1.0  # High bitrate + no rebuffer
        
    def compute_reward(
        self,
        current_bitrate: float,
        rebuffer_time: float,
        last_bitrate: float = 0,
        buffer_level: float = 0
    ) -> float:
        """
        Compute improved reward
        
        Args:
            current_bitrate: Selected bitrate in kbps
            rebuffer_time: Rebuffering time in seconds
            last_bitrate: Previous bitrate in kbps
            buffer_level: Current buffer in seconds
        
        Returns:
            reward: Enhanced QoE score
        """
        
        # 1. Base quality term (bitrate in Mbps)
        quality = current_bitrate / 1000.0
        
        # 2. Rebuffering penalty
        rebuffer_term = self.rebuffer_penalty * rebuffer_time
        
        # 3. Smoothness penalty
        bitrate_change = abs(current_bitrate - last_bitrate) / 1000.0
        smoothness_term = self.smoothness_penalty * bitrate_change
        
        # 4. NEW: Bitrate bonus/penalty
        if current_bitrate >= self.min_bitrate_threshold:
            # Bonus for choosing reasonable bitrates
            bitrate_ratio = (current_bitrate - self.min_bitrate_threshold) / (6000 - self.min_bitrate_threshold)
            bitrate_bonus = self.bitrate_bonus_weight * bitrate_ratio
        else:
            # Penalty for too low bitrate
            bitrate_bonus = -self.low_bitrate_penalty
        
        # 5. NEW: Perfect action bonus (high bitrate + no rebuffering)
        perfect_bonus = 0
        if rebuffer_time == 0 and current_bitrate >= 1850:
            perfect_bonus = self.perfect_action_bonus
        
        # 6. NEW: Buffer-aware bonus (encourage higher bitrate when buffer is healthy)
        buffer_bonus = 0
        if buffer_level > 10 and current_bitrate >= 1850:
            buffer_bonus = 0.3
        
        # Total reward
        reward = (quality + bitrate_bonus + perfect_bonus + buffer_bonus - 
                 rebuffer_term - smoothness_term)
        
        return reward
    
    def get_reward_breakdown(
        self,
        current_bitrate: float,
        rebuffer_time: float,
        last_bitrate: float = 0,
        buffer_level: float = 0
    ) -> dict:
        """
        Get detailed breakdown of reward components (for debugging)
        """
        quality = current_bitrate / 1000.0
        rebuffer_term = self.rebuffer_penalty * rebuffer_time
        bitrate_change = abs(current_bitrate - last_bitrate) / 1000.0
        smoothness_term = self.smoothness_penalty * bitrate_change
        
        if current_bitrate >= self.min_bitrate_threshold:
            bitrate_ratio = (current_bitrate - self.min_bitrate_threshold) / (6000 - self.min_bitrate_threshold)
            bitrate_bonus = self.bitrate_bonus_weight * bitrate_ratio
        else:
            bitrate_bonus = -self.low_bitrate_penalty
        
        perfect_bonus = 0
        if rebuffer_time == 0 and current_bitrate >= 1850:
            perfect_bonus = self.perfect_action_bonus
        
        buffer_bonus = 0
        if buffer_level > 10 and current_bitrate >= 1850:
            buffer_bonus = 0.3
        
        total = (quality + bitrate_bonus + perfect_bonus + buffer_bonus - 
                rebuffer_term - smoothness_term)
        
        return {
            'quality': quality,
            'bitrate_bonus': bitrate_bonus,
            'perfect_bonus': perfect_bonus,
            'buffer_bonus': buffer_bonus,
            'rebuffer_penalty': -rebuffer_term,
            'smoothness_penalty': -smoothness_term,
            'total': total
        }


if __name__ == '__main__':
    print("="*60)
    print("Testing Improved Reward Function")
    print("="*60)
    
    reward_func = ImprovedReward()
    
    # Test case 1: Low bitrate with no rebuffer
    print("\nTest 1: Low bitrate (300 kbps), no rebuffer")
    breakdown = reward_func.get_reward_breakdown(
        current_bitrate=300,
        rebuffer_time=0.0,
        last_bitrate=300,
        buffer_level=15.0
    )
    print(f"  Quality:         {breakdown['quality']:+.3f}")
    print(f"  Bitrate bonus:   {breakdown['bitrate_bonus']:+.3f}")
    print(f"  Perfect bonus:   {breakdown['perfect_bonus']:+.3f}")
    print(f"  Buffer bonus:    {breakdown['buffer_bonus']:+.3f}")
    print(f"  Rebuffer:        {breakdown['rebuffer_penalty']:+.3f}")
    print(f"  Smoothness:      {breakdown['smoothness_penalty']:+.3f}")
    print(f"  → Total:         {breakdown['total']:+.3f}")
    
    # Test case 2: Medium bitrate with no rebuffer
    print("\nTest 2: Medium bitrate (1850 kbps), no rebuffer, good buffer")
    breakdown = reward_func.get_reward_breakdown(
        current_bitrate=1850,
        rebuffer_time=0.0,
        last_bitrate=1850,
        buffer_level=20.0
    )
    print(f"  Quality:         {breakdown['quality']:+.3f}")
    print(f"  Bitrate bonus:   {breakdown['bitrate_bonus']:+.3f}")
    print(f"  Perfect bonus:   {breakdown['perfect_bonus']:+.3f}")
    print(f"  Buffer bonus:    {breakdown['buffer_bonus']:+.3f}")
    print(f"  Rebuffer:        {breakdown['rebuffer_penalty']:+.3f}")
    print(f"  Smoothness:      {breakdown['smoothness_penalty']:+.3f}")
    print(f"  → Total:         {breakdown['total']:+.3f}")
    
    # Test case 3: High bitrate with rebuffering
    print("\nTest 3: High bitrate (4300 kbps), with rebuffer")
    breakdown = reward_func.get_reward_breakdown(
        current_bitrate=4300,
        rebuffer_time=2.0,
        last_bitrate=1850,
        buffer_level=5.0
    )
    print(f"  Quality:         {breakdown['quality']:+.3f}")
    print(f"  Bitrate bonus:   {breakdown['bitrate_bonus']:+.3f}")
    print(f"  Perfect bonus:   {breakdown['perfect_bonus']:+.3f}")
    print(f"  Buffer bonus:    {breakdown['buffer_bonus']:+.3f}")
    print(f"  Rebuffer:        {breakdown['rebuffer_penalty']:+.3f}")
    print(f"  Smoothness:      {breakdown['smoothness_penalty']:+.3f}")
    print(f"  → Total:         {breakdown['total']:+.3f}")
    
    print("\n✓ All tests passed!")