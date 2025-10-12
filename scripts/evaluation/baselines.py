"""
Baseline ABR algorithms for comparison
"""

import numpy as np


class FixedBitratePolicy:
    """Always select the same bitrate"""
    
    def __init__(self, bitrate_index):
        self.bitrate_index = bitrate_index
    
    def select_action(self, state):
        return self.bitrate_index


class BufferBasedPolicy:
    """
    Simple buffer-based policy (BBA-like)
    Low buffer → low bitrate
    High buffer → high bitrate
    """
    
    def __init__(self, bitrate_levels=[300, 750, 1850, 2850, 4300, 6000]):
        self.bitrate_levels = bitrate_levels
        self.num_bitrates = len(bitrate_levels)
    
    def select_action(self, state):
        # Extract buffer level from network state
        buffer = state['network'][2, -1] * 60.0  # Denormalize
        
        # Thresholds (seconds)
        if buffer < 5:
            return 0  # Lowest bitrate
        elif buffer < 10:
            return 1
        elif buffer < 20:
            return 2
        elif buffer < 30:
            return 3
        elif buffer < 45:
            return 4
        else:
            return 5  # Highest bitrate


class ThroughputBasedPolicy:
    """
    Select bitrate based on past throughput
    (similar to rate-based algorithms)
    """
    
    def __init__(self, bitrate_levels=[300, 750, 1850, 2850, 4300, 6000], safety_factor=0.85):
        self.bitrate_levels = bitrate_levels
        self.safety_factor = safety_factor
    
    def select_action(self, state):
        # Extract past throughput
        past_throughput = state['network'][0, :] * 1000.0  # Denormalize to kbps
        
        # Get average throughput (last 3)
        if np.any(past_throughput > 0):
            avg_throughput = np.mean(past_throughput[past_throughput > 0][-3:])
        else:
            avg_throughput = 1000.0  # Default
        
        # Select bitrate with safety margin
        target_bitrate = avg_throughput * self.safety_factor
        
        # Find closest bitrate below target
        action = 0
        for i, br in enumerate(self.bitrate_levels):
            if br <= target_bitrate:
                action = i
            else:
                break
        
        return action


class PensieveBaselinePolicy:
    """
    Pensieve baseline (without content features)
    This would need a trained model, but for now we'll simulate
    """
    
    def __init__(self, model=None):
        self.model = model
    
    def select_action(self, state):
        if self.model is None:
            # Fallback: throughput-based
            policy = ThroughputBasedPolicy()
            return policy.select_action(state)
        
        # Use trained baseline model
        import torch
        network_state = torch.FloatTensor(state['network']).unsqueeze(0)
        
        with torch.no_grad():
            # Baseline Pensieve only uses network state
            # We'd need to implement baseline model here
            pass


def get_baseline_policy(name):
    """Factory function to get baseline policy"""
    
    policies = {
        'fixed_low': FixedBitratePolicy(0),      # Always 300 kbps
        'fixed_mid': FixedBitratePolicy(2),      # Always 1850 kbps
        'fixed_high': FixedBitratePolicy(5),     # Always 6000 kbps
        'buffer_based': BufferBasedPolicy(),
        'throughput_based': ThroughputBasedPolicy(),
    }
    
    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")
    
    return policies[name]
