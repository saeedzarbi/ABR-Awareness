"""
Buffer-Based Approach (BBA) Baseline.
Based on: "A Buffer-Based Approach to Rate Adaptation: Evidence from a Large Video Streaming Service" (SIGCOMM '14)
"""

import numpy as np

class BBA:
    """
    Buffer-Based Algorithm (BBA-2).
    Maps buffer level linearly to bitrate.
    """
    def __init__(self, bitrate_levels, reservoir=3.0, cushion=12.0):
        """
        Args:
            bitrate_levels: List of available bitrates (Kbps)
            reservoir (r): Minimum buffer safety margin (seconds). Below this, pick min bitrate.
            cushion (c): Size of the linear adaptation region (seconds). 
                         Max bitrate is chosen when buffer > reservoir + cushion.
        """
        self.bitrate_levels = np.array(bitrate_levels)
        self.reservoir = reservoir
        self.cushion = cushion
        
        # Upper threshold (reservoir + cushion)
        self.upper_threshold = reservoir + cushion

    def select_bitrate(self, buffer_level):
        """
        Select bitrate based on current buffer level.
        
        Formula:
            f(B) = (B - r) / c
            Rate = f(B) * (MaxRate - MinRate) + MinRate
        """
        
        # 1. Critical Zone (Panic Mode)
        if buffer_level <= self.reservoir:
            return 0  # Lowest quality
            
        # 2. Stable Zone (Max Quality)
        if buffer_level >= self.upper_threshold:
            return len(self.bitrate_levels) - 1  # Highest quality
            
        # 3. Linear Adaptation Zone (Slope)
        # Calculate the fraction of the cushion that is full (0.0 to 1.0)
        fraction = (buffer_level - self.reservoir) / self.cushion
        
        # Map fraction to bitrate index
        # We want to map [0, 1] to [0, len-1]
        # Using the continuous rate formula from the paper and snapping to nearest available level
        
        # Ideal continuous bitrate based on buffer
        min_rate = self.bitrate_levels[0]
        max_rate = self.bitrate_levels[-1]
        target_rate = min_rate + fraction * (max_rate - min_rate)
        
        # Find the bitrate level closest to (or just below) the target rate
        # BBA usually suggests picking the largest rate <= target_rate for safety
        
        # Method A: Closest
        # action = (np.abs(self.bitrate_levels - target_rate)).argmin()
        
        # Method B: Conservative (Largest rate <= target) -> Better for stability
        valid_indices = np.where(self.bitrate_levels <= target_rate)[0]
        if len(valid_indices) > 0:
            action = valid_indices[-1]
        else:
            action = 0
            
        return action

# Test block
if __name__ == "__main__":
    bitrates = [300, 750, 1200, 1850, 2850, 6000]
    bba = BBA(bitrates)
    
    print("Testing BBA Logic:")
    test_buffers = [1.0, 4.0, 8.0, 12.0, 15.0, 20.0]
    
    for buf in test_buffers:
        act = bba.select_bitrate(buf)
        print(f"Buffer: {buf:4.1f}s -> Action: {act} ({bitrates[act]} Kbps)")