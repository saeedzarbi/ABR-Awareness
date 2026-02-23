"""
Robust MPC optimized for VMAF (Content-Aware Control).
Standard baseline for IEEE TCSVT comparisons.
"""

import numpy as np
import itertools

class RobustMPC:
    def __init__(self, env, lookahead=5):
        self.env = env
        self.lookahead = lookahead
        self.past_throughput = []
        
        # Create all possible bitrate trajectories for lookahead
        # Actions: 0 to 5
        self.possible_actions = list(range(len(env.BITRATE_LEVELS)))
        # We limit lookahead depth to keep it fast (complexity is 6^horizon)
        # Horizon 3 is usually enough for real-time emulation
        self.search_horizon = 3 
        self.trajectories = list(itertools.product(self.possible_actions, repeat=self.search_horizon))

    def estimate_throughput(self, buffer_level):
        """Robust throughput estimation using Harmonic Mean."""
        if not self.past_throughput:
            return 2000.0 # Default start
        
        # Harmonic mean is robust to outliers (spikes)
        # Using last 5 samples
        samples = self.past_throughput[-5:]
        harmonic_mean = len(samples) / sum(1.0 / (t + 1e-6) for t in samples)
        return harmonic_mean

    def select_bitrate(self, buffer_level, last_throughput_kbps, last_vmaf):
        """
        Optimize for: Maximize VMAF - Penalty, over a future horizon.
        """
        # Update history
        if last_throughput_kbps > 0:
            self.past_throughput.append(last_throughput_kbps)
            
        predicted_throughput = self.estimate_throughput(buffer_level)
        
        best_action = 0
        max_reward = -float('inf')
        
        # Current VMAF scores for all bitrates (simplified lookup)
        # In a real scenario, we'd need lookahead VMAF features. 
        # Here we assume we can see VMAF of next chunks (like your Proposed method).
        # Since exact lookahead isn't passed in simple evaluate loop, 
        # we use current chunk's VMAF distribution as an approximation for future chunks.
        # This makes MPC a "fair" competitor.
        
        # Get current chunk VMAF options
        current_vmaf_options = self.env.vmaf_scores 
        
        for trajectory in self.trajectories:
            cumulative_reward = 0
            sim_buffer = buffer_level
            sim_last_vmaf = last_vmaf
            
            for step, action in enumerate(trajectory):
                bitrate_kbps = self.env.BITRATE_LEVELS[action]
                
                # 1. Quality
                # We approximate future VMAF as similar to current chunk's VMAF profile
                # (This is a standard assumption in MPC when lookahead metadata isn't perfect)
                vmaf = current_vmaf_options.get(bitrate_kbps, 35.0)
                
                # 2. Rebuffer
                chunk_size = bitrate_kbps * 4.0 * 1000 # bits
                download_time = chunk_size / (predicted_throughput * 1000 + 1e-6)
                rebuffer = max(0, download_time - sim_buffer)
                sim_buffer = max(0, sim_buffer - download_time) + 4.0
                
                # 3. Smoothness
                smoothness = abs(vmaf - sim_last_vmaf)
                
                reward = vmaf \
                         - (self.env.REBUF_PENALTY_BASE * rebuffer) \
                         - (self.env.SMOOTH_PENALTY_WEIGHT * smoothness)
                
                cumulative_reward += reward
                sim_last_vmaf = vmaf
                
                # Stop if buffer explodes (unrealistic path) or crashes
                if sim_buffer < 0: 
                    cumulative_reward -= 1000
                    break
            
            if cumulative_reward > max_reward:
                max_reward = cumulative_reward
                best_action = trajectory[0] # We only take the first step
                
        return best_action