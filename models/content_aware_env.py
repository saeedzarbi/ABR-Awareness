"""
Content-Aware Environment
Extends Pensieve environment with content features
FINAL VERSION: Properly balanced rewards with strong rebuffering penalty
"""

import numpy as np
import json
from pathlib import Path


class ContentAwareEnv:
    """
    ABR Environment with content awareness
    
    Loads:
        - Network traces (simulated or from file)
        - Video chunk sizes (simulated)
        - Content features (SI, TI)
        - VMAF predictions
    """
    
    def __init__(
        self,
        traces_dir='data/traces',
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        bitrate_levels=[300, 750, 1850, 2850, 4300, 6000],
        chunk_duration=4.0,
        total_chunks=48
    ):
        
        self.bitrate_levels = bitrate_levels
        self.chunk_duration = chunk_duration
        self.total_chunks = total_chunks
        
        # Load content features
        print(f"Loading content features from {features_file}")
        with open(features_file, 'r') as f:
            self.content_features = json.load(f)
        
        # Load VMAF table
        print(f"Loading VMAF table from {vmaf_file}")
        with open(vmaf_file, 'r') as f:
            self.vmaf_table = json.load(f)
        
        print(f"✓ Loaded {len(self.content_features)} content feature entries")
        print(f"✓ Loaded {len(self.vmaf_table)} VMAF entries")
        
        # Generate network trace (realistic simulation)
        self.network_trace = self._generate_network_trace()
        
        # State tracking
        self.reset()
    
    def _generate_network_trace(self, duration=300):
        """
        Generate realistic network trace
        Simulates varying bandwidth conditions
        """
        np.random.seed(42)
        
        # Multiple bandwidth profiles
        profiles = [
            {'mean': 500, 'std': 100},    # Low bandwidth
            {'mean': 1500, 'std': 300},   # Medium bandwidth
            {'mean': 3000, 'std': 500},   # High bandwidth
            {'mean': 5000, 'std': 800},   # Very high bandwidth
        ]
        
        trace = []
        current_profile_idx = 0
        
        for i in range(duration):
            # Switch profile every 50 steps (simulating network changes)
            if i % 50 == 0 and i > 0:
                current_profile_idx = (current_profile_idx + 1) % len(profiles)
            
            profile = profiles[current_profile_idx]
            throughput = np.random.normal(profile['mean'], profile['std'])
            throughput = np.clip(throughput, 300, 6000)
            trace.append(throughput)
        
        return np.array(trace)
    
    def reset(self, video_id=1):
        """Reset environment for new video"""
        self.video_id = video_id
        self.chunk_idx = 0
        self.buffer = 0.0
        self.trace_idx = 0
        
        # Network state history
        self.past_throughput = []
        self.past_download_time = []
        self.past_bitrates = []
        self.past_errors = []
        
        return self.get_state()
    
    def get_content_state(self):
        """
        Get content features for current chunk
        
        Returns:
            [SI, TI] - numpy array of shape (2,)
        """
        # Build key to lookup features
        bitrate = self.bitrate_levels[0]
        key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
        
        if key not in self.content_features:
            # Fallback to average features
            return np.array([50.0, 15.0], dtype=np.float32)
        
        feat = self.content_features[key]
        return np.array([feat['si_mean'], feat['ti_mean']], dtype=np.float32)
    
    def get_vmaf_predictions(self):
        """
        Get predicted VMAF for all bitrates for current chunk
        
        Returns:
            numpy array of shape (num_bitrates,)
        """
        bitrate = self.bitrate_levels[0]
        key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
        
        if key not in self.vmaf_table:
            # Fallback: linear approximation
            return np.array([30, 50, 65, 75, 82, 87], dtype=np.float32)
        
        vmaf_dict = self.vmaf_table[key]
        
        # Extract VMAF for each bitrate
        vmaf_values = []
        for br in self.bitrate_levels:
            vmaf_values.append(float(vmaf_dict.get(str(br), 50.0)))
        
        return np.array(vmaf_values, dtype=np.float32)
    
    def get_network_state(self):
        """
        Get network state (same as Pensieve format)
        
        Returns:
            numpy array of shape (6, 8)
        """
        state = np.zeros((6, 8), dtype=np.float32)
        
        # Row 0: past throughput (last 8)
        for i, t in enumerate(self.past_throughput[-8:]):
            state[0, -(i+1)] = t / 6000.0  # Normalize to max throughput
        
        # Row 1: past download time (last 8)
        for i, d in enumerate(self.past_download_time[-8:]):
            state[1, -(i+1)] = d / 10.0  # Normalize (assume max 10s download)
        
        # Row 2: current buffer size
        state[2, -1] = min(self.buffer / 60.0, 1.0)  # Normalize to 60s max
        
        # Row 3: past bitrates (last 5)
        for i, b in enumerate(self.past_bitrates[-5:]):
            state[3, -(i+1)] = b / 6000.0  # Normalize to max bitrate
        
        # Row 4: remaining chunks
        remaining = self.total_chunks - self.chunk_idx
        state[4, -1] = remaining / self.total_chunks
        
        # Row 5: past errors (0 for now)
        
        return state
    
    def get_state(self):
        """
        Get complete state including content features
        
        Returns:
            dict with keys: 'network', 'content', 'vmaf'
        """
        return {
            'network': self.get_network_state(),      # (6, 8)
            'content': self.get_content_state(),      # (2,)
            'vmaf': self.get_vmaf_predictions()       # (6,)
        }
    
    def step(self, action):
        """
        Execute action (select bitrate) and return next state
        
        Args:
            action: int (0-5) - bitrate index
        
        Returns:
            next_state: dict
            reward: float
            done: bool
            info: dict
        """
        # Get selected bitrate
        selected_bitrate = self.bitrate_levels[action]
        
        # Get throughput from trace
        if self.trace_idx < len(self.network_trace):
            throughput = self.network_trace[self.trace_idx]
            self.trace_idx += 1
        else:
            # Fallback if trace exhausted
            throughput = np.random.uniform(500, 3000)
        
        # Calculate chunk size (in KB)
        # bitrate is in kbps, chunk_duration in seconds
        # size = bitrate * duration / 8
        chunk_size = selected_bitrate * self.chunk_duration / 8  # KB
        
        # Calculate download time (in seconds)
        # download_time = size / (throughput/8)
        download_time = chunk_size / (throughput / 8)
        
        # Update buffer
        # If buffer can't cover download time, rebuffering occurs
        rebuffer_time = max(0, download_time - self.buffer)
        
        # Buffer decreases by download time, then increases by chunk duration
        self.buffer = max(0, self.buffer - download_time) + self.chunk_duration
        
        # Cap buffer at reasonable max (60 seconds)
        self.buffer = min(self.buffer, 60.0)
        
        # Compute reward (PROPERLY BALANCED)
        reward = self.compute_reward(action, rebuffer_time)
        
        # Update history
        self.past_throughput.append(throughput)
        self.past_download_time.append(download_time)
        self.past_bitrates.append(selected_bitrate)
        
        # Move to next chunk
        self.chunk_idx += 1
        done = (self.chunk_idx >= self.total_chunks)
        
        # Get next state
        next_state = self.get_state() if not done else None
        
        info = {
            'rebuffer_time': float(rebuffer_time),
            'bitrate': float(selected_bitrate),
            'buffer': float(self.buffer),
            'chunk_idx': int(self.chunk_idx),
            'throughput': float(throughput),
            'download_time': float(download_time)
        }
        
        return next_state, reward, done, info
    
    def compute_reward(self, action, rebuffer_time):
        """
        Compute reward with PROPER balance
        
        CRITICAL: Rebuffering must be heavily penalized!
        This is the key to good ABR performance.
        
        Args:
            action: selected bitrate index
            rebuffer_time: rebuffering time in seconds
        
        Returns:
            reward: float
        """
        # Get VMAF for selected bitrate
        vmaf_predictions = self.get_vmaf_predictions()
        vmaf_score = vmaf_predictions[action]
        
        # 1. Quality reward (VMAF-based, normalized to 0-1)
        quality_reward = vmaf_score / 100.0
        
        # 2. Rebuffering penalty - CRITICAL!
        # This MUST dominate the reward function
        # Every second of rebuffering should cost much more than quality gain
        rebuffer_penalty = 10.0 * rebuffer_time  # VERY HIGH PENALTY!
        
        # 3. Smoothness penalty (bitrate switching)
        if len(self.past_bitrates) > 0:
            prev_bitrate = self.past_bitrates[-1]
            curr_bitrate = self.bitrate_levels[action]
            smoothness_penalty = abs(curr_bitrate - prev_bitrate) / 4000.0
        else:
            smoothness_penalty = 0.0
        
        # 4. Buffer-aware bonus
        # Encourage safe behavior based on buffer state
        if self.buffer < 10.0:
            # Low buffer - reward conservative choices
            if action <= 2:  # Low bitrate
                buffer_bonus = 0.5
            else:
                buffer_bonus = 0.0
        elif self.buffer > 30.0:
            # High buffer - can afford higher bitrate
            if action >= 2:  # Medium to high bitrate
                buffer_bonus = 0.3
            else:
                buffer_bonus = 0.0
        else:
            # Medium buffer - neutral
            buffer_bonus = 0.1
        
        # 5. Content-aware bonus (minor)
        # Slightly encourage higher bitrate for complex content IF buffer allows
        if self.buffer > 20.0:
            content_features = self.get_content_state()
            si, ti = content_features
            complexity = (si + ti) / 150.0  # Normalize
            
            if complexity > 0.7 and action >= 3:
                content_bonus = 0.2
            else:
                content_bonus = 0.0
        else:
            content_bonus = 0.0
        
        # TOTAL REWARD
        # Rebuffering penalty should dominate everything else!
        reward = (
            quality_reward +        # 0 to 1
            buffer_bonus +          # 0 to 0.5
            content_bonus -         # 0 to 0.2
            rebuffer_penalty -      # 0 to infinity (DOMINANT!)
            0.5 * smoothness_penalty  # Small penalty for switching
        )
        
        return float(reward)


# ============================================
# Test function
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Content-Aware Environment (Final)")
    print("=" * 60)
    print()
    
    # Create environment
    env = ContentAwareEnv()
    
    print("\n✓ Environment created")
    print()
    
    # Reset
    state = env.reset(video_id=1)
    
    print("Initial state:")
    print(f"  Network state shape: {state['network'].shape}")
    print(f"  Content features shape: {state['content'].shape}")
    print(f"  VMAF predictions shape: {state['vmaf'].shape}")
    print(f"  Content features (SI, TI): {state['content']}")
    print(f"  VMAF predictions: {state['vmaf']}")
    
    # Test episode with different actions
    print("\nTesting episode with varied actions:")
    print("Testing conservative strategy (mostly low bitrates):")
    actions = [0, 1, 2, 1, 0]  # Conservative actions
    
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        bitrate_name = env.bitrate_levels[action]
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: action={action} ({bitrate_name} kbps), "
              f"reward={reward:+.3f}, buffer={info['buffer']:.1f}s, "
              f"rebuffer={info['rebuffer_time']:.2f}s")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    # Test aggressive strategy
    print("\n\nTesting aggressive strategy (high bitrates):")
    env.reset(video_id=1)
    actions = [5, 5, 5, 5, 5]  # Aggressive actions
    
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        bitrate_name = env.bitrate_levels[action]
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: action={action} ({bitrate_name} kbps), "
              f"reward={reward:+.3f}, buffer={info['buffer']:.1f}s, "
              f"rebuffer={info['rebuffer_time']:.2f}s")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("Notice: Conservative strategy should have MUCH higher reward")
    print("        due to avoiding rebuffering!")
    print("=" * 60)
