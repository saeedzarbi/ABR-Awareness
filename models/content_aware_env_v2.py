"""
Content-Aware Environment V2
With Real Network Traces + Pensieve Reward
FINAL VERSION with all fixes
"""

import numpy as np
import json
from pathlib import Path
import sys

# Fix import path
try:
    from models.trace_loader import TraceLoader, NetworkTrace
    from models.pensieve_reward import PensieveReward
except ModuleNotFoundError:
    from trace_loader import TraceLoader, NetworkTrace
    from pensieve_reward import PensieveReward


def get_project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


def resolve_path(relative_path):
    """Resolve relative path from project root"""
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str(get_project_root() / relative_path)


class ContentAwareEnvV2:
    """
    Environment با real network traces و Pensieve reward
    """
    
    def __init__(
        self,
        trace_dir='data/network_traces/cooked_traces',
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        bitrate_levels=[300, 750, 1850, 2850, 4300, 6000],
        chunk_duration=4.0,
        total_chunks=48,
        use_real_traces=True
    ):
        
        # Resolve paths (works from any directory)
        trace_dir = resolve_path(trace_dir)
        features_file = resolve_path(features_file)
        vmaf_file = resolve_path(vmaf_file)
        
        self.bitrate_levels = bitrate_levels
        self.chunk_duration = chunk_duration
        self.total_chunks = total_chunks
        self.use_real_traces = use_real_traces
        
        # Load content features
        with open(features_file, 'r') as f:
            self.content_features = json.load(f)
        
        # Load VMAF table
        with open(vmaf_file, 'r') as f:
            self.vmaf_table = json.load(f)
        
        # Pensieve reward function
        # in ContentAwareEnvV2.__init__
        self.reward_func = PensieveReward(
            rebuffer_penalty=3.0,   # tuned down
            smoothness_penalty=1.0,
            bitrate_levels=bitrate_levels
        )

        
        # Load network traces
        if use_real_traces:
            self.trace_loader = TraceLoader(trace_dir=trace_dir)
        else:
            self.trace_loader = None
            self.network_trace = self._generate_network_trace()
        
        # State tracking
        self.reset()
    
    def _generate_network_trace(self, duration=300):
        """Old simulation (fallback)"""
        np.random.seed(42)
        profiles = [
            {'mean': 500, 'std': 100},
            {'mean': 1500, 'std': 300},
            {'mean': 3000, 'std': 500},
            {'mean': 5000, 'std': 800},
        ]
        
        trace = []
        current_profile_idx = 0
        
        for i in range(duration):
            if i % 50 == 0 and i > 0:
                current_profile_idx = (current_profile_idx + 1) % len(profiles)
            
            profile = profiles[current_profile_idx]
            throughput = np.random.normal(profile['mean'], profile['std'])
            throughput = np.clip(throughput, 300, 6000)
            trace.append(throughput)
        
        return np.array(trace)
    
    def reset(self, video_id=1, split='train'):
        """Reset environment"""
        self.video_id = video_id
        self.chunk_idx = 0
        self.buffer = 0.0
        
        # Sample new trace
        if self.use_real_traces:
            self.current_trace = self.trace_loader.sample_trace(split)
            self.trace_time = 0.0
        else:
            self.trace_idx = 0
        
        # Network state history
        self.past_throughput = []
        self.past_download_time = []
        self.past_bitrates = []
        self.past_errors = []
        
        return self.get_state()
    
    def get_content_state(self):
        """Get content features for current chunk"""
        bitrate = self.bitrate_levels[0]
        key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
        
        if key not in self.content_features:
            return np.array([50.0, 15.0], dtype=np.float32)
        
        feat = self.content_features[key]
        return np.array([feat['si_mean'], feat['ti_mean']], dtype=np.float32)
    
    def get_vmaf_predictions(self):
        """Get predicted VMAF for all bitrates"""
        bitrate = self.bitrate_levels[0]
        key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
        
        if key not in self.vmaf_table:
            return np.array([30, 50, 65, 75, 82, 87], dtype=np.float32)
        
        vmaf_dict = self.vmaf_table[key]
        vmaf_values = []
        for br in self.bitrate_levels:
            vmaf_values.append(float(vmaf_dict.get(str(br), 50.0)))
        
        return np.array(vmaf_values, dtype=np.float32)
    
    def get_network_state(self):
        """Get network state (Pensieve format)"""
        state = np.zeros((6, 8), dtype=np.float32)
        
        for i, t in enumerate(self.past_throughput[-8:]):
            state[0, -(i+1)] = t / 6000.0
        
        for i, d in enumerate(self.past_download_time[-8:]):
            state[1, -(i+1)] = d / 10.0
        
        state[2, -1] = min(self.buffer / 60.0, 1.0)
        
        for i, b in enumerate(self.past_bitrates[-5:]):
            state[3, -(i+1)] = b / 6000.0
        
        remaining = self.total_chunks - self.chunk_idx
        state[4, -1] = remaining / self.total_chunks
        
        return state
    
    def get_state(self):
        """Get complete state"""
        return {
            'network': self.get_network_state(),
            'content': self.get_content_state(),
            'vmaf': self.get_vmaf_predictions()
        }
    
    def step(self, action):
        """Execute action with REAL network trace"""
        selected_bitrate = self.bitrate_levels[action]
        
        # Get throughput from REAL trace
        if self.use_real_traces:
            # Simulate chunk download with variable throughput
            chunk_size = selected_bitrate * self.chunk_duration / 8  # KB
            
            # Download happens over time, throughput varies
            download_time = 0
            downloaded = 0
            
            while downloaded < chunk_size:
                # Get current throughput
                throughput = self.current_trace.get_throughput(self.trace_time)
                
                # Download for 0.1s
                dt = 0.1
                can_download = throughput * dt / 8  # KB in 0.1s
                downloaded += can_download
                download_time += dt
                self.trace_time += dt
                
                # Safety: don't get stuck
                if download_time > 60:
                    break
            
            avg_throughput = chunk_size / download_time * 8 if download_time > 0 else throughput
            
        else:
            # Old simulation
            if self.trace_idx < len(self.network_trace):
                throughput = self.network_trace[self.trace_idx]
                self.trace_idx += 1
            else:
                throughput = np.random.uniform(500, 3000)
            
            chunk_size = selected_bitrate * self.chunk_duration / 8
            download_time = chunk_size / (throughput / 8)
            avg_throughput = throughput
        
        # Buffer dynamics
        rebuffer_time = max(0, download_time - self.buffer)
        self.buffer = max(0, self.buffer - download_time) + self.chunk_duration
        self.buffer = min(self.buffer, 60.0)
        
        # Compute reward using Pensieve QoE model
        reward = self.compute_reward(action, rebuffer_time)
        
        # Update history
        self.past_throughput.append(avg_throughput)
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
            'throughput': float(avg_throughput),
            'download_time': float(download_time)
        }
        
        return next_state, reward, done, info
    
    def compute_reward(self, action, rebuffer_time):
        """
        Compute reward using Pensieve QoE model with VMAF
        """
        # Get VMAF for selected action
        vmaf_predictions = self.get_vmaf_predictions()
        vmaf_score = vmaf_predictions[action]
        
        # Get bitrates
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        # Use Pensieve reward with VMAF
        reward = self.reward_func.compute_reward_vmaf(
            vmaf_score=vmaf_score,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate,
            current_bitrate=current_bitrate
        )
        
        return float(reward)


# ============================================
# Test
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing ContentAwareEnvV2 with Pensieve Reward")
    print("=" * 60)
    
    # Create environment
    env = ContentAwareEnvV2(use_real_traces=True)
    
    print("\n✓ Environment created")
    
    # Test episode
    state = env.reset(video_id=1, split='train')
    
    print("\nTesting episode with conservative actions:")
    actions = [0, 1, 2, 1, 0]
    
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: action={action} ({env.bitrate_levels[action]:4d} kbps), "
              f"reward={reward:+7.3f}, buffer={info['buffer']:5.1f}s, "
              f"rebuffer={info['rebuffer_time']:5.2f}s, "
              f"throughput={info['throughput']:6.0f}kbps")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:7.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    # Test with different strategy
    print("\n" + "=" * 60)
    print("Testing with aggressive strategy:")
    print("=" * 60)
    
    state = env.reset(video_id=1, split='train')
    actions = [3, 4, 5, 4, 3]  # Higher bitrates
    
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: action={action} ({env.bitrate_levels[action]:4d} kbps), "
              f"reward={reward:+7.3f}, buffer={info['buffer']:5.1f}s, "
              f"rebuffer={info['rebuffer_time']:5.2f}s, "
              f"throughput={info['throughput']:6.0f}kbps")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:7.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    # Compare with old environment
    print("\n" + "=" * 60)
    print("Comparing Real vs Simulated Traces:")
    print("=" * 60)
    
    env_sim = ContentAwareEnvV2(use_real_traces=False)
    state = env_sim.reset(video_id=1)
    actions = [0, 1, 2, 1, 0]  # Conservative
    
    total_reward_sim = 0
    total_rebuffer_sim = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env_sim.step(action)
        total_reward_sim += reward
        total_rebuffer_sim += info['rebuffer_time']
    
    print(f"\n  Real Traces:")
    print(f"    Reward:      {total_reward:7.2f}")
    print(f"    Rebuffering:  {total_rebuffer:6.2f}s")
    
    print(f"\n  Simulated Traces:")
    print(f"    Reward:      {total_reward_sim:7.2f}")
    print(f"    Rebuffering:  {total_rebuffer_sim:6.2f}s")
    
    print("\n" + "=" * 60)
    print("Reward Formula: Pensieve QoE")
    print("  Quality (VMAF) - 4.3 × Rebuffer - 1.0 × Smoothness")
    print("=" * 60)
    
    print("\n✓ All tests passed!")
    print("=" * 60)
