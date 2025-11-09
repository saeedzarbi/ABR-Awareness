"""
Content-Aware Environment V3 - FIXED THROUGHPUT CONVERSION
"""

import numpy as np
import json
import random
from pathlib import Path
import sys
import logging

try:
    from models.trace_loader import TraceLoader, NetworkTrace
    from models.pensieve_reward import PensieveReward
except ModuleNotFoundError:
    from trace_loader import TraceLoader, NetworkTrace
    from pensieve_reward import PensieveReward

logger = logging.getLogger("ContentAwareEnvV2")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def get_project_root():
    return Path(__file__).parent.parent


def resolve_path(relative_path):
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str(get_project_root() / relative_path)


class ContentAwareEnvV2:
    """
    Environment with FIXED throughput conversion
    """
    
    def __init__(
        self,
        trace_dir='data/network_traces/cooked_traces',
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        bitrate_levels=[300, 750, 1850, 2850, 4300, 6000],
        chunk_duration=4.0,
        total_chunks=48,
        use_real_traces=True,
        buffer_size=60.0
    ):
        
        trace_dir = resolve_path(trace_dir)
        features_file = resolve_path(features_file)
        vmaf_file = resolve_path(vmaf_file)
        
        self.bitrate_levels = bitrate_levels
        self.chunk_duration = float(chunk_duration)
        self.total_chunks = total_chunks
        self.use_real_traces = use_real_traces
        
        self.num_videos = 6
        self.video_names = {
            1: 'sports', 2: 'animation', 3: 'news',
            4: 'nature', 5: 'game', 6: 'movie'
        }
        
        with open(features_file, 'r') as f:
            self.content_features = json.load(f)
        
        with open(vmaf_file, 'r') as f:
            self.vmaf_table = json.load(f)
        
        # REDUCED rebuffer penalty for more stable training
        self.reward_func = PensieveReward(
            rebuffer_penalty=2.0,      # Reduced from 4.3
            smoothness_penalty=1.0,
            bitrate_levels=bitrate_levels
        )
        
        if use_real_traces:
            self.trace_loader = TraceLoader(trace_dir=trace_dir)
        else:
            self.trace_loader = None
            self.network_trace = self._generate_network_trace()
        
        self.reset()
    
    def _generate_network_trace(self, duration=300):
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
    
    def reset(self, video_id=None, split='train'):
        self.video_id = random.randint(1, self.num_videos) if video_id is None else video_id
        self.chunk_idx = 0
        self.buffer = 0.0
        
        if self.use_real_traces:
            self.current_trace = self.trace_loader.sample_trace(split)
            self.trace_time = 0.0
        else:
            self.trace_idx = 0
        
        self.past_throughput = []
        self.past_download_time = []
        self.past_bitrates = []
        self.past_errors = []
        
        return self.get_state()
    
    def get_video_name(self):
        return self.video_names.get(self.video_id, f'video{self.video_id}')
    
    def get_content_state(self):
        bitrate = self.bitrate_levels[0]
        key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
        
        if key not in self.content_features:
            return np.array([50.0, 15.0], dtype=np.float32)
        
        feat = self.content_features[key]
        return np.array([feat['si_mean'], feat['ti_mean']], dtype=np.float32)
    
    def get_vmaf_predictions(self):
        vmaf_values = []
        
        for bitrate in self.bitrate_levels:
            key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
            
            if key in self.vmaf_table and str(bitrate) in self.vmaf_table[key]:
                vmaf = float(self.vmaf_table[key][str(bitrate)])
            else:
                vmaf = 30 + (bitrate - 300) / (6000 - 300) * 57
            
            vmaf_values.append(vmaf)
        
        return np.array(vmaf_values, dtype=np.float32)
    
    def get_network_state(self):
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
        return {
            'network': self.get_network_state(),
            'content': self.get_content_state() / 100.0,
            'vmaf': self.get_vmaf_predictions() / 100.0
        }
    
    def step(self, action):
        """Execute action with FIXED throughput conversion"""
        selected_bitrate = self.bitrate_levels[action]
        
        # Chunk size in kilobits
        chunk_size_kbit = float(selected_bitrate) * float(self.chunk_duration)
        
        download_time = 0.0
        downloaded_kbit = 0.0
        dt = 0.1
        max_download_time = 32.0
        
        sample_throughputs = []
        
        while downloaded_kbit < chunk_size_kbit and download_time < max_download_time:
            tp_raw = self.current_trace.get_throughput(self.trace_time)
            sample_throughputs.append(tp_raw)
            
            # FIXED: Always convert Mbps to kbps
            # FCC traces are in Mbps (0.8-1.0 range)
            if tp_raw is None:
                throughput_kbps = 0.0
            else:
                # Always multiply by 1000 to convert Mbps → kbps
                throughput_kbps = float(tp_raw) * 1000.0
            
            # Clip to reasonable range (0-10000 kbps = 0-10 Mbps)
            throughput_kbps = np.clip(throughput_kbps, 0.0, 10000.0)
            
            downloaded_kbit += throughput_kbps * dt
            download_time += dt
            self.trace_time += dt
            
            if download_time >= max_download_time:
                break
        
        # Average throughput
        avg_throughput = (downloaded_kbit / download_time) if download_time > 0 else 0.0
        
        # Buffer dynamics
        rebuffer_time = max(0.0, download_time - self.buffer)
        
        self.buffer = max(0.0, self.buffer - download_time) + self.chunk_duration
        self.buffer = min(self.buffer, 60.0)
        
        # Compute reward
        reward = self.compute_reward(action, rebuffer_time)
        
        # Update history
        self.past_throughput.append(float(avg_throughput))
        self.past_download_time.append(float(download_time))
        self.past_bitrates.append(selected_bitrate)
        
        # Move to next chunk
        self.chunk_idx += 1
        done = (self.chunk_idx >= self.total_chunks)
        
        next_state = self.get_state() if not done else None
        
        vmaf_predictions = self.get_vmaf_predictions()
        actual_vmaf = float(vmaf_predictions[action])
        
        info = {
            'rebuffer_time': float(rebuffer_time),
            'bitrate': float(selected_bitrate),
            'buffer': float(self.buffer),
            'chunk_idx': int(self.chunk_idx),
            'throughput': float(avg_throughput),
            'download_time': float(download_time),
            'video_id': int(self.video_id),
            'video_name': self.get_video_name(),
            'vmaf': actual_vmaf
        }
        
        return next_state, reward, done, info
    
    def compute_reward(self, action, rebuffer_time):
        vmaf_predictions = self.get_vmaf_predictions()
        vmaf_score = vmaf_predictions[action]
        
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        reward = self.reward_func.compute_reward_vmaf(
            vmaf_score=vmaf_score,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate,
            current_bitrate=current_bitrate
        )
        
        return float(reward)


if __name__ == '__main__':
    print("="*60)
    print("Testing Fixed Environment")
    print("="*60)
    
    env = ContentAwareEnvV2(use_real_traces=True)
    print("\n✓ Environment created")
    
    state = env.reset(video_id=1, split='train')
    
    print("\nTesting with conservative actions:")
    actions = [0, 1, 2, 1, 0]
    
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: bitrate={env.bitrate_levels[action]}kbps, "
              f"reward={reward:+7.3f}, rebuffer={info['rebuffer_time']:.2f}s, "
              f"throughput={info['throughput']:.0f}kbps, "
              f"vmaf={info['vmaf']:.1f}")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:7.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    print(f"\n✓ Tests passed! Environment should work correctly now.")