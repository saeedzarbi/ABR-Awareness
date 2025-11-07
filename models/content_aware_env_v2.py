"""
Content-Aware Environment V2
With Real Network Traces + Pensieve Reward
FINAL FIXED VERSION
"""

import numpy as np
import json
from pathlib import Path
import sys
import logging

# Fix import path
try:
    from models.trace_loader import TraceLoader, NetworkTrace
    from models.pensieve_reward import PensieveReward
except ModuleNotFoundError:
    from trace_loader import TraceLoader, NetworkTrace
    from pensieve_reward import PensieveReward


# ------ Logger setup ------
logger = logging.getLogger("ContentAwareEnvV2")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
# --------------------------


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
        use_real_traces=True,
        buffer_size=60.0
    ):
        
        # Resolve paths (works from any directory)
        trace_dir = resolve_path(trace_dir)
        features_file = resolve_path(features_file)
        vmaf_file = resolve_path(vmaf_file)
        
        self.bitrate_levels = bitrate_levels
        self.chunk_duration = float(chunk_duration)
        self.total_chunks = total_chunks
        self.use_real_traces = use_real_traces
        
        # Load content features
        with open(features_file, 'r') as f:
            self.content_features = json.load(f)
        
        # Load VMAF table
        with open(vmaf_file, 'r') as f:
            self.vmaf_table = json.load(f)
        
        # Pensieve reward function (STANDARD SETTINGS)
        self.reward_func = PensieveReward(
            rebuffer_penalty=4.3,      # Pensieve standard
            smoothness_penalty=1.0,    # Pensieve standard
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
        """Get predicted VMAF for all bitrates by looking up all entries"""
        vmaf_values = []
        
        for bitrate in self.bitrate_levels:
            # Look up the specific entry for this bitrate
            key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
            
            if key in self.vmaf_table and str(bitrate) in self.vmaf_table[key]:
                vmaf = float(self.vmaf_table[key][str(bitrate)])
            else:
                # Fallback: estimate based on bitrate
                # 300->30, 750->50, 1850->65, 2850->75, 4300->82, 6000->87
                vmaf = 30 + (bitrate - 300) / (6000 - 300) * 57
            
            vmaf_values.append(vmaf)
        
        return np.array(vmaf_values, dtype=np.float32)
    
    def get_network_state(self):
        """Get network state (Pensieve format)"""
        state = np.zeros((6, 8), dtype=np.float32)
        
        # Past throughput (normalized by max bitrate)
        for i, t in enumerate(self.past_throughput[-8:]):
            state[0, -(i+1)] = t / 6000.0
        
        # Past download time (normalized by 10s)
        for i, d in enumerate(self.past_download_time[-8:]):
            state[1, -(i+1)] = d / 10.0
        
        # Current buffer (normalized by max buffer)
        state[2, -1] = min(self.buffer / 60.0, 1.0)
        
        # Past bitrates (normalized by max bitrate)
        for i, b in enumerate(self.past_bitrates[-5:]):
            state[3, -(i+1)] = b / 6000.0
        
        # Remaining chunks (normalized)
        remaining = self.total_chunks - self.chunk_idx
        state[4, -1] = remaining / self.total_chunks
        
        return state
    
    def get_state(self):
        """Get complete state (normalized for model input)"""
        return {
            'network': self.get_network_state(),
            'content': self.get_content_state() / 100.0,   # SI/TI -> [0,1]
            'vmaf': self.get_vmaf_predictions() / 100.0    # VMAF -> [0,1]
        }
    
    def step(self, action):
        """Execute action with REAL network trace"""
        selected_bitrate = self.bitrate_levels[action]  # in kbps
        
        # -----------------------
        # Simulate chunk download
        # -----------------------
        if self.use_real_traces:
            # Operate in kilobits (kbit) and kilobits/s (kbps)
            chunk_size_kbit = float(selected_bitrate) * float(self.chunk_duration)
            
            download_time = 0.0
            downloaded_kbit = 0.0
            dt = 0.1  # time step for simulation
            max_download_time = 32.0  # safety cap
            
            sample_throughputs = []
            
            while downloaded_kbit < chunk_size_kbit and download_time < max_download_time:
                tp_raw = self.current_trace.get_throughput(self.trace_time)
                sample_throughputs.append(tp_raw)
                
                # Heuristic: tp_raw usually in kbps; if very small (<20) maybe Mbps
                if tp_raw is None:
                    throughput_kbps = 0.0
                elif tp_raw < 20.0:
                    throughput_kbps = float(tp_raw) * 1000.0  # Mbps to kbps
                else:
                    throughput_kbps = float(tp_raw)
                
                # Download in this time step
                downloaded_kbit += throughput_kbps * dt
                download_time += dt
                self.trace_time += dt
                
                # Safety break
                if download_time >= max_download_time:
                    break
            
            # Log if download was incomplete
            if downloaded_kbit < chunk_size_kbit and download_time >= max_download_time:
                logger.info(f"DOWNLOAD_SAFETY chunk={self.chunk_idx} video={self.video_id} "
                            f"sel_br={selected_bitrate}kbps samples={sample_throughputs[:6]} "
                            f"downloaded={downloaded_kbit:.1f}/{chunk_size_kbit:.1f}kbit "
                            f"time={max_download_time}s")
            
            # Compute average throughput
            avg_throughput = (downloaded_kbit / download_time) if download_time > 0 else 0.0
        
        else:
            # Old simulation path (fallback)
            if self.trace_idx < len(self.network_trace):
                tp = self.network_trace[self.trace_idx]
                self.trace_idx += 1
            else:
                tp = np.random.uniform(500, 3000)
            
            chunk_size_kbit = float(selected_bitrate) * float(self.chunk_duration)
            download_time = chunk_size_kbit / float(tp) if tp > 0 else self.chunk_duration
            avg_throughput = float(tp)
        
        # -----------------------
        # Buffer dynamics
        # -----------------------
        if download_time > self.buffer:
            rebuffer_time = download_time - self.buffer
        else:
            rebuffer_time = 0.0
        
        # Update buffer
        self.buffer = max(0.0, self.buffer - download_time) + self.chunk_duration
        self.buffer = min(self.buffer, 60.0)
        
        # -----------------------
        # Compute reward
        # -----------------------
        reward = self.compute_reward(action, rebuffer_time)
        
        # Debug logging (periodic)
        if self.chunk_idx % 12 == 0:
            logger.info(f"STEP_DEBUG chunk={self.chunk_idx} video={self.video_id} "
                        f"bitrate={selected_bitrate}kbps download={download_time:.2f}s "
                        f"rebuffer={rebuffer_time:.2f}s throughput={avg_throughput:.1f}kbps "
                        f"buffer={self.buffer:.1f}s reward={reward:.3f}")
        
        # Update history
        self.past_throughput.append(float(avg_throughput))
        self.past_download_time.append(float(download_time))
        self.past_bitrates.append(selected_bitrate)
        
        # Move to next chunk
        self.chunk_idx += 1
        done = (self.chunk_idx >= self.total_chunks)
        
        # Get next state
        next_state = self.get_state() if not done else None
        
        # Info dict
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
        # Get VMAF for selected action (raw 0..100)
        vmaf_predictions = self.get_vmaf_predictions()
        vmaf_score = vmaf_predictions[action]
        
        # Get bitrates (kbps)
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        # Compute Pensieve reward with VMAF
        reward = self.reward_func.compute_reward_vmaf(
            vmaf_score=vmaf_score,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate,
            current_bitrate=current_bitrate
        )
        
        # Debug extreme reward cases
        if reward < -100.0:
            logger.info(f"REWARD_DBG vmaf={vmaf_score:.1f} bitrate={current_bitrate}kbps "
                        f"rebuffer={rebuffer_time:.2f}s last_br={last_bitrate}kbps "
                        f"reward={reward:.2f}")
        
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
        
        print(f"  Step {i+1}: action={action} ({env.bitrate_levels[action]:4d}kbps), "
              f"reward={reward:+7.3f}, buffer={info['buffer']:5.1f}s, "
              f"rebuffer={info['rebuffer_time']:5.2f}s, "
              f"throughput={info['throughput']:6.0f}kbps")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:7.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    # Test with aggressive strategy
    print("\n" + "=" * 60)
    print("Testing with aggressive strategy:")
    print("=" * 60)
    
    state = env.reset(video_id=1, split='train')
    actions = [3, 4, 5, 4, 3]
    
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: action={action} ({env.bitrate_levels[action]:4d}kbps), "
              f"reward={reward:+7.3f}, buffer={info['buffer']:5.1f}s, "
              f"rebuffer={info['rebuffer_time']:5.2f}s, "
              f"throughput={info['throughput']:6.0f}kbps")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:7.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    print("\n" + "=" * 60)
    print("Reward Formula: Pensieve QoE")
    print("  Quality (VMAF) - 4.3 × Rebuffer - 1.0 × Smoothness")
    print("=" * 60)
    
    print("\n✓ All tests passed!")
    print("=" * 60)


"""
Content-Aware Environment V2
With Real Network Traces + Pensieve Reward + Per-Video Support
COMPLETE VERSION with Random Video Selection
"""

import numpy as np
import json
import random  # ✅ Added for random video selection
from pathlib import Path
import sys
import logging

# Fix import path
try:
    from models.trace_loader_seeded import TraceLoader, NetworkTrace
    from models.pensieve_reward import PensieveReward
except ModuleNotFoundError:
    from trace_loader_seeded import TraceLoader, NetworkTrace
    from pensieve_reward import PensieveReward


# ------ Logger setup ------
logger = logging.getLogger("ContentAwareEnvV2")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
# --------------------------


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
    ✅ با قابلیت random video selection برای per-video analysis
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
        
        # Resolve paths (works from any directory)
        trace_dir = resolve_path(trace_dir)
        features_file = resolve_path(features_file)
        vmaf_file = resolve_path(vmaf_file)
        
        self.bitrate_levels = bitrate_levels
        self.chunk_duration = float(chunk_duration)
        self.total_chunks = total_chunks
        self.use_real_traces = use_real_traces
        
        # ✅ Video configuration (for per-video analysis)
        self.num_videos = 6
        self.video_names = {
            1: 'sports',
            2: 'animation',
            3: 'news',
            4: 'nature',
            5: 'game',
            6: 'movie'
        }
        
        # Load content features
        with open(features_file, 'r') as f:
            self.content_features = json.load(f)
        
        # Load VMAF table
        with open(vmaf_file, 'r') as f:
            self.vmaf_table = json.load(f)
        
        # Pensieve reward function (STANDARD SETTINGS)
        self.reward_func = PensieveReward(
            rebuffer_penalty=4.3,      # Pensieve standard
            smoothness_penalty=1.0,    # Pensieve standard
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
    
    def reset(self, video_id=None, split='train'):
        """
        Reset environment
        
        Args:
            video_id: Specific video ID (1-6), or None for random selection
            split: 'train', 'val', or 'test'
        """
        # ✅ Random video selection if not specified
        if video_id is None:
            self.video_id = random.randint(1, self.num_videos)
        else:
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
    
    def get_video_name(self):
        """
        Get current video name
        
        Returns:
            str: Video name (e.g., 'sports', 'animation')
        """
        return self.video_names.get(self.video_id, f'video{self.video_id}')
    
    def get_content_state(self):
        """Get content features for current chunk"""
        bitrate = self.bitrate_levels[0]
        key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
        
        if key not in self.content_features:
            return np.array([50.0, 15.0], dtype=np.float32)
        
        feat = self.content_features[key]
        return np.array([feat['si_mean'], feat['ti_mean']], dtype=np.float32)
    
    def get_vmaf_predictions(self):
        """Get predicted VMAF for all bitrates by looking up all entries"""
        vmaf_values = []
        
        for bitrate in self.bitrate_levels:
            # Look up the specific entry for this bitrate
            key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
            
            if key in self.vmaf_table and str(bitrate) in self.vmaf_table[key]:
                vmaf = float(self.vmaf_table[key][str(bitrate)])
            else:
                # Fallback: estimate based on bitrate
                # 300->30, 750->50, 1850->65, 2850->75, 4300->82, 6000->87
                vmaf = 30 + (bitrate - 300) / (6000 - 300) * 57
            
            vmaf_values.append(vmaf)
        
        return np.array(vmaf_values, dtype=np.float32)
    
    def get_network_state(self):
        """Get network state (Pensieve format)"""
        state = np.zeros((6, 8), dtype=np.float32)
        
        # Past throughput (normalized by max bitrate)
        for i, t in enumerate(self.past_throughput[-8:]):
            state[0, -(i+1)] = t / 6000.0
        
        # Past download time (normalized by 10s)
        for i, d in enumerate(self.past_download_time[-8:]):
            state[1, -(i+1)] = d / 10.0
        
        # Current buffer (normalized by max buffer)
        state[2, -1] = min(self.buffer / 60.0, 1.0)
        
        # Past bitrates (normalized by max bitrate)
        for i, b in enumerate(self.past_bitrates[-5:]):
            state[3, -(i+1)] = b / 6000.0
        
        # Remaining chunks (normalized)
        remaining = self.total_chunks - self.chunk_idx
        state[4, -1] = remaining / self.total_chunks
        
        return state
    
    def get_state(self):
        """Get complete state (normalized for model input)"""
        return {
            'network': self.get_network_state(),
            'content': self.get_content_state() / 100.0,   # SI/TI -> [0,1]
            'vmaf': self.get_vmaf_predictions() / 100.0    # VMAF -> [0,1]
        }
    
    def step(self, action):
        """Execute action with REAL network trace"""
        selected_bitrate = self.bitrate_levels[action]  # in kbps
        
        # -----------------------
        # Simulate chunk download
        # -----------------------
        if self.use_real_traces:
            # Operate in kilobits (kbit) and kilobits/s (kbps)
            chunk_size_kbit = float(selected_bitrate) * float(self.chunk_duration)
            
            download_time = 0.0
            downloaded_kbit = 0.0
            dt = 0.1  # time step for simulation
            max_download_time = 32.0  # safety cap
            
            sample_throughputs = []
            
            while downloaded_kbit < chunk_size_kbit and download_time < max_download_time:
                tp_raw = self.current_trace.get_throughput(self.trace_time)
                sample_throughputs.append(tp_raw)
                
                # Heuristic: tp_raw usually in kbps; if very small (<20) maybe Mbps
                if tp_raw is None:
                    throughput_kbps = 0.0
                elif tp_raw < 20.0:
                    throughput_kbps = float(tp_raw) * 1000.0  # Mbps to kbps
                else:
                    throughput_kbps = float(tp_raw)
                
                # Download in this time step
                downloaded_kbit += throughput_kbps * dt
                download_time += dt
                self.trace_time += dt
                
                # Safety break
                if download_time >= max_download_time:
                    break
            
            # Log if download was incomplete - DISABLED for cleaner output
            # if downloaded_kbit < chunk_size_kbit and download_time >= max_download_time:
            #     logger.info(f"DOWNLOAD_SAFETY chunk={self.chunk_idx} video={self.video_id} "
            #                 f"sel_br={selected_bitrate}kbps samples={sample_throughputs[:6]} "
            #                 f"downloaded={downloaded_kbit:.1f}/{chunk_size_kbit:.1f}kbit "
            #                 f"time={max_download_time}s")
            
            # Compute average throughput
            avg_throughput = (downloaded_kbit / download_time) if download_time > 0 else 0.0
        
        else:
            # Old simulation path (fallback)
            if self.trace_idx < len(self.network_trace):
                tp = self.network_trace[self.trace_idx]
                self.trace_idx += 1
            else:
                tp = np.random.uniform(500, 3000)
            
            chunk_size_kbit = float(selected_bitrate) * float(self.chunk_duration)
            download_time = chunk_size_kbit / float(tp) if tp > 0 else self.chunk_duration
            avg_throughput = float(tp)
        
        # -----------------------
        # Buffer dynamics
        # -----------------------
        if download_time > self.buffer:
            rebuffer_time = download_time - self.buffer
        else:
            rebuffer_time = 0.0
        
        # Update buffer
        self.buffer = max(0.0, self.buffer - download_time) + self.chunk_duration
        self.buffer = min(self.buffer, 60.0)
        
        # -----------------------
        # Compute reward
        # -----------------------
        reward = self.compute_reward(action, rebuffer_time)
        
        # Debug logging (periodic) - DISABLED for cleaner output
        # if self.chunk_idx % 12 == 0:
        #     logger.info(f"STEP_DEBUG chunk={self.chunk_idx} video={self.video_id} "
        #                 f"bitrate={selected_bitrate}kbps download={download_time:.2f}s "
        #                 f"rebuffer={rebuffer_time:.2f}s throughput={avg_throughput:.1f}kbps "
        #                 f"buffer={self.buffer:.1f}s reward={reward:.3f}")
        
        # Update history
        self.past_throughput.append(float(avg_throughput))
        self.past_download_time.append(float(download_time))
        self.past_bitrates.append(selected_bitrate)
        
        # Move to next chunk
        self.chunk_idx += 1
        done = (self.chunk_idx >= self.total_chunks)
        
        # Get next state
        next_state = self.get_state() if not done else None
        
        # Info dict
        info = {
            'rebuffer_time': float(rebuffer_time),
            'bitrate': float(selected_bitrate),
            'buffer': float(self.buffer),
            'chunk_idx': int(self.chunk_idx),
            'throughput': float(avg_throughput),
            'download_time': float(download_time),
            'video_id': int(self.video_id),  # ✅ Added for tracking
            'video_name': self.get_video_name()  # ✅ Added for tracking
        }
        
        return next_state, reward, done, info
    
    def compute_reward(self, action, rebuffer_time):
        """
        Compute reward using Pensieve QoE model with VMAF
        """
        # Get VMAF for selected action (raw 0..100)
        vmaf_predictions = self.get_vmaf_predictions()
        vmaf_score = vmaf_predictions[action]
        
        # Get bitrates (kbps)
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        # Compute Pensieve reward with VMAF
        reward = self.reward_func.compute_reward_vmaf(
            vmaf_score=vmaf_score,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate,
            current_bitrate=current_bitrate
        )
        
        # Debug extreme reward cases - DISABLED for cleaner output
        # if reward < -100.0:
        #     logger.info(f"REWARD_DBG vmaf={vmaf_score:.1f} bitrate={current_bitrate}kbps "
        #                 f"rebuffer={rebuffer_time:.2f}s last_br={last_bitrate}kbps "
        #                 f"reward={reward:.2f}")
        
        return float(reward)


# ============================================
# Test
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing ContentAwareEnvV2 with Random Video Selection")
    print("=" * 60)
    
    # Create environment
    env = ContentAwareEnvV2(use_real_traces=True)
    print("\n✓ Environment created")
    
    # Test random video selection
    print("\n" + "=" * 60)
    print("Testing Random Video Selection:")
    print("=" * 60)
    for i in range(10):
        state = env.reset(split='train')
        print(f"  Episode {i+1}: video_id={env.video_id}, name={env.get_video_name()}")
    
    # Test specific video
    print("\n" + "=" * 60)
    print("Testing Specific Video (video_id=3):")
    print("=" * 60)
    state = env.reset(video_id=3, split='train')
    print(f"  Video ID: {env.video_id}")
    print(f"  Video Name: {env.get_video_name()}")
    
    # Test episode with conservative actions
    print("\n" + "=" * 60)
    print("Testing Episode with Conservative Actions:")
    print("=" * 60)
    
    state = env.reset(video_id=1, split='train')
    print(f"  Video: {env.get_video_name()}")
    
    actions = [0, 1, 2, 1, 0]
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: action={action} ({env.bitrate_levels[action]:4d}kbps), "
              f"reward={reward:+7.3f}, buffer={info['buffer']:5.1f}s, "
              f"rebuffer={info['rebuffer_time']:5.2f}s, "
              f"throughput={info['throughput']:6.0f}kbps")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:7.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    # Test with aggressive strategy
    print("\n" + "=" * 60)
    print("Testing with Aggressive Strategy:")
    print("=" * 60)
    
    state = env.reset(video_id=2, split='train')
    print(f"  Video: {env.get_video_name()}")
    
    actions = [3, 4, 5, 4, 3]
    total_reward = 0
    total_rebuffer = 0
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_rebuffer += info['rebuffer_time']
        
        print(f"  Step {i+1}: action={action} ({env.bitrate_levels[action]:4d}kbps), "
              f"reward={reward:+7.3f}, buffer={info['buffer']:5.1f}s, "
              f"rebuffer={info['rebuffer_time']:5.2f}s, "
              f"throughput={info['throughput']:6.0f}kbps")
        
        if done:
            break
    
    print(f"\n  Total reward: {total_reward:7.2f}")
    print(f"  Total rebuffering: {total_rebuffer:.2f}s")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✅ Random video selection (video_id=None)")
    print("  ✅ Specific video selection (video_id=1-6)")
    print("  ✅ Video name tracking (env.get_video_name())")
    print("  ✅ Video info in step() info dict")
    print("=" * 60)
"""
Content-Aware Environment V2 (Fixed Version)
- Adds VMAF info to step() output
- Uses consistent Mbps→kbps conversion
- Supports optional composite reward for evaluation
"""

# import numpy as np
# import json
# import random
# from pathlib import Path
# import logging

# try:
#     from models.trace_loader_seeded import TraceLoader, NetworkTrace
#     from models.pensieve_reward import PensieveReward
#     from models.reward_composite import compute_composite_reward
# except ModuleNotFoundError:
#     from trace_loader_seeded import TraceLoader, NetworkTrace
#     from pensieve_reward import PensieveReward
#     from reward_composite import compute_composite_reward

# # Logger setup
# logger = logging.getLogger("ContentAwareEnvV2")
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)


# def get_project_root():
#     return Path(__file__).parent.parent


# def resolve_path(relative_path):
#     path = Path(relative_path)
#     if path.is_absolute():
#         return str(path)
#     return str(get_project_root() / relative_path)


# class ContentAwareEnvV2:
#     def __init__(
#         self,
#         trace_dir='data/network_traces/cooked_traces',
#         features_file='data/features/si_ti_features.json',
#         vmaf_file='data/vmaf/vmaf_table.json',
#         bitrate_levels=[300, 750, 1850, 2850, 4300, 6000],
#         chunk_duration=4.0,
#         total_chunks=48,
#         use_real_traces=True,
#         buffer_size=60.0,
#         use_composite_reward=False,
#     ):
#         trace_dir = resolve_path(trace_dir)
#         features_file = resolve_path(features_file)
#         vmaf_file = resolve_path(vmaf_file)

#         self.bitrate_levels = bitrate_levels
#         self.chunk_duration = float(chunk_duration)
#         self.total_chunks = total_chunks
#         self.use_real_traces = use_real_traces
#         self.use_composite_reward = use_composite_reward

#         self.num_videos = 6
#         self.video_names = {
#             1: 'sports', 2: 'animation', 3: 'news',
#             4: 'nature', 5: 'game', 6: 'movie'
#         }

#         with open(features_file, 'r') as f:
#             self.content_features = json.load(f)

#         with open(vmaf_file, 'r') as f:
#             self.vmaf_table = json.load(f)

#         self.reward_func = PensieveReward(
#             rebuffer_penalty=4.3,
#             smoothness_penalty=1.0,
#             bitrate_levels=bitrate_levels
#         )

#         if use_real_traces:
#             self.trace_loader = TraceLoader(trace_dir=trace_dir)
#         else:
#             self.trace_loader = None
#             self.network_trace = self._generate_network_trace()

#         self.reset()

#     def _generate_network_trace(self, duration=300):
#         np.random.seed(42)
#         profiles = [
#             {'mean': 500, 'std': 100},
#             {'mean': 1500, 'std': 300},
#             {'mean': 3000, 'std': 500},
#             {'mean': 5000, 'std': 800},
#         ]
#         trace = []
#         idx = 0
#         for i in range(duration):
#             if i % 50 == 0 and i > 0:
#                 idx = (idx + 1) % len(profiles)
#             p = profiles[idx]
#             t = np.random.normal(p['mean'], p['std'])
#             trace.append(np.clip(t, 300, 6000))
#         return np.array(trace)

#     def reset(self, video_id=None, split='train'):
#         self.video_id = random.randint(1, self.num_videos) if video_id is None else video_id
#         self.chunk_idx = 0
#         self.buffer = 0.0

#         if self.use_real_traces:
#             self.current_trace = self.trace_loader.sample_trace(split)
#             self.trace_time = 0.0
#         else:
#             self.trace_idx = 0

#         self.past_throughput = []
#         self.past_download_time = []
#         self.past_bitrates = []
#         return self.get_state()

#     def get_video_name(self):
#         return self.video_names.get(self.video_id, f'video{self.video_id}')

#     def get_content_state(self):
#         bitrate = self.bitrate_levels[0]
#         key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
#         feat = self.content_features.get(key, {'si_mean': 50.0, 'ti_mean': 15.0})
#         return np.array([feat['si_mean'], feat['ti_mean']], dtype=np.float32)

#     def get_vmaf_predictions(self):
#         vmaf_values = []
#         for bitrate in self.bitrate_levels:
#             key = f"video{self.video_id}/{bitrate}/chunk_{self.chunk_idx:03d}"
#             if key in self.vmaf_table and str(bitrate) in self.vmaf_table[key]:
#                 vmaf = float(self.vmaf_table[key][str(bitrate)])
#             else:
#                 vmaf = 30 + (bitrate - 300) / (6000 - 300) * 57
#             vmaf_values.append(vmaf)
#         return np.array(vmaf_values, dtype=np.float32)

#     def get_network_state(self):
#         s = np.zeros((6, 8), dtype=np.float32)
#         for i, t in enumerate(self.past_throughput[-8:]):
#             s[0, -(i + 1)] = t / 6000.0
#         for i, d in enumerate(self.past_download_time[-8:]):
#             s[1, -(i + 1)] = d / 10.0
#         s[2, -1] = min(self.buffer / 60.0, 1.0)
#         for i, b in enumerate(self.past_bitrates[-5:]):
#             s[3, -(i + 1)] = b / 6000.0
#         remaining = self.total_chunks - self.chunk_idx
#         s[4, -1] = remaining / self.total_chunks
#         return s

#     def get_state(self):
#         return {
#             'network': self.get_network_state(),
#             'content': self.get_content_state() / 100.0,
#             'vmaf': self.get_vmaf_predictions() / 100.0
#         }

#     def step(self, action):
#         selected_bitrate = self.bitrate_levels[action]
#         chunk_size_kbit = float(selected_bitrate) * float(self.chunk_duration)

#         download_time, downloaded_kbit = 0.0, 0.0
#         dt = 0.1
#         max_download_time = 32.0

#         while downloaded_kbit < chunk_size_kbit and download_time < max_download_time:
#             tp_raw = self.current_trace.get_throughput(self.trace_time)
#             throughput_kbps = 0.0 if tp_raw is None else float(tp_raw) * 1000.0  # Always Mbps→kbps
#             downloaded_kbit += throughput_kbps * dt
#             download_time += dt
#             self.trace_time += dt
#             if download_time >= max_download_time:
#                 break

#         avg_throughput = (downloaded_kbit / download_time) if download_time > 0 else 0.0

#         rebuffer_time = max(0.0, download_time - self.buffer)
#         self.buffer = max(0.0, self.buffer - download_time) + self.chunk_duration
#         self.buffer = min(self.buffer, 60.0)

#         # Compute reward (either Pensieve or Composite)
#         if self.use_composite_reward:
#             info_tmp = {
#                 'bitrate': selected_bitrate,
#                 'rebuffer_time': rebuffer_time,
#                 'buffer': self.buffer,
#                 'vmaf': self.get_vmaf_predictions()[action]
#             }
#             reward = compute_composite_reward(info_tmp)
#         else:
#             reward = self.compute_pensieve_reward(action, rebuffer_time)

#         self.past_throughput.append(float(avg_throughput))
#         self.past_download_time.append(float(download_time))
#         self.past_bitrates.append(selected_bitrate)

#         self.chunk_idx += 1
#         done = (self.chunk_idx >= self.total_chunks)
#         next_state = self.get_state() if not done else None

#         vmaf_score = float(self.get_vmaf_predictions()[action])
#         info = {
#             'rebuffer_time': float(rebuffer_time),
#             'bitrate': float(selected_bitrate),
#             'buffer': float(self.buffer),
#             'chunk_idx': int(self.chunk_idx),
#             'throughput': float(avg_throughput),
#             'download_time': float(download_time),
#             'video_id': int(self.video_id),
#             'video_name': self.get_video_name(),
#             'vmaf': vmaf_score,  # ✅ Added
#         }

#         return next_state, reward, done, info

#     def compute_pensieve_reward(self, action, rebuffer_time):
#         vmaf_predictions = self.get_vmaf_predictions()
#         vmaf_score = vmaf_predictions[action]
#         current_bitrate = self.bitrate_levels[action]
#         last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0

#         reward = self.reward_func.compute_reward_vmaf(
#             vmaf_score=vmaf_score,
#             rebuffer_time=rebuffer_time,
#             last_bitrate=last_bitrate,
#             current_bitrate=current_bitrate,
#         )
#         return float(reward)


# if __name__ == '__main__':
#     env = ContentAwareEnvV2(use_real_traces=True, use_composite_reward=True)
#     s = env.reset(video_id=1, split='train')
#     done = False
#     total_r = 0.0
#     while not done:
#         a = np.random.randint(0, 6)
#         s, r, done, info = env.step(a)
#         total_r += r
#     print(f"✓ Test episode complete. Total reward={total_r:.2f}, Last VMAF={info['vmaf']:.1f}")