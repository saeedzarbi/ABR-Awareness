"""
ABR (Adaptive Bitrate) Gym Environment for Deep Reinforcement Learning.
Content-aware environment with VMAF, SI/TI features.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


class ABREnv(gym.Env):
    """
    Custom Gym Environment for ABR streaming simulation.
    
    State Space:
        - Throughput history: last 8 chunks (normalized)
        - Buffer level: current buffer occupancy (seconds)
        - Last bitrate: previous quality choice
        - SI/TI: content complexity features
        - VMAF predictions: quality for each bitrate level
    
    Action Space:
        - Discrete(6): Select bitrate level (0-5)
    
    Reward:
        QoE = quality - rebuffering_penalty - smoothness_penalty
    """
    
    metadata = {'render_modes': ['human']}
    
    # Bitrate levels (Kbps)
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    
    # Chunk duration (seconds)
    CHUNK_DURATION = 4.0
    
    # Buffer parameters
    BUFFER_TARGET = 15.0  # Target buffer level (seconds)
    BUFFER_MAX = 30.0     # Maximum buffer size (seconds)
    
    # Reward weights
    REBUFFER_PENALTY = 4.3  # Penalty per second of rebuffering
    SMOOTH_PENALTY = 1.0    # Penalty for bitrate switches
    
    def __init__(
        self,
        video_name: str = 'sample1',
        trace_dir: str = 'data/network_traces/processed',
        vmaf_dir: str = 'data/vmaf_scores',
        siti_dir: str = 'data/content_features',
        max_chunks: int = 48,  # 48 chunks * 4s = 192s video
        random_seed: Optional[int] = None
    ):
        super().__init__()
        
        self.video_name = video_name
        self.trace_dir = Path(trace_dir)
        self.vmaf_dir = Path(vmaf_dir)
        self.siti_dir = Path(siti_dir)
        self.max_chunks = max_chunks
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Load data
        self._load_traces()
        self._load_vmaf_scores()
        self._load_siti_features()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.BITRATE_LEVELS))
        
        # Observation space dimensions
        obs_dim = (
            8 +   # Throughput history (8 past chunks)
            1 +   # Buffer level
            1 +   # Last bitrate index
            2 +   # SI, TI
            6     # VMAF predictions for 6 bitrate levels
        )  # Total: 18 dimensions
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_trace = None
        self.current_trace_idx = 0
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.throughput_history = []
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
    
    def _load_traces(self):
        """Load all network traces."""
        trace_files = list(self.trace_dir.glob("*.json"))
        
        if not trace_files:
            raise ValueError(f"No traces found in {self.trace_dir}")
        
        self.traces = []
        for trace_file in trace_files:
            with open(trace_file, 'r') as f:
                trace = json.load(f)
                self.traces.append(trace)
        
        print(f"✓ Loaded {len(self.traces)} network traces")
    
    def _load_vmaf_scores(self):
        """Load VMAF scores for the video."""
        vmaf_file = self.vmaf_dir / f"{self.video_name}" / "vmaf_summary.csv"
        
        # Try alternative path
        if not vmaf_file.exists():
            vmaf_file = self.vmaf_dir / "vmaf_summary.csv"
        
        if not vmaf_file.exists():
            # Use default VMAF scores if file not found
            print(f"⚠ VMAF file not found, using defaults")
            self.vmaf_scores = {
                300: 40.0, 750: 65.0, 1200: 78.0,
                1850: 85.0, 2850: 90.0, 6000: 95.0
            }
        else:
            import pandas as pd
            df = pd.read_csv(vmaf_file)
            df = df[df['video'] == self.video_name]
            self.vmaf_scores = dict(zip(df['bitrate_kbps'], df['vmaf']))
        
        # Normalize VMAF scores to [0, 1]
        self.vmaf_normalized = {
            br: vmaf / 100.0 for br, vmaf in self.vmaf_scores.items()
        }
        
        print(f"✓ Loaded VMAF scores for {self.video_name}")
    
    def _load_siti_features(self):
        """Load SI/TI features for the video."""
        siti_file = self.siti_dir / f"{self.video_name}_siti.json"
        
        if not siti_file.exists():
            # Use default values
            print(f"⚠ SI/TI file not found, using defaults")
            self.si = 50.0
            self.ti = 10.0
        else:
            with open(siti_file, 'r') as f:
                siti_data = json.load(f)
                self.si = siti_data['mean_si']
                self.ti = siti_data['mean_ti']
        
        # Normalize SI/TI (typical ranges: SI 0-100, TI 0-50)
        self.si_normalized = np.clip(self.si / 100.0, 0, 1)
        self.ti_normalized = np.clip(self.ti / 50.0, 0, 1)
        
        print(f"✓ Loaded SI/TI features: SI={self.si:.1f}, TI={self.ti:.1f}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Select random trace
        self.current_trace_idx = random.randint(0, len(self.traces) - 1)
        self.current_trace = self.traces[self.current_trace_idx]
        
        # Reset episode state
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.throughput_history = [1.0] * 8  # Initialize with neutral value
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Throughput history (last 8 chunks)
        throughput_obs = np.array(self.throughput_history[-8:], dtype=np.float32)
        
        # Buffer level (normalized)
        buffer_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        
        # Last bitrate (normalized index)
        last_bitrate_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        
        # SI/TI features
        si_obs = self.si_normalized
        ti_obs = self.ti_normalized
        
        # VMAF predictions for all bitrate levels
        vmaf_obs = np.array([
            self.vmaf_normalized.get(br, 0.5)
            for br in self.BITRATE_LEVELS
        ], dtype=np.float32)
        
        # Concatenate all features
        obs = np.concatenate([
            throughput_obs,
            [buffer_obs],
            [last_bitrate_obs],
            [si_obs, ti_obs],
            vmaf_obs
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Get selected bitrate
        bitrate_kbps = self.BITRATE_LEVELS[action]
        
        # Calculate chunk size (bits)
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        # Get current throughput (Kbps)
        time_idx = int(self.chunk_idx * self.CHUNK_DURATION)
        if time_idx < len(self.current_trace['throughput_kbps']):
            throughput_kbps = self.current_trace['throughput_kbps'][time_idx]
        else:
            # Loop trace if we exceed its length
            time_idx = time_idx % len(self.current_trace['throughput_kbps'])
            throughput_kbps = self.current_trace['throughput_kbps'][time_idx]
        
        # Calculate download time
        download_time = chunk_size_bits / (throughput_kbps * 1000)
        
        # Calculate rebuffering
        rebuffer_time = max(0, download_time - self.buffer_level)
        
        # Update buffer level
        self.buffer_level = max(0, self.buffer_level - download_time)
        self.buffer_level = min(self.buffer_level + self.CHUNK_DURATION, self.BUFFER_MAX)
        
        # Calculate reward components
        quality = self.vmaf_scores.get(bitrate_kbps, 50.0) / 100.0
        rebuffer_penalty = self.REBUFFER_PENALTY * rebuffer_time
        
        # Smoothness penalty (bitrate change)
        bitrate_change = abs(action - self.last_bitrate_idx)
        smooth_penalty = self.SMOOTH_PENALTY * bitrate_change / (len(self.BITRATE_LEVELS) - 1)
        
        # Total reward
        reward = quality - rebuffer_penalty - smooth_penalty
        
        # Update metrics
        self.total_quality += quality
        self.total_rebuffer += rebuffer_time
        self.total_smooth += smooth_penalty
        
        # Update throughput history (normalized)
        throughput_normalized = np.clip(throughput_kbps / 6000.0, 0, 1)
        self.throughput_history.append(throughput_normalized)
        
        # Update state
        self.last_bitrate_idx = action
        self.chunk_idx += 1
        
        # Check if episode is done
        terminated = self.chunk_idx >= self.max_chunks
        truncated = False
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'bitrate': bitrate_kbps,
            'throughput': throughput_kbps,
            'buffer': self.buffer_level,
            'rebuffer': rebuffer_time,
            'quality': quality,
            'reward': reward
        })
        
        return obs, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict:
        """Get episode information."""
        return {
            'chunk_idx': self.chunk_idx,
            'trace_idx': self.current_trace_idx,
            'total_rebuffer': self.total_rebuffer,
            'total_quality': self.total_quality,
            'avg_quality': self.total_quality / max(1, self.chunk_idx),
            'buffer_level': self.buffer_level
        }
    
    def render(self):
        """Render environment (optional)."""
        if self.chunk_idx > 0:
            print(f"Chunk {self.chunk_idx}/{self.max_chunks} | "
                  f"Buffer: {self.buffer_level:.1f}s | "
                  f"Rebuffer: {self.total_rebuffer:.2f}s | "
                  f"Avg Quality: {self.total_quality/self.chunk_idx:.2f}")


def test_environment():
    """Test the ABR environment."""
    print("\n🎮 Testing ABR Environment\n")
    
    # Create environment
    env = ABREnv(
        video_name='sample1',
        trace_dir='src/data_preparation/data/network_traces/processed',
        vmaf_dir='src/data_preparation/data/vmaf_scores',
        siti_dir='src/data_preparation/data/content_features',
        max_chunks=10,
        random_seed=42
    )
    
    print(f"\n✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    
    # Test episode
    print(f"\n{'='*60}")
    print("Running test episode with random actions")
    print(f"{'='*60}\n")
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(10):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: "
              f"Action={action} ({env.BITRATE_LEVELS[action]} Kbps), "
              f"Reward={reward:.2f}, "
              f"Buffer={info['buffer']:.1f}s, "
              f"Rebuffer={info['rebuffer']:.2f}s")
        
        if terminated or truncated:
            break
    
    print(f"\n{'='*60}")
    print(f"Episode finished!")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total rebuffering: {info['total_rebuffer']:.2f}s")
    print(f"  Average quality: {info['avg_quality']:.2f}")
    print(f"{'='*60}\n")
    
    print("✓ Environment test completed successfully!")
    print("\nNext step: Train PPO agent")
    print("  python src/training/train_ppo.py")


if __name__ == '__main__':
    test_environment()