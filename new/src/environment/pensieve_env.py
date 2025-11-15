"""
Pensieve-style environment (simplified).
Based on: "Neural Adaptive Video Streaming with Pensieve" (MIT, 2017)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import random


class PensieveEnv(gym.Env):
    """
    Pensieve-style ABR environment.
    
    Key differences from our ABR env:
    - State: past throughput, past chunk qualities, buffer
    - Reward: linear QoE (simpler than ours)
    - No content-awareness (no VMAF/SI/TI)
    """
    
    metadata = {'render_modes': ['human']}
    
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0
    BUFFER_MAX = 30.0
    
    # Pensieve-style linear reward
    QUALITY_WEIGHT = 1.0
    REBUFFER_PENALTY = 4.3
    SMOOTH_PENALTY = 1.0
    
    def __init__(
        self,
        trace_dir: str = 'data/network_traces/processed',
        max_chunks: int = 48,
        random_seed: Optional[int] = None
    ):
        super().__init__()
        
        self.trace_dir = Path(trace_dir)
        self.max_chunks = max_chunks
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self._load_traces()
        
        # Action space
        self.action_space = spaces.Discrete(len(self.BITRATE_LEVELS))
        
        # Observation: past throughput (8) + past quality (8) + buffer (1) + last bitrate (1)
        obs_dim = 8 + 8 + 1 + 1  # Total: 18
        
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
        self.quality_history = []
        self.total_rebuffer = 0.0
    
    def _load_traces(self):
        """Load network traces."""
        trace_files = list(self.trace_dir.glob("*.json"))
        
        if not trace_files:
            raise ValueError(f"No traces found in {self.trace_dir}")
        
        self.traces = []
        for trace_file in trace_files:
            with open(trace_file, 'r') as f:
                trace = json.load(f)
                self.traces.append(trace)
        
        print(f"✓ Pensieve env: Loaded {len(self.traces)} traces")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_trace_idx = random.randint(0, len(self.traces) - 1)
        self.current_trace = self.traces[self.current_trace_idx]
        
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.throughput_history = [1.0] * 8
        self.quality_history = [0.5] * 8
        self.total_rebuffer = 0.0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Pensieve-style observation."""
        # Past throughput (normalized)
        throughput_obs = np.array(self.throughput_history[-8:], dtype=np.float32)
        
        # Past quality (bitrate normalized)
        quality_obs = np.array(self.quality_history[-8:], dtype=np.float32)
        
        # Buffer level (normalized)
        buffer_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        
        # Last bitrate
        last_bitrate_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        
        obs = np.concatenate([
            throughput_obs,
            quality_obs,
            [buffer_obs],
            [last_bitrate_obs]
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step."""
        bitrate_kbps = self.BITRATE_LEVELS[action]
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        # Get throughput
        time_idx = int(self.chunk_idx * self.CHUNK_DURATION)
        if time_idx < len(self.current_trace['throughput_kbps']):
            throughput_kbps = self.current_trace['throughput_kbps'][time_idx]
        else:
            time_idx = time_idx % len(self.current_trace['throughput_kbps'])
            throughput_kbps = self.current_trace['throughput_kbps'][time_idx]
        
        # Download time
        download_time = chunk_size_bits / (throughput_kbps * 1000)
        
        # Rebuffering
        rebuffer_time = max(0, download_time - self.buffer_level)
        
        # Update buffer
        self.buffer_level = max(0, self.buffer_level - download_time)
        self.buffer_level = min(self.buffer_level + self.CHUNK_DURATION, self.BUFFER_MAX)
        
        # Pensieve-style linear reward
        # Quality = bitrate in Mbps
        quality = bitrate_kbps / 1000.0
        
        # Smoothness penalty
        if self.chunk_idx > 0:
            last_bitrate = self.BITRATE_LEVELS[self.last_bitrate_idx]
            smooth_penalty = abs(bitrate_kbps - last_bitrate) / 1000.0
        else:
            smooth_penalty = 0.0
        
        # Total reward (Pensieve formula)
        reward = (
            self.QUALITY_WEIGHT * quality
            - self.REBUFFER_PENALTY * rebuffer_time
            - self.SMOOTH_PENALTY * smooth_penalty
        )
        
        # Update history
        self.total_rebuffer += rebuffer_time
        
        throughput_normalized = np.clip(throughput_kbps / 6000.0, 0, 1)
        self.throughput_history.append(throughput_normalized)
        
        quality_normalized = bitrate_kbps / 6000.0
        self.quality_history.append(quality_normalized)
        
        # Update state
        self.last_bitrate_idx = action
        self.chunk_idx += 1
        
        # Check done
        terminated = self.chunk_idx >= self.max_chunks
        truncated = False
        
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'bitrate': bitrate_kbps,
            'throughput': throughput_kbps,
            'buffer': self.buffer_level,
            'rebuffer': rebuffer_time,
            'reward': reward
        })
        
        return obs, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict:
        """Get info."""
        avg_quality = np.mean(self.quality_history) if self.quality_history else 0.5        
        return {
            'chunk_idx': self.chunk_idx,
            'total_rebuffer': self.total_rebuffer,
            'avg_quality': avg_quality,
            'buffer_level': self.buffer_level
        }