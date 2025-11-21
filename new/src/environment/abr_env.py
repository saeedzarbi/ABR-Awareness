import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import random

class ABREnv(gym.Env):
    """
    Custom Gym Environment for ABR streaming simulation.
    Scientific implementation based on Lyapunov Optimization.
    """
    metadata = {'render_modes': ['human']}
    
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0
    BUFFER_TARGET = 15.0
    BUFFER_MAX = 30.0
    LYAPUNOV_GAIN = 0.5
    REBUF_PENALTY_BASE = 4.3
    SMOOTH_PENALTY_WEIGHT = 1.0 
    
    def __init__(self, video_name='bigbuckbunny', trace_dir='data/network_traces/processed', 
                 vmaf_dir='data/vmaf_scores', siti_dir='data/content_features', 
                 max_chunks=48, random_seed=None):
        super().__init__()
        self.video_name = video_name
        self.trace_dir = Path(trace_dir)
        self.vmaf_dir = Path(vmaf_dir)
        self.siti_dir = Path(siti_dir)
        self.max_chunks = max_chunks
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self._load_traces()
        self._load_vmaf_scores()
        self._load_siti_features()
        
        self.action_space = spaces.Discrete(len(self.BITRATE_LEVELS))
        obs_dim = 8 + 1 + 1 + 2 + 6 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self.current_trace = None
        self.current_trace_idx = 0
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.last_quality_metric = 0.0
        self.throughput_history = []
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
    
    def _load_traces(self):
        trace_files = list(self.trace_dir.glob("*.json"))
        if not trace_files:
            print(f"⚠ No traces found in {self.trace_dir}. Using dummy trace.")
            self.traces = [{'throughput_kbps': [2000]*1000}]
        else:
            self.traces = [json.load(open(f)) for f in trace_files]
            print(f"✓ Loaded {len(self.traces)} traces from {self.trace_dir}")
    
    def _load_vmaf_scores(self):
        vmaf_file = self.vmaf_dir / self.video_name / "vmaf_summary.csv"
        if not vmaf_file.exists(): vmaf_file = self.vmaf_dir / "vmaf_summary.csv"
        if vmaf_file.exists():
            import pandas as pd
            df = pd.read_csv(vmaf_file)
            if 'video' in df.columns: df = df[df['video'] == self.video_name]
            if not df.empty:
                self.vmaf_scores = dict(zip(df['bitrate_kbps'], df['vmaf']))
            else: self._use_default_vmaf()
        else: self._use_default_vmaf()
        self.vmaf_normalized = {k: v/100.0 for k, v in self.vmaf_scores.items()}
    
    def _use_default_vmaf(self):
        self.vmaf_scores = {300: 35.0, 750: 58.0, 1200: 74.0, 1850: 84.0, 2850: 91.0, 6000: 97.0}

    def _load_siti_features(self):
        siti_file = self.siti_dir / f"{self.video_name}_siti.json"
        if siti_file.exists():
            with open(siti_file, 'r') as f:
                data = json.load(f)
                self.si = data.get('mean_si', 50.0)
                self.ti = data.get('mean_ti', 10.0)
        else: self.si, self.ti = 50.0, 10.0
        self.si_norm = np.clip(self.si / 100.0, 0, 1)
        self.ti_norm = np.clip(self.ti / 50.0, 0, 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_trace_idx = random.randint(0, len(self.traces) - 1)
        self.current_trace = self.traces[self.current_trace_idx]
        self.chunk_idx = 0
        self.buffer_level = 1.0
        self.last_bitrate_idx = 0
        self.last_quality_metric = self.vmaf_scores[self.BITRATE_LEVELS[0]] / 100.0
        self.throughput_history = [1.0] * 8 
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
        return self._get_observation(), self._get_info()
    
    def _get_observation(self):
        tp_obs = np.array(self.throughput_history[-8:], dtype=np.float32)
        buf_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        last_br_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        content_obs = np.array([self.si_norm, self.ti_norm], dtype=np.float32)
        vmaf_obs = np.array([self.vmaf_normalized[br] for br in self.BITRATE_LEVELS], dtype=np.float32)
        return np.concatenate([tp_obs, [buf_obs], [last_br_obs], content_obs, vmaf_obs]).astype(np.float32)

    def step(self, action):
        bitrate_kbps = self.BITRATE_LEVELS[action]
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        trace_tp = self.current_trace['throughput_kbps']
        curr_time_idx = int(self.chunk_idx * self.CHUNK_DURATION) % len(trace_tp)
        avail_tp = trace_tp[curr_time_idx]
        
        download_time = chunk_size_bits / (avail_tp * 1000.0 + 1e-6)
        rebuffer_time = max(0, download_time - self.buffer_level)
        
        self.buffer_level = max(0, self.buffer_level - download_time) + self.CHUNK_DURATION
        self.buffer_level = min(self.buffer_level, self.BUFFER_MAX)
        
        # Reward
        curr_vmaf = self.vmaf_scores.get(bitrate_kbps, 35.0) / 100.0
        smooth_pen = abs(curr_vmaf - self.last_quality_metric)
        buf_dev = self.BUFFER_TARGET - self.buffer_level
        risk_factor = 1.0 + np.exp(self.LYAPUNOV_GAIN * buf_dev * 0.5) if buf_dev > 0 else 1.0
        weighted_rebuf = self.REBUF_PENALTY_BASE * risk_factor * rebuffer_time
        
        reward = curr_vmaf - weighted_rebuf - (self.SMOOTH_PENALTY_WEIGHT * smooth_pen) - (0.05 * max(0, buf_dev))
        
        self.throughput_history.append(np.clip(avail_tp / 6000.0, 0, 1))
        self.last_bitrate_idx = action
        self.last_quality_metric = curr_vmaf
        self.chunk_idx += 1
        
        terminated = self.chunk_idx >= self.max_chunks
        
        self.total_quality += curr_vmaf
        self.total_rebuffer += rebuffer_time
        self.total_smooth += smooth_pen
        
        obs = self._get_observation()
        info = self._get_info()
        # Add debug info
        info.update({'bitrate': bitrate_kbps, 'buffer': self.buffer_level, 'reward': reward})
        
        return obs, reward, terminated, False, info

    def _get_info(self) -> Dict:
        return {
            'chunk_idx': self.chunk_idx,
            'total_rebuffer': self.total_rebuffer,
            'total_quality': self.total_quality,  # <--- ADDED THIS KEY (Fixes KeyError)
            'avg_quality': self.total_quality / max(1, self.chunk_idx),
            'total_smoothness': self.total_smooth
        }


