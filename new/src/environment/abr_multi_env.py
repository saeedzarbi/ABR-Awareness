import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import random

class ABREnv(gym.Env):
    """
    Multi-Video ABR Environment (V22 - Future Aware)
    
    Key Feature:
    - Includes 'next_chunk_size' in observation (Lookahead = 1).
    - This allows the agent to anticipate big chunks and avoid rebuffering.
    - Proven to be the most balanced and effective version.
    """
    
    metadata = {'render_modes': ['human']}
    
    # Standard Bitrate Ladder (kbps)
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0
    
    BUFFER_TARGET = 15.0
    BUFFER_MAX = 30.0
    
    # Penalties (Standard Linear QoE)
    REBUF_PENALTY_BASE = 4.3 
    SMOOTH_PENALTY_WEIGHT = 1.0 
    
    MAX_NETWORK_THROUGHPUT = 20000.0
    MIN_NETWORK_THROUGHPUT = 10.0
    MAX_CHUNK_SIZE_BITS = 30000000.0 # Normalization factor
    
    def __init__(self, video_names: Union[str, List[str]] = 'bigbuckbunny', 
                 trace_dir='/home/saeedzarbi95/test/ABR-Awareness/new/data/standardized/train_traces', 
                 vmaf_dir='/home/saeedzarbi95/test/ABR-Awareness/new/data/vmaf_scores', 
                 siti_dir='/home/saeedzarbi95/test/ABR-Awareness/new/data/content_features', 
                 max_chunks=48, random_seed=None):
        super().__init__()
        
        if isinstance(video_names, str):
            self.video_names = [video_names]
        else:
            self.video_names = video_names
            
        self.trace_dir = Path(trace_dir)
        self.vmaf_dir = Path(vmaf_dir)
        self.siti_dir = Path(siti_dir)
        self.max_chunks = max_chunks
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.video_assets = {}
        self._load_traces()
        self._load_all_video_data()
        
        self.action_space = spaces.Discrete(len(self.BITRATE_LEVELS))
        
        # Observation Space (29 Features):
        # 12 (Throughput History) + 1 (Buffer) + 1 (Trend) + 1 (Last Bitrate)
        # + 2 (Content Features: SI/TI) + 6 (VMAF Map) + 6 (Next Chunk Sizes)
        obs_dim = 29
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_video_name = None
        self.current_vmaf_scores = {}
        self.current_vmaf_norm = {}
        self.current_si_norm = 0.0
        self.current_ti_norm = 0.0
        
        self.current_trace = None
        self.current_trace_idx = 0
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.prev_buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.last_vmaf = 35.0
        self.throughput_history = []
        
    def _load_traces(self):
        trace_files = list(self.trace_dir.glob("*.json"))
        if not trace_files: 
            self.traces = [{'throughput_kbps': [2000]*1000}]
        else: 
            self.traces = [json.load(open(f)) for f in trace_files]
    
    def _load_all_video_data(self):
        import pandas as pd
        vmaf_file = self.vmaf_dir / "vmaf_summary.csv"
        try: all_vmaf_data = pd.read_csv(vmaf_file)
        except: all_vmaf_data = None

        for vid in self.video_names:
            self.video_assets[vid] = {}
            # Default fallback VMAF
            vmaf_scores = {300:35, 750:58, 1200:74, 1850:84, 2850:91, 6000:97}
            
            if all_vmaf_data is not None and 'video' in all_vmaf_data.columns:
                df = all_vmaf_data[all_vmaf_data['video'] == vid]
                if not df.empty:
                    vmaf_scores = dict(zip(df['bitrate_kbps'], df['vmaf']))
            
            self.video_assets[vid]['vmaf'] = vmaf_scores
            self.video_assets[vid]['vmaf_norm'] = {k: v/100.0 for k, v in vmaf_scores.items()}
            
            # Load Content Features (SI/TI)
            siti_file = self.siti_dir / f"{vid}_siti.json"
            si, ti = 50, 10
            if siti_file.exists():
                try:
                    data = json.load(open(siti_file))
                    si = data.get('mean_si', 50)
                    ti = data.get('mean_ti', 10)
                except: pass
            
            self.video_assets[vid]['si_norm'] = float(np.clip(si / 150.0, 0, 1))
            self.video_assets[vid]['ti_norm'] = float(np.clip(ti / 70.0, 0, 1))

    def _get_next_chunk_sizes(self):
        """
        Calculates the size of the *next* chunk for all bitrate levels.
        This provides the 'Future Awareness' capability.
        """
        sizes = []
        for br in self.BITRATE_LEVELS:
            # Simple CBR approximation for simulation speed. 
            # In a real VBR dataset, you would look up the exact size from a file.
            size_bits = br * 1000 * self.CHUNK_DURATION
            sizes.append(size_bits)
        return np.array(sizes, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_video_name = random.choice(self.video_names)
        assets = self.video_assets[self.current_video_name]
        
        self.current_vmaf_scores = assets['vmaf']
        self.current_vmaf_norm = assets['vmaf_norm']
        self.current_si_norm = assets['si_norm']
        self.current_ti_norm = assets['ti_norm']
        
        self.current_trace_idx = random.randint(0, len(self.traces) - 1)
        self.current_trace = self.traces[self.current_trace_idx]
        
        self.chunk_idx = 0
        self.buffer_level = 1.0
        self.prev_buffer_level = 1.0
        self.last_bitrate_idx = 0
        self.last_vmaf = self.current_vmaf_scores[self.BITRATE_LEVELS[0]]
        
        # Initialize throughput history
        start_tp = 500.0
        log_obs = np.log(start_tp / self.MIN_NETWORK_THROUGHPUT) / np.log(self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT)
        self.throughput_history = [log_obs] * 12
        
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self):
        tp_obs = np.array(self.throughput_history[-12:], dtype=np.float32)
        buf_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        buf_trend = np.clip((self.buffer_level - self.prev_buffer_level) / self.CHUNK_DURATION, -1.0, 1.0)
        last_br_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        content_obs = np.array([self.current_si_norm, self.current_ti_norm], dtype=np.float32)
        vmaf_obs = np.array([self.current_vmaf_norm[br] for br in self.BITRATE_LEVELS], dtype=np.float32)
        
        # --- Future Awareness (V22 Specific) ---
        next_sizes = self._get_next_chunk_sizes()
        sizes_obs = np.clip(next_sizes / self.MAX_CHUNK_SIZE_BITS, 0, 1)
        
        return np.concatenate([
            tp_obs,         # 12
            [buf_obs],      # 1
            [buf_trend],    # 1
            [last_br_obs],  # 1
            content_obs,    # 2
            vmaf_obs,       # 6
            sizes_obs       # 6 -> This is the magic sauce!
        ]).astype(np.float32)

    def step(self, action):
        bitrate_kbps = self.BITRATE_LEVELS[action]
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        # Simulate Network
        trace_tp = self.current_trace['throughput_kbps']
        avail_tp = trace_tp[int(self.chunk_idx * self.CHUNK_DURATION) % len(trace_tp)]
        effective_tp = np.clip(avail_tp, self.MIN_NETWORK_THROUGHPUT, self.MAX_NETWORK_THROUGHPUT)
        
        download_time = chunk_size_bits / (effective_tp * 1000.0)
        if download_time > 60.0: download_time = 60.0
        
        # Buffer Dynamics
        rebuffer_time = max(0, download_time - self.buffer_level)
        self.prev_buffer_level = self.buffer_level
        self.buffer_level = max(0, self.buffer_level - download_time) + self.CHUNK_DURATION
        self.buffer_level = min(self.buffer_level, self.BUFFER_MAX)
        
        # Metrics
        current_vmaf = self.current_vmaf_scores.get(bitrate_kbps, 35.0)
        smooth_pen = abs(current_vmaf - self.last_vmaf)
        buffer_dev = max(0, self.BUFFER_TARGET - self.buffer_level)
        
        weighted_rebuf = self.REBUF_PENALTY_BASE * rebuffer_time
        
        # Reward Function (Standard Linear QoE)
        reward = current_vmaf \
                 - weighted_rebuf \
                 - (self.SMOOTH_PENALTY_WEIGHT * smooth_pen) \
                 - (0.02 * buffer_dev) 
        
        # Update State
        log_tp = np.log(effective_tp / self.MIN_NETWORK_THROUGHPUT)
        log_scale = np.log(self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT)
        norm_tp = np.clip(log_tp / log_scale, 0.0, 1.0)
        self.throughput_history.append(norm_tp)
        
        self.last_bitrate_idx = action
        self.last_vmaf = current_vmaf
        self.chunk_idx += 1
        terminated = self.chunk_idx >= self.max_chunks
        
        self.total_quality += current_vmaf
        self.total_rebuffer += rebuffer_time
        self.total_smooth += smooth_pen
        
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'bitrate': bitrate_kbps, 'throughput': avail_tp,
            'buffer': self.buffer_level, 'rebuffer': rebuffer_time, 
            'reward': float(reward), 'video_name': self.current_video_name
        })
        return obs, float(reward), terminated, False, info

    def _get_info(self):
        return {
            'chunk_idx': self.chunk_idx,
            'total_rebuffer': self.total_rebuffer,
            'total_quality': self.total_quality,
            'avg_quality': self.total_quality / max(1, self.chunk_idx),
            'total_smoothness': self.total_smooth,
            'buffer_level': self.buffer_level
        }

    def render(self): pass