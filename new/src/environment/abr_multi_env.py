import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import random

class ABREnv(gym.Env):
    """
    Multi-Video ABR Environment (V23 - The Hybrid Analyst)
    
    Innovation:
    - Combines "Future Awareness" (Next Chunk Sizes) from V22.
    - Adds "Trend Analysis" (Harmonic Mean & Derivative) from Control Theory.
    - This gives the agent the 'math skills' of MPC + the 'foresight' of Oracle.
    """
    
    metadata = {'render_modes': ['human']}
    
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0
    
    BUFFER_TARGET = 15.0
    BUFFER_MAX = 30.0
    
    # Tuned from V22 results
    REBUF_PENALTY_BASE = 4.8  # Slight increase to beat MPC's 7s rebuffer
    SMOOTH_PENALTY_WEIGHT = 0.5 
    
    MAX_NETWORK_THROUGHPUT = 20000.0
    MIN_NETWORK_THROUGHPUT = 10.0
    MAX_CHUNK_SIZE_BITS = 30000000.0
    
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
        
        # Obs Dim Breakdown:
        # History(12) + Buffer(1) + BufTrend(1) + LastBR(1) + Content(2) + VMAF(6) + FutureSizes(6)
        # + HarmonicMean(1) + NetworkTrend(1)
        # Total = 29 + 2 = 31 features
        obs_dim = 31
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_video_name = None
        self.current_vmaf_scores = {}
        self.current_vmaf_norm = {}
        self.current_vmaf_range = 0.0
        
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
        
        # Real-time tracking
        self.raw_throughput_history = [] # To calc harmonic mean correctly
        
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
            vmaf_scores = {300:35, 750:58, 1200:74, 1850:84, 2850:91, 6000:97}
            if all_vmaf_data is not None and 'video' in all_vmaf_data.columns:
                df = all_vmaf_data[all_vmaf_data['video'] == vid]
                if not df.empty:
                    vmaf_scores = dict(zip(df['bitrate_kbps'], df['vmaf']))
            
            self.video_assets[vid]['vmaf'] = vmaf_scores
            self.video_assets[vid]['vmaf_norm'] = {k: v/100.0 for k, v in vmaf_scores.items()}
            
            min_v = min(vmaf_scores.values())
            max_v = max(vmaf_scores.values())
            self.video_assets[vid]['vmaf_range'] = max_v - min_v

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
        sizes = []
        for br in self.BITRATE_LEVELS:
            size_bits = br * 1000 * self.CHUNK_DURATION
            sizes.append(size_bits)
        return np.array(sizes, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_video_name = random.choice(self.video_names)
        assets = self.video_assets[self.current_video_name]
        
        self.current_vmaf_scores = assets['vmaf']
        self.current_vmaf_norm = assets['vmaf_norm']
        self.current_vmaf_range = assets.get('vmaf_range', 40.0)
        self.current_si_norm = assets['si_norm']
        self.current_ti_norm = assets['ti_norm']
        
        self.current_trace_idx = random.randint(0, len(self.traces) - 1)
        self.current_trace = self.traces[self.current_trace_idx]
        
        self.chunk_idx = 0
        self.buffer_level = 1.0
        self.prev_buffer_level = 1.0
        self.last_bitrate_idx = 0
        self.last_vmaf = self.current_vmaf_scores[self.BITRATE_LEVELS[0]]
        
        start_tp = 500.0
        log_obs = np.log(start_tp / self.MIN_NETWORK_THROUGHPUT) / np.log(self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT)
        self.throughput_history = [log_obs] * 12
        self.raw_throughput_history = [start_tp] * 5 # Short memory for harmonic mean
        
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
        
        # Future Awareness
        next_sizes = self._get_next_chunk_sizes()
        sizes_obs = np.clip(next_sizes / self.MAX_CHUNK_SIZE_BITS, 0, 1)
        
        # --- NEW: Hybrid Analyst Features ---
        # 1. Harmonic Mean (MPC Style)
        recent_tp = np.array(self.raw_throughput_history[-5:]) # Last 5 chunks
        harmonic_mean = len(recent_tp) / np.sum(1.0 / (recent_tp + 1e-5))
        harmonic_obs = np.log(harmonic_mean / self.MIN_NETWORK_THROUGHPUT) / np.log(self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT)
        harmonic_obs = np.clip(harmonic_obs, 0, 1)
        
        # 2. Network Trend (Derivative)
        if len(self.raw_throughput_history) >= 2:
            trend = (self.raw_throughput_history[-1] - self.raw_throughput_history[-2]) / self.MAX_NETWORK_THROUGHPUT
        else:
            trend = 0.0
        trend_obs = np.clip(trend, -1.0, 1.0)
        
        return np.concatenate([
            tp_obs,         # 12
            [buf_obs],      # 1
            [buf_trend],    # 1
            [last_br_obs],  # 1
            content_obs,    # 2
            vmaf_obs,       # 6
            sizes_obs,      # 6
            [harmonic_obs], # 1 (NEW)
            [trend_obs]     # 1 (NEW)
        ]).astype(np.float32)

    def step(self, action):
        bitrate_kbps = self.BITRATE_LEVELS[action]
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        trace_tp = self.current_trace['throughput_kbps']
        avail_tp = trace_tp[int(self.chunk_idx * self.CHUNK_DURATION) % len(trace_tp)]
        effective_tp = np.clip(avail_tp, self.MIN_NETWORK_THROUGHPUT, self.MAX_NETWORK_THROUGHPUT)
        
        download_time = chunk_size_bits / (effective_tp * 1000.0)
        if download_time > 60.0: download_time = 60.0
        
        rebuffer_time = max(0, download_time - self.buffer_level)
        self.prev_buffer_level = self.buffer_level
        self.buffer_level = max(0, self.buffer_level - download_time) + self.CHUNK_DURATION
        self.buffer_level = min(self.buffer_level, self.BUFFER_MAX)
        
        current_vmaf = self.current_vmaf_scores.get(bitrate_kbps, 35.0)
        smooth_pen = abs(current_vmaf - self.last_vmaf)
        
        buffer_dev = max(0, self.BUFFER_TARGET - self.buffer_level)
        risk_factor = 1.0 + (0.1 * buffer_dev) 
        
        gradient_factor = self.current_vmaf_range / 40.0
        weighted_rebuf = self.REBUF_PENALTY_BASE * gradient_factor * risk_factor * rebuffer_time
        
        reward = current_vmaf \
                 - weighted_rebuf \
                 - (self.SMOOTH_PENALTY_WEIGHT * smooth_pen) \
                 - (0.02 * buffer_dev) 
        
        # Update histories
        log_tp = np.log(effective_tp / self.MIN_NETWORK_THROUGHPUT)
        log_scale = np.log(self.MAX_NETWORK_THROUGHPUT / self.MIN_NETWORK_THROUGHPUT)
        norm_tp = np.clip(log_tp / log_scale, 0.0, 1.0)
        self.throughput_history.append(norm_tp)
        self.raw_throughput_history.append(effective_tp) # Store raw for Harmonic calc
        
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
    @property
    def vmaf_scores(self): return self.current_vmaf_scores
    @property
    def video_name(self): return self.current_video_name
    @property
    def trace(self): return self.current_trace