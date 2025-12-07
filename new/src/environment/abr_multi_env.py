import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import random

class ABREnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0
    
    BUFFER_TARGET = 15.0
    BUFFER_MAX = 30.0
    
    LYAPUNOV_GAIN = 0.15
    
    REBUF_PENALTY_BASE = 60.0  
    
    SMOOTH_PENALTY_WEIGHT = 1.0
    
    def __init__(self, video_names: Union[str, List[str]] = 'bigbuckbunny', 
                 trace_dir='data/network_traces/processed', 
                 vmaf_dir='data/vmaf_scores', siti_dir='data/content_features', 
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
        obs_dim = 18
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self.current_video_name = None
        self.current_vmaf_scores = {}
        self.current_vmaf_norm = {}
        self.current_si_norm = 0.0
        self.current_ti_norm = 0.0
        
        self.current_trace = None
        self.current_trace_idx = 0
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.last_vmaf = 35.0
        self.throughput_history = []
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
    
    def _load_traces(self):
        trace_files = list(self.trace_dir.glob("*.json"))
        if not trace_files: 
            self.traces = [{'throughput_kbps': [2000]*1000}]
        else: 
            self.traces = [json.load(open(f)) for f in trace_files]
    
    def _load_all_video_data(self):
        import pandas as pd
        
        vmaf_file = self.vmaf_dir / "vmaf_summary.csv"
        all_vmaf_data = pd.DataFrame()
        if vmaf_file.exists():
            try: all_vmaf_data = pd.read_csv(vmaf_file)
            except: pass

        for vid in self.video_names:
            self.video_assets[vid] = {}
            
            # --- VMAF ---
            vmaf_scores = {300:35, 750:58, 1200:74, 1850:84, 2850:91, 6000:97}
            if not all_vmaf_data.empty and 'video' in all_vmaf_data.columns:
                df = all_vmaf_data[all_vmaf_data['video'] == vid]
                if not df.empty:
                    vmaf_scores = dict(zip(df['bitrate_kbps'], df['vmaf']))
            
            self.video_assets[vid]['vmaf'] = vmaf_scores
            self.video_assets[vid]['vmaf_norm'] = {k: v/100.0 for k, v in vmaf_scores.items()}

            # --- SI/TI ---
            siti_file = self.siti_dir / f"{vid}_siti.json"
            si, ti = 50, 10
            if siti_file.exists():
                try:
                    data = json.load(open(siti_file))
                    si = data.get('mean_si', 50)
                    ti = data.get('mean_ti', 10)
                except: pass
            
            self.video_assets[vid]['si_norm'] = np.clip(si / 150.0, 0, 1) 
            self.video_assets[vid]['ti_norm'] = np.clip(ti / 70.0, 0, 1) 

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
        self.last_bitrate_idx = 0
        self.last_vmaf = self.current_vmaf_scores[self.BITRATE_LEVELS[0]]
        self.throughput_history = [1.0] * 8 
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
        
        if random.random() < 0.01:
            si_raw = self.current_si_norm * 300.0 
            
            log_message = (
                f"\n🔍 [Environment Debug] Video: {self.current_video_name}\n"
                f"   SI Input: {self.current_si_norm:.2f} (Raw ~{si_raw:.0f})\n"
                f"   Penalty Used: {self.REBUF_PENALTY_BASE}\n"
                f"{'-' * 40}\n"
            )
            
            try:
                with open("env_debug_log.txt", "a") as f:
                    f.write(log_message)
            except Exception as e:
                print(f"Error writing to log file: {e}")
            
        return self._get_observation(), self._get_info()
    
    def _get_observation(self):
        tp_obs = np.array(self.throughput_history[-8:], dtype=np.float32)
        buf_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        last_br_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        
        content_obs = np.array([self.current_si_norm, self.current_ti_norm], dtype=np.float32)
        vmaf_obs = np.array([self.current_vmaf_norm[br] for br in self.BITRATE_LEVELS], dtype=np.float32)
        
        return np.concatenate([tp_obs, [buf_obs], [last_br_obs], content_obs, vmaf_obs]).astype(np.float32)

    def step(self, action):
        bitrate_kbps = self.BITRATE_LEVELS[action]
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        trace_tp = self.current_trace['throughput_kbps']
        avail_tp = trace_tp[int(self.chunk_idx * self.CHUNK_DURATION) % len(trace_tp)]
        
        download_time = chunk_size_bits / (avail_tp * 1000.0 + 1e-6)
        rebuffer_time = max(0, download_time - self.buffer_level)
        self.buffer_level = max(0, self.buffer_level - download_time) + self.CHUNK_DURATION
        self.buffer_level = min(self.buffer_level, self.BUFFER_MAX)
        
        current_vmaf = self.current_vmaf_scores.get(bitrate_kbps, 35.0)
        smooth_pen = abs(current_vmaf - self.last_vmaf)
        
        buffer_dev = max(0, self.BUFFER_TARGET - self.buffer_level)
        risk_factor = 1.0 + np.exp(self.LYAPUNOV_GAIN * buffer_dev) if buffer_dev > 0 else 1.0
        risk_factor = min(risk_factor, 15.0)
        
        weighted_rebuf = self.REBUF_PENALTY_BASE * risk_factor * rebuffer_time
        
        reward = current_vmaf \
                 - weighted_rebuf \
                 - (self.SMOOTH_PENALTY_WEIGHT * smooth_pen) \
                 - (0.1 * buffer_dev)
        
        self.throughput_history.append(np.clip(avail_tp / 6000.0, 0, 1))
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
            'buffer': self.buffer_level, 'buffer_level': self.buffer_level,
            'rebuffer': rebuffer_time, 'reward': reward,
            'video_name': self.current_video_name
        })
        return obs, reward, terminated, False, info

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