import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import random

class ABREnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    CHUNK_DURATION = 4.0
    
    # --- Tuned Parameters for Stability & Exploration ---
    BUFFER_TARGET = 15.0
    BUFFER_MAX = 30.0
    
    # 1. کاهش حساسیت نمایی (قبلاً 0.5 بود که خیلی تند بود)
    LYAPUNOV_GAIN = 0.1  
    
    # 2. کاهش جریمه پایه (قبلاً 85 بود)
    # نسبت منطقی: 1 ثانیه قطعی = از دست دادن نیمی از کیفیت ماکسیمم
    REBUF_PENALTY_BASE = 40.0 
    
    # 3. بازگرداندن جریمه نرمی (خیلی کم) برای جلوگیری از نوسان بی‌دلیل
    SMOOTH_PENALTY_WEIGHT = 0.1 
    
    def __init__(self, video_name='sample1', trace_dir='data/network_traces/processed', 
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
        obs_dim = 18
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
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
        if not trace_files: self.traces = [{'throughput_kbps': [2000]*1000}]
        else: self.traces = [json.load(open(f)) for f in trace_files]
    
    def _load_vmaf_scores(self):
        vmaf_file = self.vmaf_dir / "vmaf_summary.csv"
        loaded = False
        if vmaf_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(vmaf_file)
                if 'video' in df.columns: df = df[df['video'] == self.video_name]
                if not df.empty:
                    self.vmaf_scores = dict(zip(df['bitrate_kbps'], df['vmaf']))
                    loaded = True
            except: pass
        if not loaded: self.vmaf_scores = {300:35, 750:58, 1200:74, 1850:84, 2850:91, 6000:97}
        self.vmaf_norm = {k: v/100.0 for k, v in self.vmaf_scores.items()}

    def _load_siti_features(self):
        siti_file = self.siti_dir / f"{self.video_name}_siti.json"
        if siti_file.exists():
            data = json.load(open(siti_file))
            self.si, self.ti = data.get('mean_si', 50), data.get('mean_ti', 10)
        else: self.si, self.ti = 50, 10
        self.si_norm, self.ti_norm = np.clip(self.si/100,0,1), np.clip(self.ti/50,0,1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_trace_idx = random.randint(0, len(self.traces) - 1)
        self.current_trace = self.traces[self.current_trace_idx]
        self.chunk_idx = 0
        self.buffer_level = 1.0
        self.last_bitrate_idx = 0
        self.last_vmaf = self.vmaf_scores[self.BITRATE_LEVELS[0]]
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
        vmaf_obs = np.array([self.vmaf_norm[br] for br in self.BITRATE_LEVELS], dtype=np.float32)
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
        
        # --- CORRECTED REWARD LOGIC ---
        current_vmaf = self.vmaf_scores.get(bitrate_kbps, 35.0)
        smooth_pen = abs(current_vmaf - self.last_vmaf)
        
        # Calculated Risk: Slower growth
        # buffer_dev=5 -> exp(0.1 * 5) = 1.6 (Manageable penalty increase)
        # buffer_dev=10 -> exp(0.1 * 10) = 2.7 (Strong penalty)
        buffer_dev = max(0, self.BUFFER_TARGET - self.buffer_level)
        risk_factor = 1.0 + np.exp(self.LYAPUNOV_GAIN * buffer_dev) if buffer_dev > 0 else 1.0
        
        # Cap risk factor to avoid NaN or Infinity exploding gradients
        risk_factor = min(risk_factor, 10.0) 
        
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
            'rebuffer': rebuffer_time, 'reward': reward
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