"""
ABR (Adaptive Bitrate) Gym Environment for Deep Reinforcement Learning.
Scientific implementation based on Lyapunov Optimization for IEEE TCSVT.

Key Improvements:
1. Replaced heuristic reward weights with Lyapunov-based drift-plus-penalty formulation.
2. Implemented continuous risk factor based on buffer deviation.
3. Standardized smoothness penalty using VMAF metric difference.
"""

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
    
    State Space (18 dims):
        - Throughput history: last 8 chunks (normalized)
        - Buffer level: current buffer occupancy (normalized)
        - Last bitrate: previous quality choice (normalized)
        - SI/TI: content complexity features (normalized)
        - VMAF predictions: predicted quality for next chunk at all 6 bitrate levels
    
    Action Space:
        - Discrete(6): Select bitrate level (0-5)
    
    Reward Function (Lyapunov-Based):
        Maximize: VMAF - (Risk_Factor * Rebuffering) - (Smooth_Penalty * |ΔVMAF|)
        Where Risk_Factor is dynamically derived from buffer queue deviation.
    """
    
    metadata = {'render_modes': ['human']}
    
    # Bitrate levels (Kbps) - Standard DASH ladder
    BITRATE_LEVELS = np.array([300, 750, 1200, 1850, 2850, 6000])
    
    # Video configuration
    CHUNK_DURATION = 4.0  # seconds
    
    # Control Theory Parameters (Lyapunov)
    BUFFER_TARGET = 15.0      # Q_ref: Reference buffer level (seconds)
    BUFFER_MAX = 30.0         # B_max: Maximum buffer capacity
    LYAPUNOV_GAIN = 0.5       # θ: Sensitivity to buffer deviation
    
    # Standard QoE weights (Reference: Pensieve, Comyco)
    REBUF_PENALTY_BASE = 4.3  # Base penalty weight (μ)
    SMOOTH_PENALTY_WEIGHT = 1.0 
    
    def __init__(
        self,
        video_name: str = 'sample1',
        trace_dir: str = 'data/network_traces/processed',
        vmaf_dir: str = 'data/vmaf_scores',
        siti_dir: str = 'data/content_features',
        max_chunks: int = 48,
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
        
        # Load external data
        self._load_traces()
        self._load_vmaf_scores()
        self._load_siti_features()
        
        # Define spaces
        self.action_space = spaces.Discrete(len(self.BITRATE_LEVELS))
        
        # Observation: [Throughput(8), Buffer(1), LastBitrate(1), SI(1), TI(1), VMAF_Preds(6)]
        obs_dim = 8 + 1 + 1 + 2 + 6 
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.current_trace = None
        self.current_trace_idx = 0
        self.chunk_idx = 0
        self.buffer_level = 0.0
        self.last_bitrate_idx = 0
        self.last_quality_metric = 0.0  # Track last VMAF for smoothness
        self.throughput_history = []
        
        # Logging metrics
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
    
    def _load_traces(self):
        """Load network traces from JSON files."""
        trace_files = list(self.trace_dir.glob("*.json"))
        if not trace_files:
            # Fallback for testing without data
            print(f"⚠ No traces found in {self.trace_dir}. Using dummy trace.")
            self.traces = [{'throughput_kbps': [2000]*1000}]
        else:
            self.traces = []
            for trace_file in trace_files:
                with open(trace_file, 'r') as f:
                    self.traces.append(json.load(f))
            print(f"✓ Loaded {len(self.traces)} network traces")
    
    def _load_vmaf_scores(self):
        """Load pre-computed VMAF scores."""
        vmaf_file = self.vmaf_dir / self.video_name / "vmaf_summary.csv"
        if not vmaf_file.exists():
            vmaf_file = self.vmaf_dir / "vmaf_summary.csv"
            
        if vmaf_file.exists():
            import pandas as pd
            df = pd.read_csv(vmaf_file)
            # Filter for current video if file contains multiple
            if 'video' in df.columns:
                df = df[df['video'] == self.video_name]
            self.vmaf_scores = dict(zip(df['bitrate_kbps'], df['vmaf']))
        else:
            print(f"⚠ VMAF file not found. Using default estimates.")
            self.vmaf_scores = {
                300: 40.0, 750: 65.0, 1200: 78.0,
                1850: 85.0, 2850: 90.0, 6000: 95.0
            }
            
        # Normalize VMAF to [0, 1] range for NN input
        self.vmaf_normalized = {k: v/100.0 for k, v in self.vmaf_scores.items()}
    
    def _load_siti_features(self):
        """Load SI/TI complexity features."""
        siti_file = self.siti_dir / f"{self.video_name}_siti.json"
        if siti_file.exists():
            with open(siti_file, 'r') as f:
                data = json.load(f)
                self.si = data.get('mean_si', 50.0)
                self.ti = data.get('mean_ti', 10.0)
        else:
            self.si, self.ti = 50.0, 10.0
            
        self.si_norm = np.clip(self.si / 100.0, 0, 1)
        self.ti_norm = np.clip(self.ti / 50.0, 0, 1)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Randomly select a trace
        self.current_trace_idx = random.randint(0, len(self.traces) - 1)
        self.current_trace = self.traces[self.current_trace_idx]
        
        # Reset state
        self.chunk_idx = 0
        self.buffer_level = 1.0  # Start with small buffer
        self.last_bitrate_idx = 0
        self.last_quality_metric = self.vmaf_scores[self.BITRATE_LEVELS[0]] / 100.0
        
        # Reset history with neutral values
        self.throughput_history = [1.0] * 8 
        
        # Reset metrics
        self.total_rebuffer = 0.0
        self.total_quality = 0.0
        self.total_smooth = 0.0
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """Construct the state vector."""
        # 1. Throughput History (Normalized)
        # Assuming 6000 Kbps is max scale
        tp_obs = np.array(self.throughput_history[-8:], dtype=np.float32)
        
        # 2. Buffer Level (Normalized)
        buf_obs = np.clip(self.buffer_level / self.BUFFER_MAX, 0, 1)
        
        # 3. Last Bitrate (Normalized Index)
        last_br_obs = self.last_bitrate_idx / (len(self.BITRATE_LEVELS) - 1)
        
        # 4. Content Features (SI/TI)
        content_obs = np.array([self.si_norm, self.ti_norm], dtype=np.float32)
        
        # 5. VMAF Predictions (Normalized) for next chunk
        vmaf_obs = np.array([self.vmaf_normalized[br] for br in self.BITRATE_LEVELS], dtype=np.float32)
        
        return np.concatenate([
            tp_obs, 
            [buf_obs], 
            [last_br_obs], 
            content_obs, 
            vmaf_obs
        ]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Apply action (bitrate selection) and simulate network environment."""
        
        # 1. Get chosen bitrate
        bitrate_kbps = self.BITRATE_LEVELS[action]
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        # 2. Simulate Network Throughput
        # Get throughput at current time
        trace_throughput = self.current_trace['throughput_kbps']
        trace_len = len(trace_throughput)
        current_time_idx = int(self.chunk_idx * self.CHUNK_DURATION) % trace_len
        
        # Simple simulation: assume constant throughput for chunk duration
        # (For higher accuracy, one would integrate over the duration)
        available_throughput = trace_throughput[current_time_idx]
        
        # 3. Calculate delays
        download_time = chunk_size_bits / (available_throughput * 1000.0 + 1e-6)
        rebuffer_time = max(0, download_time - self.buffer_level)
        
        # 4. Update Buffer
        self.buffer_level = max(0, self.buffer_level - download_time)
        self.buffer_level += self.CHUNK_DURATION
        self.buffer_level = min(self.buffer_level, self.BUFFER_MAX)
        
        # =================================================================
        #                   SCIENTIFIC REWARD FORMULATION
        #             Based on Lyapunov Drift-plus-Penalty Theory
        # =================================================================
        
        # Metric 1: Quality (VMAF Normalized 0-1)
        current_vmaf = self.vmaf_scores.get(bitrate_kbps, 50.0) / 100.0
        
        # Metric 2: Smoothness (Difference in VMAF)
        # Scientific standard: |q_t - q_{t-1}|
        smoothness_penalty = abs(current_vmaf - self.last_quality_metric)
        
        # Metric 3: Rebuffering with Dynamic Risk Factor
        # We define a virtual queue deviation: Q_dev = Q_target - Q_current
        # Risk grows exponentially as buffer depletes below target.
        
        buffer_deviation = self.BUFFER_TARGET - self.buffer_level
        
        if buffer_deviation > 0:
            # Buffer is below target -> Risk zone
            # Formula: 1 + exp(θ * deviation)
            # This approximates the gradient of a Lyapunov potential function
            risk_factor = 1.0 + np.exp(self.LYAPUNOV_GAIN * buffer_deviation * 0.5)
        else:
            # Buffer is healthy -> Standard penalty
            risk_factor = 1.0
            
        # Weighted Rebuffering Penalty
        # When buffer is low, risk_factor becomes large (e.g., >5.0), effectively
        # forcing the agent to prioritize buffer safety over quality.
        weighted_rebuffer = self.REBUF_PENALTY_BASE * risk_factor * rebuffer_time
        
        # Total Reward Calculation
        reward = current_vmaf \
                 - weighted_rebuffer \
                 - (self.SMOOTH_PENALTY_WEIGHT * smoothness_penalty) \
                 - (0.05 * max(0, buffer_deviation)) # Small linear drift term
        
        # =================================================================
        
        # Update State
        self.throughput_history.append(np.clip(available_throughput / 6000.0, 0, 1))
        self.last_bitrate_idx = action
        self.last_quality_metric = current_vmaf
        self.chunk_idx += 1
        
        # Check Termination
        terminated = self.chunk_idx >= self.max_chunks
        truncated = False
        
        # Update Logs
        self.total_quality += current_vmaf
        self.total_rebuffer += rebuffer_time
        self.total_smooth += smoothness_penalty
        
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'bitrate': bitrate_kbps,
            'throughput': available_throughput,
            'buffer': self.buffer_level,
            'rebuffer': rebuffer_time,
            'risk_factor': risk_factor,
            'reward': reward
        })
        
        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict:
        return {
            'chunk_idx': self.chunk_idx,
            'total_rebuffer': self.total_rebuffer,
            'avg_quality': self.total_quality / max(1, self.chunk_idx),
            'total_smoothness': self.total_smooth
        }

    def render(self):
        pass


# --- Testing Block ---
if __name__ == '__main__':
    print("🔬 Testing Scientific ABR Environment...")
    env = ABREnv(max_chunks=20)
    obs, _ = env.reset()
    
    print(f"Observation Shape: {obs.shape}")
    print(f"Action Space: {env.action_space}")
    
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i}: Buf={info['buffer']:.1f}s | "
              f"Risk={info['risk_factor']:.1f} | "
              f"Rebuf={info['rebuffer']:.2f}s | "
              f"Rew={reward:.2f}")
        
        if done: break
    
    print("✓ Test Complete.")