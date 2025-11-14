"""
Model Predictive Control (MPC) for ABR.
Based on: "A Control-Theoretic Approach for Dynamic Adaptive Video Streaming"
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.environment.abr_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()


class MPC:
    """
    Model Predictive Control for ABR.
    Predicts future throughput and optimizes bitrate selection.
    """
    
    def __init__(self, bitrate_levels, horizon=5):
        self.bitrate_levels = np.array(bitrate_levels)
        self.horizon = horizon  # Look-ahead window
        self.past_throughput = []
        self.past_bandwidth_ests = []
    
    def estimate_throughput(self) -> float:
        """Estimate future throughput using harmonic mean of past observations."""
        if len(self.past_throughput) == 0:
            return self.bitrate_levels[2]  # Default: middle bitrate
        
        # Harmonic mean (conservative estimate)
        harmonic_mean = len(self.past_throughput) / np.sum(1.0 / (np.array(self.past_throughput) + 1e-6))
        return harmonic_mean
    
    def predict_qoe(self, bitrate_idx: int, buffer_level: float, estimated_throughput: float) -> float:
        """
        Predict QoE for a given bitrate choice.
        
        Args:
            bitrate_idx: Bitrate index to evaluate
            buffer_level: Current buffer level
            estimated_throughput: Estimated future throughput
            
        Returns:
            Predicted QoE score
        """
        bitrate = self.bitrate_levels[bitrate_idx]
        
        # Estimate download time
        chunk_size = bitrate * 4  # 4 second chunks
        download_time = chunk_size / (estimated_throughput + 1e-6)
        
        # Predict rebuffering
        rebuffer_time = max(0, download_time - buffer_level)
        
        # Simple QoE model
        quality = bitrate / 6000.0  # Normalize
        rebuffer_penalty = 4.3 * rebuffer_time
        
        qoe = quality - rebuffer_penalty
        
        return qoe
    
    def select_bitrate(self, buffer_level: float, last_throughput: float) -> int:
        """
        Select optimal bitrate using MPC.
        
        Args:
            buffer_level: Current buffer in seconds
            last_throughput: Most recent throughput observation (Kbps)
            
        Returns:
            Selected bitrate index
        """
        # Update history
        if last_throughput > 0:
            self.past_throughput.append(last_throughput)
            if len(self.past_throughput) > 5:
                self.past_throughput.pop(0)
        
        # Estimate future throughput
        estimated_throughput = self.estimate_throughput()
        
        # Evaluate all bitrate options
        best_qoe = -float('inf')
        best_idx = 0
        
        for idx in range(len(self.bitrate_levels)):
            qoe = self.predict_qoe(idx, buffer_level, estimated_throughput)
            
            if qoe > best_qoe:
                best_qoe = qoe
                best_idx = idx
        
        return best_idx


def evaluate_mpc(num_episodes: int = 10):
    """Evaluate MPC algorithm."""
    print("\n" + "="*60)
    print("📊 Evaluating MPC (Model Predictive Control)")
    print("="*60 + "\n")
    
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48,
        random_seed=42
    )
    
    mpc = MPC(env.BITRATE_LEVELS, horizon=5)
    
    rewards, rebuffers, qualities, switches = [], [], [], []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        last_action = 0
        ep_switches = 0
        last_throughput = 2000.0  # Initialize
        
        while not done:
            # MPC decision
            action = mpc.select_bitrate(
                buffer_level=info['buffer_level'],
                last_throughput=last_throughput
            )
            
            if action != last_action:
                ep_switches += 1
            last_action = action
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            last_throughput = info.get('throughput', last_throughput)
            done = terminated or truncated
        
        rewards.append(episode_reward)
        rebuffers.append(info['total_rebuffer'])
        qualities.append(info['avg_quality'])
        switches.append(ep_switches)
        
        print(f"Episode {ep+1:2d}/{num_episodes}: "
              f"Reward={episode_reward:7.2f}, "
              f"Rebuffer={info['total_rebuffer']:5.2f}s, "
              f"Quality={info['avg_quality']:.3f}, "
              f"Switches={ep_switches:2d}")
    
    print(f"\n{'='*60}")
    print("MPC Results:")
    print(f"{'='*60}")
    print(f"  Avg Reward:     {np.mean(rewards):7.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg Rebuffer:   {np.mean(rebuffers):5.2f}s ± {np.std(rebuffers):.2f}s")
    print(f"  Avg Quality:    {np.mean(qualities):.3f} ± {np.std(qualities):.3f}")
    print(f"  Avg Switches:   {np.mean(switches):5.1f} ± {np.std(switches):.1f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    
    evaluate_mpc(num_episodes=args.episodes)