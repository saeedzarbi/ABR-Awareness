"""
Buffer-Based ABR (BBA) baseline algorithm.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.abr_env import ABREnv
from configs.paths import get_paths
import numpy as np

PATHS = get_paths()


class BBA:
    """
    Buffer-Based ABR algorithm.
    Selects bitrate based on current buffer level.
    
    Reference: "A Buffer-Based Approach to Rate Adaptation" (Huang et al., 2014)
    """
    
    def __init__(self, bitrate_levels):
        self.bitrate_levels = np.array(bitrate_levels)
        
        # Buffer thresholds (seconds)
        self.reservoir = 5.0   # Below this: minimum bitrate
        self.cushion = 20.0    # Above this: maximum bitrate
    
    def select_bitrate(self, buffer_level: float) -> int:
        """
        Select bitrate based on buffer level.
        
        Args:
            buffer_level: Current buffer in seconds
            
        Returns:
            Bitrate index (0-5)
        """
        if buffer_level <= self.reservoir:
            # Low buffer: minimum bitrate
            return 0
        elif buffer_level >= self.cushion:
            # High buffer: maximum bitrate
            return len(self.bitrate_levels) - 1
        else:
            # Linear mapping between reservoir and cushion
            ratio = (buffer_level - self.reservoir) / (self.cushion - self.reservoir)
            bitrate_idx = int(ratio * (len(self.bitrate_levels) - 1))
            return bitrate_idx


def evaluate_bba(num_episodes: int = 10):
    """Evaluate BBA algorithm."""
    print("\n" + "="*60)
    print("📊 Evaluating BBA (Buffer-Based ABR)")
    print("="*60 + "\n")
    
    print("BBA Algorithm:")
    print("  - Reservoir: 5s (below → min bitrate)")
    print("  - Cushion: 20s (above → max bitrate)")
    print("  - In between → linear interpolation\n")
    
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48,
        random_seed=42
    )
    
    bba = BBA(env.BITRATE_LEVELS)
    
    rewards = []
    rebuffers = []
    qualities = []
    bitrate_switches = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        last_action = 0
        switches = 0
        
        while not done:
            # BBA decision based on buffer
            action = bba.select_bitrate(info['buffer_level'])
            
            # Count switches
            if action != last_action:
                switches += 1
            last_action = action
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        rebuffers.append(info['total_rebuffer'])
        qualities.append(info['avg_quality'])
        bitrate_switches.append(switches)
        
        print(f"Episode {ep+1:2d}/{num_episodes}: "
              f"Reward={episode_reward:7.2f}, "
              f"Rebuffer={info['total_rebuffer']:5.2f}s, "
              f"Quality={info['avg_quality']:.3f}, "
              f"Switches={switches:2d}")
    
    print(f"\n{'='*60}")
    print("BBA Results:")
    print(f"{'='*60}")
    print(f"  Avg Reward:     {np.mean(rewards):7.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg Rebuffer:   {np.mean(rebuffers):5.2f}s ± {np.std(rebuffers):.2f}s")
    print(f"  Avg Quality:    {np.mean(qualities):.3f} ± {np.std(qualities):.3f}")
    print(f"  Avg Switches:   {np.mean(bitrate_switches):5.1f} ± {np.std(bitrate_switches):.1f}")
    print(f"{'='*60}\n")
    
    return {
        'rewards': rewards,
        'rebuffers': rebuffers,
        'qualities': qualities,
        'switches': bitrate_switches
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate BBA baseline')
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes (default: 10)'
    )
    
    args = parser.parse_args()
    
    evaluate_bba(num_episodes=args.episodes)