"""
Training Logger با Real-time Monitoring
"""

import json
import time
from pathlib import Path
from collections import deque
import numpy as np


class TrainingLogger:
    """
    Log training progress با real-time updates
    """
    
    def __init__(self, log_dir='results/logs', run_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if run_name is None:
            run_name = f"run_{int(time.time())}"
        
        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}.jsonl"
        self.stats_file = self.log_dir / f"{run_name}_stats.json"
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_rebuffers = []
        self.episode_bitrates = []
        
        self.update_stats = []
        self.start_time = time.time()
        
        # Moving averages
        self.reward_ma = deque(maxlen=100)
        self.rebuffer_ma = deque(maxlen=100)
        
        print(f"📊 Logging to: {self.log_file}")
    
    def log_episode(self, reward, length, rebuffer=0, avg_bitrate=0):
        """Log episode results"""
        self.episode_rewards.append(float(reward))
        self.episode_lengths.append(int(length))
        self.episode_rebuffers.append(float(rebuffer))
        self.episode_bitrates.append(float(avg_bitrate))
        
        self.reward_ma.append(float(reward))
        self.rebuffer_ma.append(float(rebuffer))
    
    def log_update(self, update_num, timesteps, stats):
        """Log training update"""
        
        # Compute averages
        recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
        recent_rebuffers = self.episode_rebuffers[-20:] if len(self.episode_rebuffers) >= 20 else self.episode_rebuffers
        recent_bitrates = self.episode_bitrates[-20:] if len(self.episode_bitrates) >= 20 else self.episode_bitrates
        
        log_entry = {
            'update': int(update_num),
            'timesteps': int(timesteps),
            'time_elapsed': float(time.time() - self.start_time),
            
            # Episode stats
            'total_episodes': len(self.episode_rewards),
            'avg_reward_20': float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            'avg_reward_100': float(np.mean(self.reward_ma)) if self.reward_ma else 0.0,
            'avg_rebuffer_20': float(np.mean(recent_rebuffers)) if recent_rebuffers else 0.0,
            'avg_bitrate_20': float(np.mean(recent_bitrates)) if recent_bitrates else 0.0,
            
            # Training stats
            'policy_loss': float(stats.get('policy_loss', 0)),
            'value_loss': float(stats.get('value_loss', 0)),
            'entropy': float(stats.get('entropy', 0)),
        }
        
        self.update_stats.append(log_entry)
        
        # Append to JSONL file (one line per update)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return log_entry
    
    def print_update(self, log_entry):
        """Print formatted update"""
        print(f"Update {log_entry['update']:4d} | "
              f"Steps: {log_entry['timesteps']:7d} | "
              f"Episodes: {log_entry['total_episodes']:4d} | "
              f"Reward(20): {log_entry['avg_reward_20']:+7.2f} | "
              f"Reward(100): {log_entry['avg_reward_100']:+7.2f} | "
              f"Rebuffer: {log_entry['avg_rebuffer_20']:5.1f}s | "
              f"Bitrate: {log_entry['avg_bitrate_20']:4.0f}kbps")
    
    def save_summary(self):
        """Save summary statistics"""
        summary = {
            'run_name': self.run_name,
            'total_episodes': len(self.episode_rewards),
            'total_updates': len(self.update_stats),
            'total_time': float(time.time() - self.start_time),
            
            'final_reward_avg': float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else 0.0,
            'best_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0,
            'worst_reward': float(np.min(self.episode_rewards)) if self.episode_rewards else 0.0,
            
            'all_episode_rewards': [float(r) for r in self.episode_rewards[-1000:]],  # Last 1000
            'all_episode_rebuffers': [float(r) for r in self.episode_rebuffers[-1000:]],
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


if __name__ == '__main__':
    print("Logger module loaded!")
