"""
models/video_tracker.py
=======================
Track and analyze evaluation results per video
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Optional
import json


class PerVideoTracker:
    """
    Track evaluation results separately for each video
    
    Usage:
        tracker = PerVideoTracker()
        
        # During evaluation loop:
        for episode in episodes:
            video_name = env.get_video_name()
            ... run episode ...
            tracker.add_episode(video_name, reward, rebuffering, bitrate)
        
        # Print summary:
        tracker.print_summary()
        
        # Save to file:
        tracker.save_to_json('results.json')
    """
    
    def __init__(self):
        """Initialize empty tracker"""
        self.results = defaultdict(lambda: {
            'rewards': [],
            'rebuffering': [],
            'bitrates': []
        })
    
    def add_episode(self, 
                    video_name: str,
                    reward: float,
                    rebuffering: float,
                    avg_bitrate: float):
        """
        Add results from one episode
        
        Args:
            video_name: Name of video (e.g., 'sports', 'animation')
            reward: Episode total reward
            rebuffering: Total rebuffering time (seconds)
            avg_bitrate: Average bitrate (kbps)
        """
        self.results[video_name]['rewards'].append(float(reward))
        self.results[video_name]['rebuffering'].append(float(rebuffering))
        self.results[video_name]['bitrates'].append(float(avg_bitrate))
    
    def get_summary_df(self) -> pd.DataFrame:
        """
        Get summary statistics as pandas DataFrame
        
        Returns:
            DataFrame with columns: video, episodes, mean_reward, std_reward,
                                   mean_rebuffer, std_rebuffer, mean_bitrate
        """
        if not self.results:
            return pd.DataFrame()
        
        summary = []
        for video, data in self.results.items():
            summary.append({
                'video': video,
                'episodes': len(data['rewards']),
                'mean_reward': np.mean(data['rewards']),
                'std_reward': np.std(data['rewards']),
                'mean_rebuffer': np.mean(data['rebuffering']),
                'std_rebuffer': np.std(data['rebuffering']),
                'mean_bitrate': np.mean(data['bitrates']),
                'std_bitrate': np.std(data['bitrates'])
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('mean_reward', ascending=False)
        return df
    
    def print_summary(self, title: Optional[str] = None):
        """
        Print nicely formatted summary table
        
        Args:
            title: Optional title for the table
        """
        df = self.get_summary_df()
        
        if df.empty:
            print("\n⚠️  No results to display")
            return
        
        print("\n" + "="*100)
        if title:
            print(f"📊 {title}")
        else:
            print("📊 PER-VIDEO BREAKDOWN")
        print("="*100)
        
        # Header
        print(f"{'Video':<15} {'Episodes':>8} {'Reward':>12} {'Rebuffer(s)':>12} {'Bitrate(kbps)':>15}")
        print("-"*100)
        
        # Data rows
        for _, row in df.iterrows():
            print(f"{row['video']:<15} "
                  f"{row['episodes']:>8} "
                  f"{row['mean_reward']:>+11.2f} "
                  f"{row['mean_rebuffer']:>12.2f} "
                  f"{row['mean_bitrate']:>15.0f}")
        
        print("="*100)
        
        # Overall statistics
        all_rewards = []
        all_rebuffers = []
        all_bitrates = []
        
        for data in self.results.values():
            all_rewards.extend(data['rewards'])
            all_rebuffers.extend(data['rebuffering'])
            all_bitrates.extend(data['bitrates'])
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Mean Reward:      {np.mean(all_rewards):+.2f} ± {np.std(all_rewards):.2f}")
        print(f"   Mean Rebuffering: {np.mean(all_rebuffers):.2f}s ± {np.std(all_rebuffers):.2f}s")
        print(f"   Mean Bitrate:     {np.mean(all_bitrates):.0f} ± {np.std(all_bitrates):.0f} kbps")
        print("="*100 + "\n")
    
    def save_to_json(self, filename: str):
        """
        Save detailed results to JSON file
        
        Args:
            filename: Output filename (e.g., 'results/per_video_results.json')
        """
        output = {}
        
        for video, data in self.results.items():
            output[video] = {
                'rewards': [float(x) for x in data['rewards']],
                'rebuffering': [float(x) for x in data['rebuffering']],
                'bitrates': [float(x) for x in data['bitrates']],
                'summary': {
                    'episodes': len(data['rewards']),
                    'mean_reward': float(np.mean(data['rewards'])),
                    'std_reward': float(np.std(data['rewards'])),
                    'mean_rebuffer': float(np.mean(data['rebuffering'])),
                    'std_rebuffer': float(np.std(data['rebuffering'])),
                    'mean_bitrate': float(np.mean(data['bitrates'])),
                    'std_bitrate': float(np.std(data['bitrates']))
                }
            }
        
        # Overall statistics
        all_rewards = [r for v in self.results.values() for r in v['rewards']]
        all_rebuffers = [r for v in self.results.values() for r in v['rebuffering']]
        all_bitrates = [b for v in self.results.values() for b in v['bitrates']]
        
        output['_overall'] = {
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'mean_rebuffer': float(np.mean(all_rebuffers)),
            'std_rebuffer': float(np.std(all_rebuffers)),
            'mean_bitrate': float(np.mean(all_bitrates)),
            'std_bitrate': float(np.std(all_bitrates))
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Detailed results saved: {filename}")
    
    def save_to_csv(self, filename: str):
        """
        Save summary to CSV file
        
        Args:
            filename: Output filename (e.g., 'results/per_video_summary.csv')
        """
        df = self.get_summary_df()
        df.to_csv(filename, index=False)
        print(f"💾 Summary saved: {filename}")
    
    def get_overall_stats(self) -> Dict[str, float]:
        """
        Get overall statistics as dictionary
        
        Returns:
            Dictionary with mean_reward, mean_rebuffer, mean_bitrate
        """
        all_rewards = [r for v in self.results.values() for r in v['rewards']]
        all_rebuffers = [r for v in self.results.values() for r in v['rebuffering']]
        all_bitrates = [b for v in self.results.values() for b in v['bitrates']]
        
        return {
            'mean_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'std_reward': float(np.std(all_rewards)) if all_rewards else 0.0,
            'mean_rebuffer': float(np.mean(all_rebuffers)) if all_rebuffers else 0.0,
            'std_rebuffer': float(np.std(all_rebuffers)) if all_rebuffers else 0.0,
            'mean_bitrate': float(np.mean(all_bitrates)) if all_bitrates else 0.0,
            'std_bitrate': float(np.std(all_bitrates)) if all_bitrates else 0.0,
            'total_episodes': len(all_rewards)
        }


# ============================================
# Test
# ============================================
if __name__ == '__main__':
    print("="*80)
    print("Testing PerVideoTracker")
    print("="*80)
    
    # Create tracker
    tracker = PerVideoTracker()
    
    # Simulate some episodes
    print("\n📝 Adding simulated episodes...")
    
    # Sports video (5 episodes)
    for i in range(5):
        tracker.add_episode('sports', 
                          reward=105.0 + np.random.randn()*5,
                          rebuffering=1.0 + np.random.rand()*0.5,
                          avg_bitrate=1200 + np.random.randn()*100)
    
    # Animation video (4 episodes)
    for i in range(4):
        tracker.add_episode('animation',
                          reward=108.0 + np.random.randn()*5,
                          rebuffering=0.8 + np.random.rand()*0.3,
                          avg_bitrate=1250 + np.random.randn()*100)
    
    # News video (3 episodes)
    for i in range(3):
        tracker.add_episode('news',
                          reward=102.0 + np.random.randn()*5,
                          rebuffering=1.2 + np.random.rand()*0.4,
                          avg_bitrate=1100 + np.random.randn()*100)
    
    # Nature video (4 episodes)
    for i in range(4):
        tracker.add_episode('nature',
                          reward=106.0 + np.random.randn()*5,
                          rebuffering=0.9 + np.random.rand()*0.4,
                          avg_bitrate=1180 + np.random.randn()*100)
    
    # Print summary
    tracker.print_summary(title="Test Results")
    
    # Test DataFrame
    print("\n📊 Testing DataFrame export:")
    df = tracker.get_summary_df()
    print(df.head())
    
    # Test overall stats
    print("\n📊 Testing overall stats:")
    stats = tracker.get_overall_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}")
    
    # Test save functions
    print("\n💾 Testing save functions:")
    tracker.save_to_json('test_per_video.json')
    tracker.save_to_csv('test_per_video.csv')
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)