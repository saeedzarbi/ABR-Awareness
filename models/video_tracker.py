import numpy as np
import pandas as pd
from collections import defaultdict
import json

class PerVideoTracker:
    """Track evaluation results per video"""
    
    def __init__(self):
        self.results = defaultdict(lambda: {
            'rewards': [],
            'rebuffering': [],
            'bitrates': []
        })
    
    def add_episode(self, video_name: str, reward: float, 
                    rebuffering: float, avg_bitrate: float):
        """Add one episode result"""
        self.results[video_name]['rewards'].append(reward)
        self.results[video_name]['rebuffering'].append(rebuffering)
        self.results[video_name]['bitrates'].append(avg_bitrate)
    
    def get_summary_df(self):
        """Get summary as DataFrame"""
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
            })
        df = pd.DataFrame(summary)
        return df.sort_values('mean_reward', ascending=False)
    
    def print_summary(self):
        """Print nicely formatted table"""
        df = self.get_summary_df()
        
        print("\n" + "="*100)
        print("📊 PER-VIDEO BREAKDOWN")
        print("="*100)
        
        print(f"{'Video':<15} {'Episodes':>8} {'Reward':>12} {'Rebuffer(s)':>12} {'Bitrate(kbps)':>15}")
        print("-"*100)
        
        for _, row in df.iterrows():
            print(f"{row['video']:<15} "
                  f"{row['episodes']:>8} "
                  f"{row['mean_reward']:>+11.2f} "
                  f"{row['mean_rebuffer']:>12.2f} "
                  f"{row['mean_bitrate']:>15.0f}")
        
        print("="*100)
        
        # Overall
        all_rewards = [r for v in self.results.values() for r in v['rewards']]
        all_rebuffers = [r for v in self.results.values() for r in v['rebuffering']]
        all_bitrates = [b for v in self.results.values() for b in v['bitrates']]
        
        print(f"\n📈 OVERALL:")
        print(f"   Mean Reward:      {np.mean(all_rewards):+.2f} ± {np.std(all_rewards):.2f}")
        print(f"   Mean Rebuffering: {np.mean(all_rebuffers):.2f}s ± {np.std(all_rebuffers):.2f}s")
        print(f"   Mean Bitrate:     {np.mean(all_bitrates):.0f} ± {np.std(all_bitrates):.0f} kbps")
        print("="*100 + "\n")
    
    def save_to_json(self, filename: str):
        """Save detailed results"""
        output = {}
        for video, data in self.results.items():
            output[video] = {
                'rewards': [float(x) for x in data['rewards']],
                'rebuffering': [float(x) for x in data['rebuffering']],
                'bitrates': [float(x) for x in data['bitrates']],
                'summary': {
                    'mean_reward': float(np.mean(data['rewards'])),
                    'mean_rebuffer': float(np.mean(data['rebuffering'])),
                    'mean_bitrate': float(np.mean(data['bitrates']))
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"💾 Saved: {filename}")
