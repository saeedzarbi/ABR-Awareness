"""
Deep analysis of PPO V3 behavior to find improvement opportunities.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from src.baselines.bba import BBA
from configs.paths import get_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

PATHS = get_paths()


class DeepAnalyzer:
    """Deep analysis of ABR agent behavior."""
    
    def __init__(self, model_path: str, video_name: str = 'sample1'):
        self.model = PPO.load(model_path)
        self.video_name = video_name
        self.env = ABREnv(
            video_name=video_name,
            trace_dir=str(PATHS['processed_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=48,
            random_seed=42
        )
        self.bitrate_levels = self.env.BITRATE_LEVELS
        
    def analyze_action_distribution(self, num_episodes: int = 50):
        """Analyze which bitrates are selected and when."""
        
        print("\n" + "="*70)
        print("📊 Action Distribution Analysis")
        print("="*70 + "\n")
        
        all_actions = []
        all_buffers = []
        all_throughputs = []
        all_rewards = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                
                all_actions.append(action)
                all_buffers.append(info.get('buffer_level', 0))
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                all_throughputs.append(info.get('throughput', 0))
                all_rewards.append(reward)
                done = terminated or truncated
        
        # Convert to arrays
        actions = np.array(all_actions)
        buffers = np.array(all_buffers)
        throughputs = np.array(all_throughputs)
        rewards = np.array(all_rewards)
        
        # Action distribution
        print("Action Distribution:")
        print("-" * 70)
        for i, br in enumerate(self.bitrate_levels):
            count = np.sum(actions == i)
            pct = (count / len(actions)) * 100
            bar = "█" * int(pct / 2)
            print(f"  {br:4d} Kbps: {count:5d} ({pct:5.1f}%) {bar}")
        
        print(f"\n  Most selected: {self.bitrate_levels[np.argmax(np.bincount(actions))]} Kbps")
        print(f"  Average bitrate: {np.mean([self.bitrate_levels[a] for a in actions]):.0f} Kbps")
        print(f"  Median bitrate: {np.median([self.bitrate_levels[a] for a in actions]):.0f} Kbps")
        
        # Analyze by buffer level
        print("\n" + "="*70)
        print("Action Distribution by Buffer Level:")
        print("="*70 + "\n")
        
        buffer_ranges = [
            (0, 5, "Low (0-5s)"),
            (5, 10, "Medium (5-10s)"),
            (10, 15, "Good (10-15s)"),
            (15, 30, "High (15-30s)")
        ]
        
        for low, high, label in buffer_ranges:
            mask = (buffers >= low) & (buffers < high)
            if np.sum(mask) > 0:
                actions_in_range = actions[mask]
                avg_bitrate = np.mean([self.bitrate_levels[a] for a in actions_in_range])
                print(f"{label:20s}: {np.sum(mask):5d} decisions, Avg bitrate: {avg_bitrate:6.0f} Kbps")
        
        # Analyze by throughput
        print("\n" + "="*70)
        print("Action Distribution by Network Throughput:")
        print("="*70 + "\n")
        
        throughput_ranges = [
            (0, 500, "Very Low (<500 Kbps)"),
            (500, 1000, "Low (500-1000 Kbps)"),
            (1000, 2000, "Medium (1-2 Mbps)"),
            (2000, 4000, "Good (2-4 Mbps)"),
            (4000, 10000, "High (>4 Mbps)")
        ]
        
        for low, high, label in throughput_ranges:
            mask = (throughputs >= low) & (throughputs < high)
            if np.sum(mask) > 0:
                actions_in_range = actions[mask]
                avg_bitrate = np.mean([self.bitrate_levels[a] for a in actions_in_range])
                print(f"{label:25s}: {np.sum(mask):5d} decisions, Avg bitrate: {avg_bitrate:6.0f} Kbps")
        
        return {
            'actions': actions,
            'buffers': buffers,
            'throughputs': throughputs,
            'rewards': rewards
        }
    
    def analyze_failure_cases(self, num_episodes: int = 100):
        """Find episodes with high rebuffering to understand failures."""
        
        print("\n" + "="*70)
        print("🔍 Failure Case Analysis")
        print("="*70 + "\n")
        
        episodes = []
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            
            episode_data = {
                'actions': [],
                'buffers': [],
                'throughputs': [],
                'rebuffers': [],
                'rewards': []
            }
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                episode_data['actions'].append(action)
                episode_data['buffers'].append(info.get('buffer_level', 0))
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_data['throughputs'].append(info.get('throughput', 0))
                episode_data['rebuffers'].append(info.get('rebuffer', 0))
                episode_data['rewards'].append(reward)
                
                done = terminated or truncated
            
            episodes.append({
                'total_rebuffer': info['total_rebuffer'],
                'total_reward': sum(episode_data['rewards']),
                'avg_quality': info['avg_quality'],
                'data': episode_data
            })
        
        # Sort by rebuffering
        episodes.sort(key=lambda x: x['total_rebuffer'], reverse=True)
        
        # Analyze worst cases
        print("Top 10 Worst Episodes (by rebuffering):")
        print("-" * 70)
        print(f"{'Rank':<6} {'Rebuffer':>10} {'Reward':>10} {'Quality':>10} {'Issue':<30}")
        print("-" * 70)
        
        for i, ep in enumerate(episodes[:10], 1):
            # Diagnose issue
            avg_bitrate = np.mean([self.bitrate_levels[a] for a in ep['data']['actions']])
            avg_throughput = np.mean(ep['data']['throughputs'])
            
            if avg_bitrate > avg_throughput * 0.8:
                issue = "Bitrate too high for network"
            elif np.min(ep['data']['buffers']) < 2.0:
                issue = "Buffer depletion"
            elif np.std(ep['data']['actions']) < 0.5:
                issue = "Not adaptive (stuck)"
            else:
                issue = "Network variability"
            
            print(f"{i:<6} {ep['total_rebuffer']:>9.2f}s {ep['total_reward']:>10.2f} "
                  f"{ep['avg_quality']:>10.3f} {issue:<30}")
        
        print("\n" + "="*70)
        print("Success Rate:")
        print("-" * 70)
        
        excellent = len([e for e in episodes if e['total_rebuffer'] < 1.0])
        good = len([e for e in episodes if 1.0 <= e['total_rebuffer'] < 3.0])
        fair = len([e for e in episodes if 3.0 <= e['total_rebuffer'] < 10.0])
        poor = len([e for e in episodes if e['total_rebuffer'] >= 10.0])
        
        print(f"  Excellent (<1s rebuffer):    {excellent:3d} ({excellent/num_episodes*100:.1f}%)")
        print(f"  Good (1-3s rebuffer):        {good:3d} ({good/num_episodes*100:.1f}%)")
        print(f"  Fair (3-10s rebuffer):       {fair:3d} ({fair/num_episodes*100:.1f}%)")
        print(f"  Poor (>10s rebuffer):        {poor:3d} ({poor/num_episodes*100:.1f}%)")
        
        return episodes
    
    def compare_with_oracle(self, num_episodes: int = 20):
        """Compare with theoretical optimal policy."""
        
        print("\n" + "="*70)
        print("🎯 Oracle Comparison (Theoretical Upper Bound)")
        print("="*70 + "\n")
        
        ppo_rewards = []
        oracle_rewards = []
        
        for ep in range(num_episodes):
            # PPO performance
            obs, info = self.env.reset()
            done = False
            ppo_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ppo_reward += reward
                done = terminated or truncated
            
            ppo_rewards.append(ppo_reward)
            
            # Oracle: perfect future knowledge
            obs, info = self.env.reset()
            done = False
            oracle_reward = 0
            
            # Simple oracle: always select bitrate that fits in throughput
            while not done:
                # Assume we know exact throughput
                current_throughput = info.get('throughput', 2000)
                buffer = info.get('buffer_level', 0)
                
                # Conservative oracle: select highest bitrate that won't rebuffer
                best_action = 0
                for i, br in enumerate(self.bitrate_levels):
                    chunk_time = (br * 4) / current_throughput  # 4s chunks
                    if chunk_time < buffer + 3:  # 3s safety margin
                        best_action = i
                
                obs, reward, terminated, truncated, info = self.env.step(best_action)
                oracle_reward += reward
                done = terminated or truncated
            
            oracle_rewards.append(oracle_reward)
        
        print(f"PPO V3 Average Reward:    {np.mean(ppo_rewards):8.2f} ± {np.std(ppo_rewards):.2f}")
        print(f"Oracle Average Reward:    {np.mean(oracle_rewards):8.2f} ± {np.std(oracle_rewards):.2f}")
        print(f"Gap:                      {np.mean(oracle_rewards) - np.mean(ppo_rewards):8.2f}")
        print(f"PPO achieves:             {np.mean(ppo_rewards)/np.mean(oracle_rewards)*100:6.1f}% of oracle")
        
    def analyze_reward_components(self, num_episodes: int = 50):
        """Break down reward into components."""
        
        print("\n" + "="*70)
        print("💰 Reward Component Analysis")
        print("="*70 + "\n")
        
        quality_sum = 0
        rebuffer_sum = 0
        smooth_sum = 0
        buffer_penalty_sum = 0
        total_steps = 0
        
        for ep in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Approximate component extraction
                # This is rough - actual values are inside env
                total_steps += 1
                
                done = terminated or truncated
        
        print("Average Reward Components (per step):")
        print("-" * 70)
        print("  Note: Run with modified env to get exact breakdown")
        print("  Current analysis shows aggregate statistics")
        
    def plot_behavior_analysis(self, data: dict):
        """Create detailed behavior plots."""
        
        print("\n📊 Creating behavior analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('PPO V3 Behavior Analysis', fontsize=16, fontweight='bold')
        
        actions = data['actions']
        buffers = data['buffers']
        throughputs = data['throughputs']
        rewards = data['rewards']
        bitrates = [self.bitrate_levels[a] for a in actions]
        
        # 1. Action histogram
        ax = axes[0, 0]
        ax.hist(bitrates, bins=len(self.bitrate_levels), edgecolor='black')
        ax.set_xlabel('Bitrate (Kbps)')
        ax.set_ylabel('Frequency')
        ax.set_title('Bitrate Selection Distribution')
        ax.grid(alpha=0.3)
        
        # 2. Buffer vs Bitrate
        ax = axes[0, 1]
        scatter = ax.scatter(buffers, bitrates, c=rewards, cmap='RdYlGn', alpha=0.5, s=10)
        ax.set_xlabel('Buffer Level (s)')
        ax.set_ylabel('Selected Bitrate (Kbps)')
        ax.set_title('Buffer Level vs Bitrate Selection')
        plt.colorbar(scatter, ax=ax, label='Reward')
        ax.grid(alpha=0.3)
        
        # 3. Throughput vs Bitrate
        ax = axes[0, 2]
        ax.scatter(throughputs, bitrates, alpha=0.3, s=10)
        ax.plot([0, max(throughputs)], [0, max(throughputs)], 'r--', label='Bitrate = Throughput')
        ax.set_xlabel('Network Throughput (Kbps)')
        ax.set_ylabel('Selected Bitrate (Kbps)')
        ax.set_title('Throughput vs Bitrate Selection')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Reward distribution
        ax = axes[1, 0]
        ax.hist(rewards, bins=50, edgecolor='black')
        ax.axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 5. Buffer level distribution
        ax = axes[1, 1]
        ax.hist(buffers, bins=30, edgecolor='black')
        ax.axvline(5, color='r', linestyle='--', label='Danger threshold')
        ax.axvline(15, color='g', linestyle='--', label='Target')
        ax.set_xlabel('Buffer Level (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Buffer Level Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 6. Efficiency: Bitrate/Throughput ratio
        ax = axes[1, 2]
        efficiency = np.array(bitrates) / (np.array(throughputs) + 1)
        efficiency = np.clip(efficiency, 0, 3)  # Clip outliers
        ax.hist(efficiency, bins=50, edgecolor='black')
        ax.axvline(1.0, color='r', linestyle='--', label='Perfect match')
        ax.set_xlabel('Bitrate / Throughput Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Bandwidth Utilization Efficiency')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = PATHS['results'] / 'behavior_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        
        plt.close()


def main():
    """Run deep analysis."""
    
    print("\n" + "="*80)
    print("🔬 Deep Analysis of PPO V3")
    print("="*80)
    
    # Load model
    model_path = PATHS['models'] / 'ppo_abr_v3' / 'best_model' / 'best_model'
    
    if not model_path.with_suffix('.zip').exists():
        model_path = PATHS['models'] / 'ppo_abr_v3' / 'final_model'
    
    print(f"\nModel: {model_path}")
    
    analyzer = DeepAnalyzer(str(model_path), video_name='sample1')
    
    # 1. Action distribution
    data = analyzer.analyze_action_distribution(num_episodes=100)
    
    # 2. Failure analysis
    episodes = analyzer.analyze_failure_cases(num_episodes=100)
    
    # 3. Oracle comparison
    analyzer.compare_with_oracle(num_episodes=50)
    
    # 4. Plots
    analyzer.plot_behavior_analysis(data)
    
    print("\n" + "="*80)
    print("✓ Deep analysis complete!")
    print("="*80)
    print("\nKey files:")
    print("  - results/behavior_analysis.png")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()