"""
Comprehensive comparison of all methods on all videos.
Compares: PPO V1, V2, V3, V4, Pensieve, BBA, MPC, Random
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
from typing import Optional

PATHS = get_paths()

# Try to import MPC if available
try:
    from src.baselines.mpc import MPC
    HAS_MPC = True
except:
    HAS_MPC = False
    print("⚠ MPC baseline not available")

# Try to import Pensieve environment
try:
    from src.environment.pensieve_env import PensieveEnv
    HAS_PENSIEVE_ENV = True
except:
    HAS_PENSIEVE_ENV = False
    print("⚠ Pensieve environment not available")


class ComprehensiveEvaluator:
    """Evaluate all methods on all videos."""
    
    def __init__(self):
        self.results = []
        self.methods = {}
        
    def load_models(self):
        """Load all available models."""
        print("\n📦 Loading models...\n")
        
        # PPO V4 (newest)
        try:
            v4_path = PATHS['models'] / 'ppo_abr_v4' / 'best_model' / 'best_model'
            if not v4_path.with_suffix('.zip').exists():
                v4_path = PATHS['models'] / 'ppo_abr_v4' / 'final_model'
            self.methods['PPO_V4'] = PPO.load(str(v4_path))
            print("  ✓ PPO V4 loaded (Buffer-Aware Dynamic Reward)")
        except Exception as e:
            print(f"  ⚠ PPO V4 not available: {e}")
        
        # PPO V3
        try:
            v3_path = PATHS['models'] / 'ppo_abr_v3' / 'best_model' / 'best_model'
            if not v3_path.with_suffix('.zip').exists():
                v3_path = PATHS['models'] / 'ppo_abr_v3' / 'final_model'
            self.methods['PPO_V3'] = PPO.load(str(v3_path))
            print("  ✓ PPO V3 loaded")
        except Exception as e:
            print(f"  ⚠ PPO V3 not available: {e}")
        
        # PPO V2
        try:
            v2_path = PATHS['models'] / 'ppo_abr_v2' / 'best_model' / 'best_model'
            if not v2_path.with_suffix('.zip').exists():
                v2_path = PATHS['models'] / 'ppo_abr_v2' / 'final_model'
            self.methods['PPO_V2'] = PPO.load(str(v2_path))
            print("  ✓ PPO V2 loaded")
        except Exception as e:
            print(f"  ⚠ PPO V2 not available: {e}")
        
        # PPO V1
        try:
            v1_path = PATHS['models'] / 'ppo_abr' / 'best_model' / 'best_model'
            if not v1_path.with_suffix('.zip').exists():
                v1_path = PATHS['models'] / 'ppo_abr' / 'final_model'
            self.methods['PPO_V1'] = PPO.load(str(v1_path))
            print("  ✓ PPO V1 loaded")
        except Exception as e:
            print(f"  ⚠ PPO V1 not available: {e}")
        
        # Pensieve (MIT 2017 approach)
        if HAS_PENSIEVE_ENV:
            try:
                pensieve_path = PATHS['models'] / 'pensieve' / 'best_model' / 'best_model'
                if not pensieve_path.with_suffix('.zip').exists():
                    pensieve_path = PATHS['models'] / 'pensieve' / 'final_model'
                self.methods['Pensieve'] = PPO.load(str(pensieve_path))
                print("  ✓ Pensieve loaded (MIT 2017 - A3C replicated with PPO)")
            except Exception as e:
                print(f"  ⚠ Pensieve not available: {e}")
        
        # BBA
        self.methods['BBA'] = BBA([300, 750, 1200, 1850, 2850, 6000])
        print("  ✓ BBA loaded")
        
        # MPC
        if HAS_MPC:
            self.methods['MPC'] = MPC([300, 750, 1200, 1850, 2850, 6000])
            print("  ✓ MPC loaded")
        
        # Random (no model needed)
        self.methods['Random'] = None
        print("  ✓ Random baseline ready")
        
        print(f"\n  Total methods: {len(self.methods)}\n")
    
    def get_available_videos(self):
        """Get list of processed videos."""
        videos = []
        
        # Check VMAF directory
        vmaf_dir = PATHS['vmaf_scores']
        if vmaf_dir.exists():
            for item in vmaf_dir.iterdir():
                if item.is_dir():
                    videos.append(item.name)
        
        # Also check encoded videos
        encoded_dir = PATHS['data_dir'] / 'encoded_videos'
        if encoded_dir.exists():
            for item in encoded_dir.iterdir():
                if item.is_dir() and item.name not in videos:
                    videos.append(item.name)
        
        return sorted(videos)
    
    def evaluate_method_on_video(
        self,
        method_name: str,
        model,
        video_name: str,
        num_episodes: int = 20
    ) -> dict:
        """Evaluate a single method on a single video."""
        
        # Create appropriate environment
        try:
            if method_name == 'Pensieve':
                # Use Pensieve environment (no content features)
                if not HAS_PENSIEVE_ENV:
                    print(f"    ✗ Pensieve env not available for {video_name}")
                    return None
                env = PensieveEnv(
                    trace_dir=str(PATHS['processed_traces']),
                    max_chunks=48,
                    random_seed=42
                )
            else:
                # Use standard ABR environment (with content features)
                env = ABREnv(
                    video_name=video_name,
                    trace_dir=str(PATHS['processed_traces']),
                    vmaf_dir=str(PATHS['vmaf_scores']),
                    siti_dir=str(PATHS['content_features']),
                    max_chunks=48,
                    random_seed=42
                )
        except Exception as e:
            print(f"    ✗ Failed to create env for {video_name}: {e}")
            return None
        
        # Run episodes
        rewards = []
        rebuffers = []
        qualities = []
        switches = []
        bitrates = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            last_action = 0
            ep_switches = 0
            ep_bitrates = []
            last_throughput = 2000.0
            
            while not done:
                # Select action based on method
                if method_name == 'Random':
                    action = env.action_space.sample()
                elif method_name == 'BBA':
                    action = model.select_bitrate(info['buffer_level'])
                elif method_name == 'MPC':
                    action = model.select_bitrate(
                        buffer_level=info['buffer_level'],
                        last_throughput=last_throughput
                    )
                else:  # PPO models (including Pensieve)
                    action, _ = model.predict(obs, deterministic=True)
                
                # Track switches
                if action != last_action:
                    ep_switches += 1
                last_action = action
                
                # Track bitrates
                ep_bitrates.append(env.BITRATE_LEVELS[action])
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                last_throughput = info.get('throughput', last_throughput)
                done = terminated or truncated
            
            rewards.append(episode_reward)
            rebuffers.append(info['total_rebuffer'])
            qualities.append(info['avg_quality'])
            switches.append(ep_switches)
            bitrates.append(np.mean(ep_bitrates))
        
        # Calculate statistics
        result = {
            'method': method_name,
            'video': video_name,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'rebuffer_mean': np.mean(rebuffers),
            'rebuffer_std': np.std(rebuffers),
            'quality_mean': np.mean(qualities),
            'quality_std': np.std(qualities),
            'switches_mean': np.mean(switches),
            'switches_std': np.std(switches),
            'bitrate_mean': np.mean(bitrates),
            'bitrate_std': np.std(bitrates),
            'num_episodes': num_episodes
        }
        
        return result
    
    def run_full_evaluation(self, num_episodes: int = 20):
        """Run evaluation on all methods and videos."""
        
        videos = self.get_available_videos()
        
        if not videos:
            print("✗ No videos found!")
            return
        
        print(f"\n{'='*70}")
        print(f"Running Comprehensive Evaluation")
        print(f"{'='*70}")
        print(f"Videos: {len(videos)} ({', '.join(videos)})")
        print(f"Methods: {len(self.methods)}")
        print(f"Episodes per method per video: {num_episodes}")
        print(f"Total episodes: {len(videos) * len(self.methods) * num_episodes}")
        print(f"{'='*70}\n")
        
        total_tasks = len(videos) * len(self.methods)
        current_task = 0
        
        for video in videos:
            print(f"\n{'─'*70}")
            print(f"📹 Video: {video}")
            print(f"{'─'*70}\n")
            
            for method_name, model in self.methods.items():
                current_task += 1
                progress = (current_task / total_tasks) * 100
                
                print(f"  [{current_task}/{total_tasks}] ({progress:.1f}%) {method_name}...", end=' ')
                
                try:
                    result = self.evaluate_method_on_video(
                        method_name, model, video, num_episodes
                    )
                    
                    if result:
                        self.results.append(result)
                        print(f"✓ Reward: {result['reward_mean']:7.2f}, "
                              f"Rebuffer: {result['rebuffer_mean']:5.2f}s")
                    else:
                        print("✗ Failed")
                        
                except Exception as e:
                    print(f"✗ Error: {str(e)[:50]}")
        
        print(f"\n{'='*70}")
        print(f"✓ Evaluation complete!")
        print(f"  Total results: {len(self.results)}")
        print(f"{'='*70}\n")
    
    def create_summary_tables(self) -> pd.DataFrame:
        """Create summary tables."""
        
        if not self.results:
            print("No results to summarize!")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Overall summary (average across all videos)
        print(f"\n{'='*80}")
        print("Overall Summary (Averaged Across All Videos)")
        print(f"{'='*80}")
        
        summary = df.groupby('method').agg({
            'reward_mean': ['mean', 'std'],
            'rebuffer_mean': ['mean', 'std'],
            'quality_mean': ['mean', 'std'],
            'switches_mean': ['mean', 'std'],
            'bitrate_mean': ['mean', 'std']
        }).round(2)
        
        print(summary)
        print(f"{'='*80}\n")
        
        # Per-video summary
        print(f"\n{'='*80}")
        print("Per-Video Results")
        print(f"{'='*80}\n")
        
        for video in df['video'].unique():
            print(f"📹 {video}:")
            print(f"{'─'*80}")
            
            video_df = df[df['video'] == video].sort_values('reward_mean', ascending=False)
            
            print(f"{'Method':<15} | {'Reward':>12} | {'Rebuffer':>12} | {'Quality':>10} | {'Switches':>10}")
            print(f"{'─'*80}")
            
            for _, row in video_df.iterrows():
                print(f"{row['method']:<15} | "
                      f"{row['reward_mean']:6.2f} ± {row['reward_std']:4.2f} | "
                      f"{row['rebuffer_mean']:5.2f} ± {row['rebuffer_std']:4.2f}s | "
                      f"{row['quality_mean']:5.3f} ± {row['quality_std']:4.3f} | "
                      f"{row['switches_mean']:5.1f} ± {row['switches_std']:3.1f}")
            
            print()
        
        return df
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create comparison plots."""
        
        if df is None or df.empty:
            return
        
        print("\n📊 Creating visualizations...\n")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive ABR Method Comparison', fontsize=16, fontweight='bold')
        
        # Overall averages
        overall = df.groupby('method').agg({
            'reward_mean': 'mean',
            'rebuffer_mean': 'mean',
            'quality_mean': 'mean',
            'switches_mean': 'mean',
            'bitrate_mean': 'mean'
        }).reset_index()
        
        # Sort by reward for consistent ordering
        overall = overall.sort_values('reward_mean', ascending=False)
        
        # 1. Reward comparison
        ax = axes[0, 0]
        bars = ax.bar(overall['method'], overall['reward_mean'], 
                      color=sns.color_palette("husl", len(overall)))
        ax.set_ylabel('Average Reward')
        ax.set_title('Overall Reward (Higher is Better)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Rebuffering comparison
        ax = axes[0, 1]
        bars = ax.bar(overall['method'], overall['rebuffer_mean'],
                      color=sns.color_palette("husl", len(overall)))
        ax.set_ylabel('Average Rebuffering (seconds)')
        ax.set_title('Rebuffering Time (Lower is Better)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Quality comparison
        ax = axes[0, 2]
        bars = ax.bar(overall['method'], overall['quality_mean'],
                      color=sns.color_palette("husl", len(overall)))
        ax.set_ylabel('Average Quality (VMAF/100)')
        ax.set_title('Video Quality (Higher is Better)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Switches comparison
        ax = axes[1, 0]
        bars = ax.bar(overall['method'], overall['switches_mean'],
                      color=sns.color_palette("husl", len(overall)))
        ax.set_ylabel('Average Bitrate Switches')
        ax.set_title('Bitrate Switching Frequency')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 5. Average bitrate
        ax = axes[1, 1]
        bars = ax.bar(overall['method'], overall['bitrate_mean'],
                      color=sns.color_palette("husl", len(overall)))
        ax.set_ylabel('Average Bitrate (Kbps)')
        ax.set_title('Average Selected Bitrate')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 6. Per-video heatmap (Reward)
        ax = axes[1, 2]
        pivot = df.pivot_table(values='reward_mean', index='method', columns='video')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
                    cbar_kws={'label': 'Reward'})
        ax.set_title('Reward by Video (Heatmap)')
        
        plt.tight_layout()
        
        # Save figure
        save_path = PATHS['results'] / 'comprehensive_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot: {save_path}")
        
        # Create DRL methods comparison (PPO variants + Pensieve)
        if any('PPO' in m or 'Pensieve' in m for m in df['method'].unique()):
            self._plot_drl_progression(df)
    
    def _plot_drl_progression(self, df: pd.DataFrame):
        """Plot DRL method progression."""
        
        # Filter DRL methods
        drl_df = df[df['method'].str.contains('PPO|Pensieve')]
        
        if drl_df.empty:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('DRL-Based ABR Evolution: Pensieve → Our PPO (V1→V2→V3→V4)', 
                     fontsize=14, fontweight='bold')
        
        drl_summary = drl_df.groupby('method').agg({
            'reward_mean': 'mean',
            'rebuffer_mean': 'mean',
            'quality_mean': 'mean'
        }).reset_index()
        
        # Sort by logical progression: Pensieve, then V1→V2→V3→V4
        method_order = ['Pensieve', 'PPO_V1', 'PPO_V2', 'PPO_V3', 'PPO_V4']
        drl_summary['method'] = pd.Categorical(
            drl_summary['method'],
            categories=[v for v in method_order if v in drl_summary['method'].values],
            ordered=True
        )
        drl_summary = drl_summary.sort_values('method')
        
        # Reward progression
        axes[0].plot(drl_summary['method'], drl_summary['reward_mean'], 
                    'o-', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('QoE Evolution')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Rebuffering progression
        axes[1].plot(drl_summary['method'], drl_summary['rebuffer_mean'], 
                    'o-', linewidth=2, markersize=8, color='coral')
        axes[1].set_ylabel('Rebuffering (s)')
        axes[1].set_title('Rebuffering Reduction')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Quality progression
        axes[2].plot(drl_summary['method'], drl_summary['quality_mean'], 
                    'o-', linewidth=2, markersize=8, color='seagreen')
        axes[2].set_ylabel('Quality')
        axes[2].set_title('Quality Evolution')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = PATHS['results'] / 'drl_progression.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved DRL progression: {save_path}")
    
    def save_results(self, df: pd.DataFrame):
        """Save results to CSV."""
        
        if df is None or df.empty:
            return
        
        # Save detailed results
        csv_path = PATHS['results'] / 'comprehensive_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Detailed results saved: {csv_path}")
        
        # Save summary
        summary = df.groupby('method').agg({
            'reward_mean': ['mean', 'std'],
            'rebuffer_mean': ['mean', 'std'],
            'quality_mean': ['mean', 'std'],
            'switches_mean': ['mean', 'std']
        }).round(3)
        
        summary_path = PATHS['results'] / 'summary_statistics.csv'
        summary.to_csv(summary_path)
        print(f"✓ Summary statistics saved: {summary_path}")


def main():
    """Main evaluation function."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive ABR evaluation')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Episodes per method per video (default: 20)')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 Comprehensive ABR Method Comparison")
    print("   Including: V1, V2, V3, V4, Pensieve, BBA, MPC, Random")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Load models
    evaluator.load_models()
    
    # Run evaluation
    evaluator.run_full_evaluation(num_episodes=args.episodes)
    
    # Create summaries
    df = evaluator.create_summary_tables()
    
    # Create visualizations
    if not args.skip_plots and df is not None:
        evaluator.create_visualizations(df)
    
    # Save results
    if df is not None:
        evaluator.save_results(df)
    
    print("\n" + "="*80)
    print("✓ Comprehensive evaluation complete!")
    print("="*80)
    print("\nResults saved in: results/")
    print("  - comprehensive_results.csv")
    print("  - summary_statistics.csv")
    print("  - comprehensive_comparison.png")
    print("  - drl_progression.png (Pensieve→V1→V2→V3→V4)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()