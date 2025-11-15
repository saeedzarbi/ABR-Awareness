"""
Ablation study: analyze the contribution of each component.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from configs.paths import get_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATHS = get_paths()


class AblationStudy:
    """Ablation study evaluator."""
    
    def __init__(self):
        self.results = []
        
    def evaluate_variant(
        self,
        variant_name: str,
        model_path: Path,
        video_name: str = 'sample1',
        num_episodes: int = 50
    ) -> dict:
        """Evaluate one ablation variant."""
        
        print(f"\n{'─'*70}")
        print(f"Evaluating: {variant_name}")
        print(f"{'─'*70}\n")
        
        try:
            model = PPO.load(str(model_path))
            print(f"  ✓ Model loaded from {model_path.name}")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            return None
        
        # Create environment
        env = ABREnv(
            video_name=video_name,
            trace_dir=str(PATHS['processed_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=48,
            random_seed=42
        )
        
        # Run evaluation
        rewards = []
        rebuffers = []
        qualities = []
        switches = []
        bitrates = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            last_action = 0
            ep_switches = 0
            ep_bitrates = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                if action != last_action:
                    ep_switches += 1
                last_action = action
                
                ep_bitrates.append(env.BITRATE_LEVELS[action])
                
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            
            rewards.append(ep_reward)
            rebuffers.append(info['total_rebuffer'])
            qualities.append(info['avg_quality'])
            switches.append(ep_switches)
            bitrates.append(np.mean(ep_bitrates))
        
        result = {
            'variant': variant_name,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'rebuffer_mean': np.mean(rebuffers),
            'rebuffer_std': np.std(rebuffers),
            'quality_mean': np.mean(qualities),
            'quality_std': np.std(qualities),
            'switches_mean': np.mean(switches),
            'switches_std': np.std(switches),
            'bitrate_mean': np.mean(bitrates),
            'bitrate_std': np.std(bitrates)
        }
        
        print(f"  Results:")
        print(f"    Reward:    {result['reward_mean']:7.2f} ± {result['reward_std']:5.2f}")
        print(f"    Rebuffer:  {result['rebuffer_mean']:7.2f}s ± {result['rebuffer_std']:5.2f}s")
        print(f"    Quality:   {result['quality_mean']:7.3f} ± {result['quality_std']:5.3f}")
        print(f"    Switches:  {result['switches_mean']:7.2f} ± {result['switches_std']:5.2f}")
        print(f"    Bitrate:   {result['bitrate_mean']:7.0f} Kbps")
        
        return result
    
    def run_all_ablations(self):
        """Run all ablation experiments."""
        
        print("\n" + "="*70)
        print("🔬 Ablation Study: Component Contribution Analysis")
        print("="*70)
        
        # Define variants
        variants = {
            'Full (V4)': PATHS['models'] / 'ppo_abr_v4' / 'best_model' / 'best_model',
            'No Buffer-Aware (V3)': PATHS['models'] / 'ppo_abr_v3' / 'best_model' / 'best_model',
            'Conservative (V2)': PATHS['models'] / 'ppo_abr_v2' / 'best_model' / 'best_model',
            'Baseline (V1)': PATHS['models'] / 'ppo_abr' / 'best_model' / 'best_model',
        }
        
        # Evaluate each variant
        for variant_name, model_path in variants.items():
            if model_path.with_suffix('.zip').exists():
                result = self.evaluate_variant(variant_name, model_path)
                if result:
                    self.results.append(result)
            else:
                print(f"  ⚠ Model not found: {variant_name}")
        
        print("\n" + "="*70)
        print("✓ Ablation study complete!")
        print(f"  Evaluated {len(self.results)} variants")
        print("="*70 + "\n")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create ablation comparison table."""
        
        if not self.results:
            print("No results to compare!")
            return None
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("Ablation Study Results")
        print("="*80 + "\n")
        
        # Sort by reward
        df = df.sort_values('reward_mean', ascending=False)
        
        print(f"{'Variant':<25} | {'Reward':>12} | {'Rebuffer':>12} | {'Quality':>10} | {'Switches':>10}")
        print("─" * 85)
        
        for _, row in df.iterrows():
            print(f"{row['variant']:<25} | "
                  f"{row['reward_mean']:6.2f} ± {row['reward_std']:4.2f} | "
                  f"{row['rebuffer_mean']:5.2f} ± {row['rebuffer_std']:4.2f}s | "
                  f"{row['quality_mean']:5.3f} ± {row['quality_std']:4.3f} | "
                  f"{row['switches_mean']:5.1f} ± {row['switches_std']:3.1f}")
        
        print("\n" + "="*80)
        print("Component Impact Analysis:")
        print("="*80 + "\n")
        
        # Calculate improvements
        if 'Full (V4)' in df['variant'].values and 'No Buffer-Aware (V3)' in df['variant'].values:
            v4 = df[df['variant'] == 'Full (V4)'].iloc[0]
            v3 = df[df['variant'] == 'No Buffer-Aware (V3)'].iloc[0]
            
            buffer_aware_impact = ((v4['reward_mean'] - v3['reward_mean']) / abs(v3['reward_mean'])) * 100
            bitrate_impact = ((v4['bitrate_mean'] - v3['bitrate_mean']) / v3['bitrate_mean']) * 100
            
            print(f"Buffer-Aware Dynamic Reward Impact:")
            print(f"  Reward change:  {buffer_aware_impact:+.1f}%")
            print(f"  Bitrate change: {bitrate_impact:+.1f}%")
            print(f"  Quality change: {(v4['quality_mean'] - v3['quality_mean']):+.3f}")
            print()
        
        if 'Full (V4)' in df['variant'].values and 'Baseline (V1)' in df['variant'].values:
            v4 = df[df['variant'] == 'Full (V4)'].iloc[0]
            v1 = df[df['variant'] == 'Baseline (V1)'].iloc[0]
            
            print(f"Overall Content-Awareness Impact (V4 vs V1):")
            print(f"  Reward improvement: {v4['reward_mean'] - v1['reward_mean']:+.2f}")
            print(f"  Rebuffer reduction: {v1['rebuffer_mean'] - v4['rebuffer_mean']:.2f}s")
            print(f"  Quality change: {(v4['quality_mean'] - v1['quality_mean']):+.3f}")
            print()
        
        return df
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create ablation study plots."""
        
        if df is None or df.empty:
            return
        
        print("\n📊 Creating ablation study plots...\n")
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ablation Study: Component Contribution', fontsize=16, fontweight='bold')
        
        # Sort for consistent visualization
        df = df.sort_values('reward_mean', ascending=False)
        
        # 1. Reward comparison
        ax = axes[0, 0]
        bars = ax.barh(df['variant'], df['reward_mean'], color=sns.color_palette("viridis", len(df)))
        ax.set_xlabel('Average Reward')
        ax.set_title('Overall QoE (Higher is Better)')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['reward_mean'], i, f" {row['reward_mean']:.1f}", 
                   va='center', fontsize=9)
        
        # 2. Rebuffering comparison
        ax = axes[0, 1]
        bars = ax.barh(df['variant'], df['rebuffer_mean'], color=sns.color_palette("rocket_r", len(df)))
        ax.set_xlabel('Average Rebuffering (seconds)')
        ax.set_title('Rebuffering Time (Lower is Better)')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['rebuffer_mean'], i, f" {row['rebuffer_mean']:.2f}s", 
                   va='center', fontsize=9)
        
        # 3. Quality comparison
        ax = axes[1, 0]
        bars = ax.barh(df['variant'], df['quality_mean'], color=sns.color_palette("mako", len(df)))
        ax.set_xlabel('Average Quality (VMAF/100)')
        ax.set_title('Video Quality (Higher is Better)')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['quality_mean'], i, f" {row['quality_mean']:.3f}", 
                   va='center', fontsize=9)
        
        # 4. Average bitrate
        ax = axes[1, 1]
        bars = ax.barh(df['variant'], df['bitrate_mean'], color=sns.color_palette("crest", len(df)))
        ax.set_xlabel('Average Bitrate (Kbps)')
        ax.set_title('Selected Bitrate (Higher = More Aggressive)')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['bitrate_mean'], i, f" {row['bitrate_mean']:.0f}", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        save_path = PATHS['results'] / 'ablation_study.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot: {save_path}")
    
    def save_results(self, df: pd.DataFrame):
        """Save ablation results."""
        
        if df is None or df.empty:
            return
        
        csv_path = PATHS['results'] / 'ablation_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved: {csv_path}")


def main():
    """Run ablation study."""
    
    study = AblationStudy()
    study.run_all_ablations()
    df = study.create_comparison_table()
    
    if df is not None:
        study.create_visualizations(df)
        study.save_results(df)
    
    print("\n" + "="*80)
    print("✓ Ablation study complete!")
    print("="*80)
    print("\nKey files:")
    print("  - results/ablation_results.csv")
    print("  - results/ablation_study.png")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()