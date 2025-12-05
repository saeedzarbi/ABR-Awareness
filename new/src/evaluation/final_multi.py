import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from src.baselines.mpc_vmaf import RobustMPC 
from src.baselines.genie import Genie
from src.baselines.bba import BBA
from configs.paths import get_paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATHS = get_paths()

class MultiVideoEvaluator:
    """Evaluator for multiple videos - extends TCSVT_Evaluator for multi-video analysis"""
    
    def __init__(self, video_list=None):
        self.test_trace_dir = PATHS['test_traces']
        self.results_detailed = []  # Store every episode with video info
        self.results_per_video = {}  # Store aggregated results per video
        
        # Default video list (can be overridden)
        if video_list is None:
            # Default: only park_joy for evaluation
            self.video_list = ['park_joy']
        else:
            self.video_list = video_list if isinstance(video_list, list) else [video_list]
        
    def load_methods(self):
        """Load all evaluation methods (same as final_comparison.py)"""
        methods = {}
        
        # 1. Proposed
        try:
            path = PATHS['models'] / 'ppo_abr_multi_dynamic' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_multi_dynamic' / 'final_model'
            methods['Proposed'] = PPO.load(str(path))
            print("✓ Loaded Proposed model")
        except Exception as e: 
            print(f"⚠ Proposed missing: {e}")
        
        # 2. Baseline: Pensieve*
        try:
            path = PATHS['models'] / 'pensieve_multi_vmaf' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_multi_vmaf' / 'final_model'
            methods['Pensieve'] = PPO.load(str(path))
            print("✓ Loaded Pensieve model")
        except Exception as e: 
            print(f"⚠ Pensieve missing: {e}")

        # 3. RobustMPC
        methods['RobustMPC'] = 'mpc_placeholder' 
        
        # 4. Genie
        methods['Genie'] = 'genie_placeholder'
        
        # 5. BBA
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
        print("✓ Loaded BBA baseline")
            
        return methods

    def evaluate_video(self, methods, video_name, episodes=50):
        """Evaluate all methods on a single video"""
        print(f"\n📹 Evaluating on video: '{video_name}' ({episodes} episodes)")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print(f"❌ No traces found for {video_name}.")
            return

        video_results = []
        
        for name, model in methods.items():
            print(f"   Running {name}...", end='', flush=True)
            
            try:
                env = ABREnv(
                    video_name=video_name,
                    trace_dir=str(self.test_trace_dir),
                    vmaf_dir=str(PATHS['vmaf_scores']),
                    siti_dir=str(PATHS['content_features']),
                    max_chunks=48,
                    random_seed=12345
                )
            except Exception as e:
                print(f" ✗ Failed to create env: {e}")
                continue
            
            # Init specific models
            active_model = model
            if name == 'RobustMPC': 
                active_model = RobustMPC(env)
            elif name == 'Genie': 
                active_model = Genie(env)
            
            for ep in range(episodes):
                try:
                    obs, info = env.reset()
                    done = False
                    last_br = 0
                    switches = 0
                    last_tp = 2000.0
                    trace_tp = env.current_trace['throughput_kbps']
                    
                    while not done:
                        if name == 'RobustMPC':
                            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
                            action = active_model.select_bitrate(info['buffer_level'], last_tp, cur_vmaf)
                        elif name == 'Genie':
                            action = active_model.select_bitrate(env.chunk_idx, env.buffer_level, trace_tp)
                        elif name == 'BBA':
                            action = active_model.select_bitrate(info['buffer_level'])
                        else:
                            if name == 'Pensieve': 
                                obs[10:] = 0.0  # Zero out content features
                            action, _ = active_model.predict(obs, deterministic=True)
                        
                        if action != last_br: 
                            switches += 1
                        last_br = action
                        
                        obs, _, done, _, info = env.step(action)
                        last_tp = info.get('throughput', last_tp)
                    
                    # Calculate QoE
                    qoe = info['total_quality'] - (env.REBUF_PENALTY_BASE * info['total_rebuffer']) - (env.SMOOTH_PENALTY_WEIGHT * info['total_smoothness'])
                    
                    # Save PER EPISODE metrics with video info
                    result = {
                        'Method': name,
                        'Video': video_name,
                        'Episode': ep,
                        'VMAF': info['avg_quality'],
                        'Rebuffer': (info['total_rebuffer'] / (48*4)) * 100,  # Percentage
                        'QoE': qoe,
                        'Switch': switches,
                        'AvgBitrate': info.get('avg_bitrate', 0.0)
                    }
                    
                    self.results_detailed.append(result)
                    video_results.append(result)
                    
                except Exception as e:
                    print(f" ✗ Episode {ep} failed: {e}")
                    continue
                    
            print(" Done.")
        
        # Aggregate per video
        if video_results:
            df_video = pd.DataFrame(video_results)
            summary = df_video.groupby('Method').agg({
                'QoE': ['mean', 'std'],
                'VMAF': ['mean', 'std'],
                'Rebuffer': ['mean', 'std'],
                'Switch': ['mean', 'std'],
                'AvgBitrate': ['mean', 'std']
            }).round(2)
            # Reset index to get Method as column
            summary = summary.reset_index()
            # Flatten MultiIndex columns
            summary.columns = [
                f"{col[0]}_{col[1]}" if isinstance(col, tuple) and len(col) == 2 else str(col)
                for col in summary.columns
            ]
            # Store summary
            self.results_per_video[video_name] = summary

    def evaluate_all_videos(self, methods, episodes_per_video=50):
        """Evaluate all methods on all videos"""
        print(f"\n{'='*80}")
        print(f"🎬 MULTI-VIDEO EVALUATION")
        print(f"{'='*80}")
        print(f"Videos: {len(self.video_list)}")
        print(f"Episodes per video: {episodes_per_video}")
        print(f"Total episodes: {len(self.video_list) * episodes_per_video * len(methods)}")
        print(f"Videos: {', '.join(self.video_list)}")
        print(f"{'='*80}")
        
        for video_name in self.video_list:
            self.evaluate_video(methods, video_name, episodes_per_video)
        
        print(f"\n{'='*80}")
        print("✅ All videos evaluated!")
        print(f"{'='*80}")

    def save_statistics(self):
        """Save detailed and aggregated statistics"""
        if not self.results_detailed: 
            print("❌ No results to save.")
            return None
        
        df = pd.DataFrame(self.results_detailed)
        
        # Save detailed results
        detailed_path = PATHS['results'] / 'detailed_stats_multi_video.csv'
        df.to_csv(detailed_path, index=False)
        print(f"\n✓ Detailed statistics saved to: detailed_stats_multi_video.csv")
        
        # Save per-video summary
        if self.results_per_video:
            per_video_path = PATHS['results'] / 'per_video_summary.csv'
            per_video_list = []
            for video, summary in self.results_per_video.items():
                # summary already has Method as column and flattened columns
                # Melt to convert wide format to long format
                # Get all columns except 'Method' as value_vars
                value_vars = [col for col in summary.columns if col != 'Method']
                summary_melted = summary.melt(
                    id_vars=['Method'],
                    value_vars=value_vars,
                    var_name='Metric',
                    value_name='Value'
                )
                summary_melted['Video'] = video
                per_video_list.append(summary_melted)
            
            per_video_df = pd.concat(per_video_list, ignore_index=True)
            per_video_df.to_csv(per_video_path, index=False)
            print(f"✓ Per-video summary saved to: {per_video_path}")
        
        # Overall summary (across all videos)
        print("\n🏆 Overall Statistical Summary (All Videos):")
        print("="*80)
        overall_summary = df.groupby('Method').agg({
            'QoE': ['mean', 'std'],
            'VMAF': ['mean', 'std'],
            'Rebuffer': ['mean', 'std'],
            'Switch': ['mean', 'std']
        }).round(2)
        print(overall_summary)
        
        # Per-video breakdown
        print("\n📊 Per-Video Summary:")
        print("="*80)
        for video_name in self.video_list:
            video_df = df[df['Video'] == video_name]
            if not video_df.empty:
                print(f"\n📹 {video_name}:")
                video_summary = video_df.groupby('Method').agg({
                    'QoE': ['mean', 'std'],
                    'VMAF': ['mean', 'std'],
                    'Rebuffer': ['mean', 'std']
                }).round(2)
                print(video_summary)
        
        # Save overall summary
        overall_path = PATHS['results'] / 'overall_summary_multi_video.csv'
        overall_summary.to_csv(overall_path, index=True)
        print(f"\n✓ Overall summary saved to: {overall_path}")
        
        return df

    def plot_results(self, save_path=None):
        """Create visualization plots for multi-video results"""
        if not self.results_detailed:
            print("❌ No results to plot.")
            return
        
        df = pd.DataFrame(self.results_detailed)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        _, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. QoE comparison per video
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='Video', y='QoE', hue='Method', ax=ax1)
        ax1.set_title('QoE Distribution by Video and Method', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Video', fontsize=12)
        ax1.set_ylabel('QoE Score', fontsize=12)
        ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. VMAF comparison per video
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='Video', y='VMAF', hue='Method', ax=ax2)
        ax2.set_title('VMAF Distribution by Video and Method', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Video', fontsize=12)
        ax2.set_ylabel('VMAF Score', fontsize=12)
        ax2.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Rebuffer comparison per video
        ax3 = axes[1, 0]
        sns.boxplot(data=df, x='Video', y='Rebuffer', hue='Method', ax=ax3)
        ax3.set_title('Rebuffer Percentage by Video and Method', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Video', fontsize=12)
        ax3.set_ylabel('Rebuffer %', fontsize=12)
        ax3.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Overall comparison (bar plot)
        ax4 = axes[1, 1]
        overall_means = df.groupby('Method').agg({
            'QoE': 'mean',
            'VMAF': 'mean',
            'Rebuffer': 'mean'
        })
        overall_means.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Overall Performance Comparison (All Videos)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Method', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.legend(['QoE (scaled)', 'VMAF', 'Rebuffer %'])
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = PATHS['plots'] / 'multi_video_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
        plt.close()


if __name__ == '__main__':
    # Configuration
    VIDEO_LIST = [
        'park_joy'  # Only evaluate park_joy video
    ]
    
    EPISODES_PER_VIDEO = 50
    
    # Initialize evaluator
    evaluator = MultiVideoEvaluator(video_list=VIDEO_LIST)
    
    # Load methods
    methods = evaluator.load_methods()
    
    if methods:
        # Evaluate all videos
        evaluator.evaluate_all_videos(methods, episodes_per_video=EPISODES_PER_VIDEO)
        
        # Save statistics
        evaluator.save_statistics()
        
        # Create plots
        evaluator.plot_results()
        
        print("\n✅ Multi-video evaluation complete!")
    else:
        print("❌ No methods loaded. Please check model paths.")

