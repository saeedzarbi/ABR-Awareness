import sys
from pathlib import Path
import os
import json
import urllib.request
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from src.baselines.mpc_vmaf import RobustMPC 
from src.baselines.genie import Genie
from src.baselines.bba import BBA
from configs.paths import get_paths
import numpy as np
import pandas as pd
import argparse

PATHS = get_paths()

# Slack webhook URL (from environment variable)
SLACK_WEBHOOK = os.getenv('SLACK_WEBHOOK_URL', '')

def send_slack_message(status, step, message):
    """Send message to Slack"""
    if not SLACK_WEBHOOK:
        return
    
    color = "good"
    emoji = "✅"
    if status == "error":
        color = "danger"
        emoji = "❌"
    elif status == "info":
        color = "#36a64f"
        emoji = "ℹ️"
    elif status == "warning":
        color = "warning"
        emoji = "⚠️"
    
    payload = {
        "attachments": [
            {
                "color": color,
                "title": f"{emoji} {step}",
                "text": message,
                "footer": "Final Evaluation",
                "ts": int(pd.Timestamp.now().timestamp())
            }
        ]
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            SLACK_WEBHOOK,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"⚠️ Failed to send Slack notification: {e}")

class TCSVT_Evaluator:
    def __init__(self):
        self.test_trace_dir = PATHS['test_traces']
        self.results_detailed = [] 

        self.test_videos = [
            'bigbuckbunny',    
            'crowd_run',    
            'parkjoy',       
            'tearsofsteel_short' 
        ]

    def load_methods(self):
        methods = {}
        
        # 1. Proposed
        try:
            path = PATHS['models'] / 'ppo_abr_multi_dynamic_9' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_multi_dynamic_8' / 'final_model'
            methods['Proposed'] = PPO.load(str(path))
            print(f"✓ Loaded Proposed from: {path}")
        except: print("⚠ Proposed missing.")
        
        # 2. Pensieve
        try:
            path = PATHS['models'] / 'pensieve_multi_vmaf_new_9' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_multi_vmaf_new_9' / 'final_model'
            methods['Pensieve'] = PPO.load(str(path))
            print(f"✓ Loaded Pensieve from: {path}")
        except: print("⚠ Pensieve missing.")

        # Baselines
        methods['RobustMPC'] = 'mpc_placeholder' 
        methods['Genie'] = 'genie_placeholder'
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
            
        return methods

    def evaluate(self, methods, episodes_per_video=20):
        print(f"\n🔬 Running Statistical Evaluation (N={episodes_per_video} per video)...")
        
        send_slack_message("info", "Evaluation Started", 
                          f"Starting evaluation on {len(self.test_videos)} videos with {episodes_per_video} episodes each")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No traces found.")
            send_slack_message("error", "Evaluation Failed", "No traces found in test directory")
            return

        for video_idx, video_name in enumerate(self.test_videos, 1):
            print(f"\n📹 Testing on Video: {video_name} ({video_idx}/{len(self.test_videos)})")
            
            send_slack_message("info", f"Testing Video {video_idx}/{len(self.test_videos)}", 
                            f"Evaluating on video: {video_name}")
            
            if not (PATHS['content_features'] / f"{video_name}_siti.json").exists():
                print(f"   ⚠ Skipping {video_name}: Data not found.")
                send_slack_message("warning", f"Video Skipped", f"{video_name}: Data not found")
                continue

            env = ABREnv(
                video_name=video_name,
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345
            )

            for name, model in methods.items():
                print(f"   > Running {name}...", end='\r')
                
                active_model = model
                if name == 'RobustMPC': active_model = RobustMPC(env)
                elif name == 'Genie': active_model = Genie(env)
                
                for ep in range(episodes_per_video):
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
                            if name == 'Pensieve': obs[10:] = 0.0
                            action, _ = active_model.predict(obs, deterministic=True)
                        
                        if action != last_br: switches += 1
                        last_br = action
                        
                        obs, _, done, _, info = env.step(action)
                        last_tp = info.get('throughput', last_tp)
                    
                    qoe = info['total_quality'] - (env.REBUF_PENALTY_BASE * info['total_rebuffer']) - (env.SMOOTH_PENALTY_WEIGHT * info['total_smoothness'])
                    
                    video_duration = env.chunk_idx * 4.0
                    rebuf_ratio = (info['total_rebuffer'] / video_duration) * 100 if video_duration > 0 else 0

                    self.results_detailed.append({
                        'Method': name,
                        'Video': video_name, 
                        'Episode': ep,
                        'VMAF': info['avg_quality'],
                        'Rebuffer': rebuf_ratio,
                        'QoE': qoe,
                        'Switch': switches
                    })
            print(f"   ✓ {video_name} Done.        ")
            
            # Send progress update
            video_results = [r for r in self.results_detailed if r['Video'] == video_name]
            if video_results:
                video_df = pd.DataFrame(video_results)
                video_summary = video_df.groupby('Method').agg({
                    'VMAF': 'mean',
                    'Rebuffer': 'mean',
                    'QoE': 'mean'
                }).round(2)
                
                summary_text = f"*{video_name} Results:*\n"
                for method, row in video_summary.iterrows():
                    summary_text += f"• {method}: VMAF={row['VMAF']:.2f}, Rebuf={row['Rebuffer']:.2f}%, QoE={row['QoE']:.1f}\n"
                
                send_slack_message("success", f"{video_name} Completed", summary_text)

    def save_statistics(self):
        if not self.results_detailed: 
            send_slack_message("warning", "No Results", "No evaluation results to save")
            return
        
        df = pd.DataFrame(self.results_detailed)
        path = PATHS['results'] / 'detailed_stats_multi_video_9.csv'
        df.to_csv(path, index=False)
        print(f"\n✓ Saved results to: {path}")
        
        send_slack_message("info", "Results Saved", f"Results saved to: {path}")
        
        self.print_summary(df)
        return df
    
    def load_and_calculate_statistics(self, csv_file):
        path = PATHS['results'] / csv_file
        if not path.exists():
            print(f"❌ File not found: {path}")
            return
        
        print(f"\n📊 Loading from: {csv_file}")
        df = pd.read_csv(path)
        self.print_summary(df)
        return df

    def print_summary(self, df):
        summary = df.groupby('Method').agg({
            'QoE': ['mean', 'std'],
            'VMAF': ['mean', 'std'],
            'Rebuffer': ['mean', 'std']
        }).round(2)
        print("\n🏆 Overall Statistical Summary:")
        print(summary)
        
        # Send detailed results to Slack
        details_msg = "📊 *Final Evaluation Results*\n\n"
        
        # Overall statistics per method
        details_msg += "*Overall Performance (All Videos):*\n"
        for method in summary.index:
            qoe_mean = summary.loc[method, ('QoE', 'mean')]
            qoe_std = summary.loc[method, ('QoE', 'std')]
            vmaf_mean = summary.loc[method, ('VMAF', 'mean')]
            vmaf_std = summary.loc[method, ('VMAF', 'std')]
            rebuf_mean = summary.loc[method, ('Rebuffer', 'mean')]
            rebuf_std = summary.loc[method, ('Rebuffer', 'std')]
            
            details_msg += f"• *{method}:*\n"
            details_msg += f"  - QoE: {qoe_mean:.1f} ± {qoe_std:.1f}\n"
            details_msg += f"  - VMAF: {vmaf_mean:.2f} ± {vmaf_std:.2f}\n"
            details_msg += f"  - Rebuffer: {rebuf_mean:.2f}% ± {rebuf_std:.2f}%\n\n"
        
        # Per-video breakdown
        if 'Video' in df.columns:
            details_msg += "*Per-Video Breakdown:*\n"
            for video in df['Video'].unique():
                video_df = df[df['Video'] == video]
                video_summary = video_df.groupby('Method').agg({
                    'VMAF': 'mean',
                    'Rebuffer': 'mean',
                    'QoE': 'mean'
                }).round(2)
                
                details_msg += f"\n*{video}:*\n"
                for method, row in video_summary.iterrows():
                    details_msg += f"  • {method}: VMAF={row['VMAF']:.2f}, Rebuf={row['Rebuffer']:.2f}%, QoE={row['QoE']:.1f}\n"
        
        # Best performer
        best_qoe_method = summary[('QoE', 'mean')].idxmax()
        best_vmaf_method = summary[('VMAF', 'mean')].idxmax()
        lowest_rebuf_method = summary[('Rebuffer', 'mean')].idxmin()
        
        details_msg += f"\n*Best Performers:*\n"
        details_msg += f"• Best QoE: {best_qoe_method} ({summary.loc[best_qoe_method, ('QoE', 'mean')]:.1f})\n"
        details_msg += f"• Best VMAF: {best_vmaf_method} ({summary.loc[best_vmaf_method, ('VMAF', 'mean')]:.2f})\n"
        details_msg += f"• Lowest Rebuffer: {lowest_rebuf_method} ({summary.loc[lowest_rebuf_method, ('Rebuffer', 'mean')]:.2f}%)\n"
        
        # Test configuration
        details_msg += f"\n*Test Configuration:*\n"
        details_msg += f"• Videos: {', '.join(df['Video'].unique()) if 'Video' in df.columns else 'N/A'}\n"
        details_msg += f"• Total Episodes: {len(df)}\n"
        details_msg += f"• Methods Tested: {len(summary)}\n"
        
        send_slack_message("success", "Evaluation Completed", details_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Load existing CSV')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per video')
    args = parser.parse_args()
    
    evaluator = TCSVT_Evaluator()
    
    if args.load:
        evaluator.load_and_calculate_statistics(args.load)
    else:
        methods = evaluator.load_methods()
        if methods:
            evaluator.evaluate(methods, episodes_per_video=args.episodes)
            evaluator.save_statistics()