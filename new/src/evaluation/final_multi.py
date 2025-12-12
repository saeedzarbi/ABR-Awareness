import sys
from pathlib import Path
import os
import json
import urllib.request
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_multi_env import ABREnv  # Changed from abr_env to abr_multi_env
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

# ============================================================================
# Evaluation Logger
# ============================================================================

class EvaluationLogger:
    """
    Detailed logger for evaluation phase
    Tracks chunk-level and episode-level metrics
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_logs = []
        self.episode_logs = []
    
    def log_chunk(self, env, action, info, episode_num):
        """Log each chunk during evaluation"""
        chunk_data = {
            'episode': episode_num,
            'chunk': env.chunk_idx - 1,
            'video': env.current_video_name,
            'action': int(action),
            'bitrate': int(env.BITRATE_LEVELS[action]),
            'buffer': float(info['buffer_level']),
            'throughput': float(info['throughput']),
            'rebuffer': float(info.get('rebuffer', 0)),
            'vmaf': float(env.last_vmaf),
            'reward': float(info.get('reward', 0)),
            'risk_factor': float(info.get('risk_factor', 1.0)),
            'weighted_penalty': float(info.get('weighted_rebuf_penalty', 0))
        }
        self.chunk_logs.append(chunk_data)
    
    def log_episode(self, env, total_reward, switches, episode_num):
        """Log complete episode statistics"""
        rebuffer_rate = (env.total_rebuffer / (env.chunk_idx * 4.0)) * 100 if env.chunk_idx > 0 else 0
        
        episode_data = {
            'episode': episode_num,
            'video': env.current_video_name,
            'trace_idx': env.current_trace_idx,
            'total_reward': float(total_reward),
            'avg_vmaf': float(env.total_quality / env.chunk_idx),
            'total_rebuffer': float(env.total_rebuffer),
            'rebuffer_rate': float(rebuffer_rate),
            'total_smooth': float(env.total_smooth),
            'switches': int(switches),
            'chunks': int(env.chunk_idx)
        }
        self.episode_logs.append(episode_data)
    
    def save_logs(self, method_name='method'):
        """Save all logs to CSV files"""
        
        # Chunk-level logs
        if self.chunk_logs:
            df_chunks = pd.DataFrame(self.chunk_logs)
            chunk_path = self.log_dir / f'{method_name}_chunks.csv'
            df_chunks.to_csv(chunk_path, index=False)
            print(f"   ✅ Saved chunk logs: {chunk_path} ({len(self.chunk_logs)} chunks)")
        
        # Episode-level logs
        if self.episode_logs:
            df_episodes = pd.DataFrame(self.episode_logs)
            episode_path = self.log_dir / f'{method_name}_episodes.csv'
            df_episodes.to_csv(episode_path, index=False)
            print(f"   ✅ Saved episode logs: {episode_path} ({len(self.episode_logs)} episodes)")
    
    def analyze_logs(self, method_name='method'):
        """Quick analysis of logs"""
        if not self.episode_logs:
            return
        
        df = pd.DataFrame(self.episode_logs)
        
        print(f"\n📊 {method_name} - Performance Analysis:")
        print("="*60)
        
        # Overall stats
        print(f"   Avg VMAF:      {df['avg_vmaf'].mean():.2f} ± {df['avg_vmaf'].std():.2f}")
        print(f"   Avg Rebuffer:  {df['rebuffer_rate'].mean():.2f}% ± {df['rebuffer_rate'].std():.2f}%")
        print(f"   Avg QoE:       {df['total_reward'].mean():.1f} ± {df['total_reward'].std():.1f}")
        print(f"   Avg Switches:  {df['switches'].mean():.1f}")
        
        # Per-video breakdown
        if 'video' in df.columns and df['video'].nunique() > 1:
            print(f"\n   Per-Video Breakdown:")
            video_stats = df.groupby('video').agg({
                'avg_vmaf': 'mean',
                'rebuffer_rate': 'mean',
                'total_reward': 'mean'
            }).round(2)
            for video, row in video_stats.iterrows():
                print(f"      {video:20s}: VMAF={row['avg_vmaf']:5.1f}, "
                      f"Rebuf={row['rebuffer_rate']:5.2f}%, QoE={row['total_reward']:7.1f}")
        
        # Rebuffer analysis
        high_rebuf = df[df['rebuffer_rate'] > 5.0]
        if len(high_rebuf) > 0:
            print(f"\n   ⚠️ High rebuffer episodes (>5%): {len(high_rebuf)}/{len(df)}")
        
        print("="*60)

# ============================================================================
# Slack Integration
# ============================================================================

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
                "footer": "Final Evaluation V10",
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

# ============================================================================
# Main Evaluator
# ✓ project_root        : /home/saeedzarbi95/test/ABR-Awareness/new
# ✓ data_dir            : /home/saeedzarbi95/test/ABR-Awareness/new/data
# ✓ raw_videos          : /home/saeedzarbi95/test/ABR-Awareness/new/data/raw_videos
# ✓ encoded_videos      : /home/saeedzarbi95/test/ABR-Awareness/new/data/encoded_videos
# ✓ vmaf_scores         : /home/saeedzarbi95/test/ABR-Awareness/new/data/vmaf_scores
# ✓ content_features    : /home/saeedzarbi95/test/ABR-Awareness/new/data/content_features
# ✓ processed_traces    : /home/saeedzarbi95/test/ABR-Awareness/new/data/standardized/train_traces
# ✓ train_traces        : /home/saeedzarbi95/test/ABR-Awareness/new/data/standardized/train_traces
# ✓ test_traces         : /home/saeedzarbi95/test/ABR-Awareness/new/data/standardized/test_traces
# ✓ network_traces      : /home/saeedzarbi95/test/ABR-Awareness/new/data/network_traces
# ✓ results             : /home/saeedzarbi95/test/ABR-Awareness/new/results
# ✓ models              : /home/saeedzarbi95/test/ABR-Awareness/new/results/models
# ✓ logs                : /home/saeedzarbi95/test/ABR-Awareness/new/results/logs
# ✓ plots               : /home/saeedzarbi95/test/ABR-Awareness/new/results/plots
# ============================================================================

class TCSVT_Evaluator:
    def __init__(self):
        self.test_trace_dir = PATHS['test_traces']
        self.results_detailed = [] 

        self.test_videos = [
            'bigbuckbunny',    
            'crowd_run',    
            # # 'parkjoy',       
            'tearsofsteel_short',
            'sintel'
        ]

    def load_methods(self):
        methods = {}
        
        # 1. Proposed (V10)
        try:
            path = PATHS['models'] / 'ppo_abr_multi_dynamic_21' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                path = PATHS['models'] / 'ppo_abr_multi_dynamic_21' / 'final_model'
            methods['Proposed'] = PPO.load(str(path))
            print(f"✅ Loaded Proposed from: {path}")
        except Exception as e:
            print(f"⚠️ Proposed missing: {e}")
        
        # 2. Pensieve
        try:
            path = PATHS['models'] / 'pensieve_multi_vmaf_new_16' / 'best_model' / 'best_model'
            if not path.with_suffix('.zip').exists():
                 path = PATHS['models'] / 'pensieve_multi_vmaf_new_16' / 'final_model'
            methods['Pensieve'] = PPO.load(str(path))
            print(f"✅ Loaded Pensieve from: {path}")
        except Exception as e:
            print(f"⚠️ Pensieve missing: {e}")

        # Baselines
        methods['RobustMPC'] = 'mpc_placeholder' 
        methods['Genie'] = 'genie_placeholder'
        methods['BBA'] = BBA(ABREnv.BITRATE_LEVELS)
            
        return methods

    def evaluate(self, methods, episodes_per_video=20, enable_logging=True):
        print(f"\n🔬 Running Evaluation V21 (N={episodes_per_video} per video)...")
        print(f"   Enhanced logging: {'Enabled' if enable_logging else 'Disabled'}")
        
        send_slack_message("info", "Evaluation Started", 
                          f"Starting V21 evaluation on {len(self.test_videos)} videos")
        
        if not list(self.test_trace_dir.glob("*.json")):
            print("❌ No traces found.")
            send_slack_message("error", "Evaluation Failed", "No traces found")
            return

        for video_idx, video_name in enumerate(self.test_videos, 1):
            print(f"\n📹 Testing on Video: {video_name} ({video_idx}/{len(self.test_videos)})")
            
            if not (PATHS['content_features'] / f"{video_name}_siti.json").exists():
                print(f"   ⚠️ Skipping {video_name}: Data not found.")
                continue

            env = ABREnv(
                video_names=video_name,  # Changed from video_name to video_names (accepts single string)
                trace_dir=str(self.test_trace_dir),
                vmaf_dir=str(PATHS['vmaf_scores']),
                siti_dir=str(PATHS['content_features']),
                max_chunks=48,
                random_seed=12345
            )

            for name, model in methods.items():
                print(f"   > Running {name}...", end='\r')
                
                # Create logger for this method
                if enable_logging:
                    logger = EvaluationLogger(PATHS['logs'] / 'evaluation_v21')
                
                active_model = model
                if name == 'RobustMPC': 
                    active_model = RobustMPC(env)
                elif name == 'Genie': 
                    active_model = Genie(env)
                
                for ep in range(episodes_per_video):
                    obs, info = env.reset()
                    done = False
                    last_br = 0
                    switches = 0
                    last_tp = 2000.0
                    trace_tp = env.current_trace['throughput_kbps']
                    total_reward = 0
                    
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
                                obs[10:] = 0.0
                            action, _ = active_model.predict(obs, deterministic=True)
                        
                        if action != last_br: 
                            switches += 1
                        last_br = action
                        
                        obs, reward, done, _, info = env.step(action)
                        total_reward += reward
                        last_tp = info.get('throughput', last_tp)
                        
                        # Log chunk
                        if enable_logging and name == 'Proposed':
                            logger.log_chunk(env, action, info, ep)
                    
                    # Calculate QoE
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
                    
                    # Log episode
                    if enable_logging and name == 'Proposed':
                        logger.log_episode(env, total_reward, switches, ep)
                
                # Save logs for this method
                if enable_logging and name == 'Proposed':
                    logger.save_logs(f'{name}_{video_name}')
                    logger.analyze_logs(f'{name}_{video_name}')
                
            print(f"   ✅ {video_name} Done.        ")

    def save_statistics(self):
        if not self.results_detailed: 
            send_slack_message("warning", "No Results", "No evaluation results to save")
            return
        
        df = pd.DataFrame(self.results_detailed)
        
        # ✅ Changed filename to _10
        path = PATHS['results'] / 'detailed_stats_multi_video_21.csv'
        df.to_csv(path, index=False)
        print(f"\n✅ Saved results to: {path}")
        
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
        
        # Send to Slack
        details_msg = "📊 *Final Evaluation Results V21*\n\n"
        
        details_msg += "*Overall Performance:*\n"
        for method in summary.index:
            qoe_mean = summary.loc[method, ('QoE', 'mean')]
            vmaf_mean = summary.loc[method, ('VMAF', 'mean')]
            rebuf_mean = summary.loc[method, ('Rebuffer', 'mean')]
            
            details_msg += f"• *{method}:* QoE={qoe_mean:.1f}, VMAF={vmaf_mean:.2f}, Rebuf={rebuf_mean:.2f}%\n"
        
        send_slack_message("success", "Evaluation Completed", details_msg)

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Load existing CSV')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per video')
    parser.add_argument('--no-logging', action='store_true', help='Disable detailed logging')
    args = parser.parse_args()
    
    evaluator = TCSVT_Evaluator()
    
    if args.load:
        evaluator.load_and_calculate_statistics(args.load)
    else:
        methods = evaluator.load_methods()
        if methods:
            evaluator.evaluate(methods, 
                             episodes_per_video=args.episodes,
                             enable_logging=not args.no_logging)
            evaluator.save_statistics()