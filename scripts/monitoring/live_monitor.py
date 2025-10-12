"""
Live Training Monitor
"""

import json
import time
import sys
from pathlib import Path
import numpy as np


def clear_screen():
    print("\033[2J\033[H", end="")


def plot_ascii(values, width=60, height=10):
    """Simple ASCII plot"""
    if not values or len(values) < 2:
        return "Not enough data"
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return "No variation yet"
    
    # Normalize
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Sample if too many points
    if len(normalized) > width:
        step = len(normalized) / width
        normalized = [normalized[int(i * step)] for i in range(width)]
    
    # Create plot
    lines = []
    for row in range(height):
        threshold = 1.0 - (row / height)
        line = ""
        for val in normalized:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        lines.append(line)
    
    # Add axis
    plot = f"Max: {max_val:+7.2f} │" + lines[0] + "\n"
    for line in lines[1:-1]:
        plot += "             │" + line + "\n"
    plot += f"Min: {min_val:+7.2f} │" + lines[-1]
    
    return plot


def monitor_training(log_file, refresh_interval=10):
    """Monitor training log file"""
    
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"Waiting for {log_file}...")
        while not log_path.exists():
            time.sleep(2)
        print(f"✓ Found {log_file}")
    
    print(f"📊 Monitoring: {log_file}")
    print(f"Refresh: {refresh_interval}s")
    print("Press Ctrl+C to stop\n")
    time.sleep(2)
    
    last_size = 0
    
    try:
        while True:
            current_size = log_path.stat().st_size
            
            if current_size > last_size or current_size == 0:
                last_size = current_size
                
                # Read log entries
                updates = []
                try:
                    with open(log_path, 'r') as f:
                        for line in f:
                            try:
                                updates.append(json.loads(line))
                            except:
                                pass
                except:
                    pass
                
                if updates:
                    clear_screen()
                    
                    latest = updates[-1]
                    
                    # Header
                    print("=" * 80)
                    print("🚀 LIVE TRAINING MONITOR")
                    print("=" * 80)
                    print()
                    
                    # Current stats
                    print("📈 CURRENT STATUS:")
                    print(f"   Update:      {latest['update']:>6}")
                    print(f"   Timesteps:   {latest['timesteps']:>8,}")
                    print(f"   Episodes:    {latest['total_episodes']:>6}")
                    print(f"   Time:        {latest['time_elapsed']/3600:>6.1f}h")
                    print()
                    
                    # Performance
                    print("🎯 PERFORMANCE:")
                    print(f"   Reward (last 20):   {latest['avg_reward_20']:>+8.2f}")
                    print(f"   Reward (last 100):  {latest['avg_reward_100']:>+8.2f}")
                    print(f"   Rebuffer (last 20): {latest['avg_rebuffer_20']:>7.2f}s")
                    print(f"   Bitrate (last 20):  {latest['avg_bitrate_20']:>7.0f}kbps")
                    print()
                    
                    # Training metrics
                    print("🔧 TRAINING METRICS:")
                    print(f"   Policy Loss:  {latest['policy_loss']:>8.4f}")
                    print(f"   Value Loss:   {latest['value_loss']:>8.1f}")
                    print(f"   Entropy:      {latest['entropy']:>8.4f}")
                    print()
                    
                    # Learning curve
                    recent = updates[-100:]
                    rewards = [u['avg_reward_20'] for u in recent]
                    
                    print("📊 REWARD CURVE (last 100 updates):")
                    print(plot_ascii(rewards))
                    print()
                    
                    # Footer
                    print("=" * 80)
                    print(f"Last update: {time.strftime('%H:%M:%S')}")
                    print(f"Refreshing in {refresh_interval}s...")
                else:
                    print("Waiting for data...")
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True)
    parser.add_argument('--refresh', type=int, default=10)
    args = parser.parse_args()
    
    monitor_training(args.log_file, args.refresh)
