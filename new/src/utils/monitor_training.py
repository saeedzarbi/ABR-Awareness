"""
Monitor training progress without TensorBoard.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import json
import numpy as np
from configs.paths import get_paths

PATHS = get_paths()


def read_monitor_logs(log_dir: Path):
    """Read Stable-Baselines3 monitor logs."""
    monitor_files = list(log_dir.glob("**/0.monitor.csv"))
    
    if not monitor_files:
        return None
    
    # Read latest monitor file
    monitor_file = monitor_files[0]
    
    try:
        # Skip first line (metadata)
        with open(monitor_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) <= 2:
            return None
        
        # Parse episodes
        episodes = []
        for line in lines[2:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    reward = float(parts[0])
                    length = float(parts[1])
                    time_elapsed = float(parts[2])
                    episodes.append({
                        'reward': reward,
                        'length': length,
                        'time': time_elapsed
                    })
                except:
                    continue
        
        return episodes
    except:
        return None


def read_checkpoints(checkpoint_dir: Path):
    """Get list of saved checkpoints."""
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = sorted(checkpoint_dir.glob("ppo_v2_*_steps.zip"))
    return [
        {
            'path': cp,
            'steps': int(cp.stem.split('_')[-2]),
            'time': cp.stat().st_mtime
        }
        for cp in checkpoints
    ]


def print_progress(log_dir: Path, checkpoint_dir: Path):
    """Print training progress."""
    
    print("\n" + "="*70)
    print("📊 Training Progress Monitor")
    print("="*70 + "\n")
    
    last_episode_count = 0
    
    while True:
        try:
            # Read monitor logs
            episodes = read_monitor_logs(log_dir)
            
            if episodes and len(episodes) > last_episode_count:
                # Get recent episodes
                recent = episodes[-20:]
                
                rewards = [ep['reward'] for ep in recent]
                lengths = [ep['length'] for ep in recent]
                
                print(f"\n{'='*70}")
                print(f"Episodes completed: {len(episodes)}")
                print(f"{'='*70}")
                print(f"Recent 20 episodes:")
                print(f"  Reward:  {np.mean(rewards):7.2f} ± {np.std(rewards):6.2f}")
                print(f"           Min: {np.min(rewards):7.2f}, Max: {np.max(rewards):7.2f}")
                print(f"  Length:  {np.mean(lengths):7.1f} ± {np.std(lengths):6.1f}")
                
                # Show last 5 episodes
                print(f"\nLast 5 episodes:")
                for i, ep in enumerate(episodes[-5:], 1):
                    print(f"  {len(episodes)-5+i:4d}. Reward: {ep['reward']:7.2f}, Length: {ep['length']:.0f}")
                
                last_episode_count = len(episodes)
            
            # Read checkpoints
            checkpoints = read_checkpoints(checkpoint_dir)
            
            if checkpoints:
                latest = checkpoints[-1]
                print(f"\nCheckpoints saved: {len(checkpoints)}")
                print(f"  Latest: {latest['steps']:,} steps")
                print(f"  Progress: {latest['steps']/500000*100:.1f}%")
            
            print(f"\n{'='*70}")
            print(f"Refreshing in 30 seconds... (Ctrl+C to stop)")
            print(f"{'='*70}")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n✓ Monitor stopped")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            time.sleep(10)


def main():
    """Main monitor script."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        type=str,
        default='v2',
        help='Version to monitor (v1 or v2)'
    )
    
    args = parser.parse_args()
    
    log_dir = PATHS['logs'] / f'ppo_abr_{args.version}'
    checkpoint_dir = PATHS['models'] / f'ppo_abr_{args.version}' / 'checkpoints'
    
    if not log_dir.exists():
        print(f"✗ Log directory not found: {log_dir}")
        print("  Make sure training has started")
        return
    
    print_progress(log_dir, checkpoint_dir)


if __name__ == '__main__':
    main()