"""
Real-time Monitor برای Training بهبود یافته
"""

import json
import time
import os
from datetime import datetime
import numpy as np

def clear_screen():
    """پاک کردن صفحه"""
    os.system('clear' if os.name != 'nt' else 'cls')

def plot_ascii(values, width=60, height=8, title=""):
    """نمودار ASCII ساده"""
    if not values or len(values) < 2:
        return "Not enough data yet..."
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val - min_val < 0.1:
        return "No variation yet..."
    
    # Normalize
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Sample اگه خیلی زیاده
    if len(normalized) > width:
        step = len(normalized) / width
        normalized = [normalized[int(i * step)] for i in range(width)]
    
    # ساخت plot
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
    
    # اضافه کردن axis
    result = ""
    if title:
        result += f"{title}\n"
    result += f"Max: {max_val:+7.1f} │" + lines[0] + "\n"
    for line in lines[1:-1]:
        result += "             │" + line + "\n"
    result += f"Min: {min_val:+7.1f} │" + lines[-1]
    
    return result

def monitor_training():
    """مانیتور کردن training"""
    
    log_file = 'results/fcc_training_improved/training_log.json'
    
    print("=" * 80)
    print("📊 Live Training Monitor")
    print("=" * 80)
    print(f"Watching: {log_file}")
    print("Press Ctrl+C to stop")
    print()
    
    # منتظر بمان تا فایل log بسازه
    if not os.path.exists(log_file):
        print("⏳ Waiting for training to start...")
        while not os.path.exists(log_file):
            time.sleep(2)
        print("✅ Training started!\n")
        time.sleep(1)
    
    last_update = 0
    
    try:
        while True:
            try:
                # خواندن log
                with open(log_file, 'r') as f:
                    training_log = json.load(f)
                
                if not training_log:
                    print("⏳ Waiting for data...")
                    time.sleep(5)
                    continue
                
                # آخرین update
                latest = training_log[-1]
                current_update = latest['update']
                
                # فقط اگه update جدید باشه نمایش بده
                if current_update > last_update:
                    last_update = current_update
                    
                    clear_screen()
                    
                    # Header
                    print("=" * 80)
                    print(f"🚀 IMPROVED TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
                    print("=" * 80)
                    print()
                    
                    # Progress
                    progress = (current_update / 500) * 100  # فرض: max 500 updates
                    bar_length = 40
                    filled = int(bar_length * progress / 100)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    
                    print(f"📈 PROGRESS: [{bar}] {progress:.1f}%")
                    print(f"   Update: {current_update}")
                    print(f"   Elapsed: {latest['elapsed_time']/60:.1f} min")
                    print()
                    
                    # Current metrics
                    print("🎯 CURRENT METRICS:")
                    print(f"   Reward:       {latest['mean_reward']:>+8.2f}")
                    print(f"   Policy Loss:  {latest['policy_loss']:>8.4f}")
                    print(f"   Value Loss:   {latest['value_loss']:>8.2f}")
                    print(f"   Entropy:      {latest['entropy']:>8.4f}")
                    if 'entropy_coef' in latest:
                        print(f"   Entropy Coef: {latest['entropy_coef']:>8.4f}")
                    print(f"   Episodes:     {latest['n_episodes']:>8d}")
                    print()
                    
                    # Validation (اگه باشه)
                    if 'val_reward_mean' in latest:
                        marker = "🏆" if latest.get('new_best') else "  "
                        print(f"{marker} VALIDATION:")
                        print(f"   Reward:  {latest['val_reward_mean']:>+8.2f} ± {latest['val_reward_std']:>6.2f}")
                        
                        # Early stopping status
                        if 'no_improvement_count' in latest:
                            print(f"   No improvement: {latest['no_improvement_count']}/5")
                        print()
                    
                    # Recent performance (آخرین 20 update)
                    recent = training_log[-20:]
                    recent_rewards = [u['mean_reward'] for u in recent]
                    
                    if len(recent_rewards) >= 2:
                        print("📊 RECENT PERFORMANCE (last 20 updates):")
                        print(f"   Mean:   {np.mean(recent_rewards):>+8.2f}")
                        print(f"   Std:    {np.std(recent_rewards):>8.2f}")
                        print(f"   Min:    {np.min(recent_rewards):>+8.2f}")
                        print(f"   Max:    {np.max(recent_rewards):>+8.2f}")
                        
                        # Trend
                        if len(recent_rewards) >= 10:
                            first_half = np.mean(recent_rewards[:10])
                            second_half = np.mean(recent_rewards[10:])
                            trend = "📈 Improving" if second_half > first_half else "📉 Declining"
                            print(f"   Trend:  {trend}")
                        print()
                    
                    # Learning curve
                    all_updates = training_log[-100:] if len(training_log) > 100 else training_log
                    all_rewards = [u['mean_reward'] for u in all_updates]
                    
                    if len(all_rewards) >= 2:
                        print(plot_ascii(all_rewards, width=60, height=8, 
                                       title="📈 REWARD CURVE (last 100 updates):"))
                        print()
                    
                    # Entropy curve
                    all_entropy = [u['entropy'] for u in all_updates]
                    if len(all_entropy) >= 2:
                        print(plot_ascii(all_entropy, width=60, height=6,
                                       title="🔄 ENTROPY CURVE:"))
                        print()
                    
                    # Status
                    print("=" * 80)
                    print("💡 IMPROVEMENTS:")
                    print("   ✅ Data Augmentation active (50% probability)")
                    print("   ✅ Early Stopping enabled (patience=5)")
                    print("   ✅ Entropy decay active (0.995 per update)")
                    print()
                    
                    # Comparison
                    baseline = 102.16
                    if latest['mean_reward'] > baseline * 0.9:
                        status = "🏆 Near/Above baseline!"
                    elif latest['mean_reward'] > 50:
                        status = "✅ Good progress"
                    elif latest['mean_reward'] > 0:
                        status = "⚠️  Learning..."
                    else:
                        status = "❌ Struggling"
                    
                    print(f"📊 Status: {status}")
                    print(f"   Current: {latest['mean_reward']:+.2f}")
                    print(f"   BBA Baseline: +102.16")
                    print(f"   Gap: {latest['mean_reward'] - baseline:+.2f}")
                    print()
                    
                    print("=" * 80)
                    print("⏰ Refreshing in 10 seconds... (Ctrl+C to stop)")
                    
                else:
                    print(f"⏳ Waiting for update {last_update + 1}...")
                
            except json.JSONDecodeError:
                print("⚠️  Log file being written, retrying...")
            except FileNotFoundError:
                print("⚠️  Log file not found, waiting...")
            
            time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n\n")
        print("=" * 80)
        print("⏸️  Monitoring Stopped")
        print("=" * 80)
        
        if training_log:
            latest = training_log[-1]
            print(f"\nFinal Status:")
            print(f"  Updates completed: {latest['update']}")
            print(f"  Latest reward: {latest['mean_reward']:+.2f}")
            print(f"  Time elapsed: {latest['elapsed_time']/60:.1f} min")
            
            if 'val_reward_mean' in latest:
                print(f"  Last validation: {latest['val_reward_mean']:+.2f}")
        
        print("\n✅ Monitor closed")

if __name__ == '__main__':
    monitor_training()