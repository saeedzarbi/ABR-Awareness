"""
Real-time monitoring
"""
import json
import time
import os

log_file = 'results/fcc_training_low_entropy/training_log.json'

print("=" * 80)
print("📊 Real-time Training Monitor")
print("=" * 80)
print()

last_update = 0

while True:
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log = json.load(f)
            
            if len(log) > last_update:
                # Clear screen (optional)
                # os.system('clear')
                
                latest = log[-1]
                update = latest['update']
                
                # نمایش آخرین 10 update
                print("\n" + "=" * 80)
                print(f"📊 Training Progress - Update {update}")
                print("=" * 80)
                print()
                
                # آخرین metrics
                print(f"Last Update:")
                print(f"  Reward:       {latest['mean_reward']:+8.2f}")
                print(f"  Policy Loss:  {latest['policy_loss']:8.4f}")
                print(f"  Entropy:      {latest['entropy']:8.4f}")
                print(f"  Episodes:     {latest['n_episodes']:8d}")
                print(f"  Update Time:  {latest['update_time']:8.1f}s")
                print()
                
                # Validation (اگه باشه)
                if 'val_reward_mean' in latest:
                    marker = "🏆" if latest.get('new_best') else "  "
                    print(f"Validation:")
                    print(f"  {marker} Reward: {latest['val_reward_mean']:+.2f} ± {latest['val_reward_std']:.2f}")
                    print()
                
                # Progress
                recent = log[-10:]
                rewards = [x['mean_reward'] for x in recent]
                print(f"Recent 10 Updates:")
                print(f"  Mean:   {sum(rewards)/len(rewards):+.2f}")
                print(f"  Min:    {min(rewards):+.2f}")
                print(f"  Max:    {max(rewards):+.2f}")
                print()
                
                # ETA
                elapsed = latest['elapsed_time']
                total_updates = 500
                eta = (elapsed / update) * (total_updates - update)
                print(f"Progress:")
                print(f"  {update}/{total_updates} updates ({update/total_updates*100:.1f}%)")
                print(f"  Elapsed: {elapsed/60:.1f} min")
                print(f"  ETA:     {eta/60:.1f} min")
                print()
                
                last_update = len(log)
        
        time.sleep(10)  # هر 10 ثانیه چک کن
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitor stopped")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
