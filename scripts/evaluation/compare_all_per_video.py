"""
scripts/evaluation/compare_all_per_video.py
============================================
Compare ALL methods with Per-Video Analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.trace_loader import TraceLoader
from models.policy_wrapper import BufferAwarePolicy, SmoothPolicy
from models.video_tracker import PerVideoTracker

print("="*100)
print("🏆 COMPLETE BASELINE COMPARISON with PER-VIDEO ANALYSIS")
print("="*100)

# ============================================
# Configuration
# ============================================
CONFIG = {
    'episodes_per_video': 10,
    'trace_dir': 'data/network_traces/cooked_traces',
    'features_file': 'data/features/si_ti_features.json',
    'vmaf_file': 'data/vmaf/vmaf_table.json',
}

VIDEO_IDS = [1, 2, 3, 4, 5, 6]
VIDEO_NAMES = {
    1: 'sports',
    2: 'animation', 
    3: 'news',
    4: 'nature',
    5: 'game',
    6: 'movie'
}

# ============================================
# Helper Function
# ============================================
def evaluate_method(policy, env, method_name, DEVICE=None):
    """Evaluate one method and return tracker"""
    tracker = PerVideoTracker()
    
    total_episodes = len(VIDEO_IDS) * CONFIG['episodes_per_video']
    pbar = tqdm(total=total_episodes, desc=f"Eval ({method_name})")
    
    for video_id in VIDEO_IDS:
        video_name = VIDEO_NAMES[video_id]
        
        for ep in range(CONFIG['episodes_per_video']):
            if hasattr(policy, 'reset'):
                policy.reset()
            
            state = env.reset(video_id=video_id, split='test')
            if state is None:
                continue
            
            ep_reward = 0
            ep_rebuffer = 0
            ep_bitrates = []
            done = False
            recent_rebuffer = 0.0
            
            while not done:
                # Get action based on policy type
                if hasattr(policy, 'select_action'):
                    # Your model or wrapped policy
                    action = policy.select_action(state, env.buffer, recent_rebuffer)
                elif DEVICE and hasattr(policy, 'forward'):
                    # Neural network policy (Pensieve)
                    with torch.no_grad():
                        net = torch.FloatTensor(state['network']).unsqueeze(0).to(DEVICE)
                        cont = torch.FloatTensor(state['content']).unsqueeze(0).to(DEVICE)
                        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(DEVICE)
                        probs, _ = policy(net, cont, vmaf)
                        action = probs.argmax(dim=1).item()
                else:
                    # Simple policy (buffer-based, etc.)
                    action = policy(state, env.buffer)
                
                state, reward, done, info = env.step(action)
                ep_reward += reward
                ep_rebuffer += info['rebuffer_time']
                ep_bitrates.append(info['bitrate'])
                recent_rebuffer = info['rebuffer_time']
            
            avg_bitrate = np.mean(ep_bitrates) if ep_bitrates else 0
            tracker.add_episode(video_name, ep_reward, ep_rebuffer, avg_bitrate)
            
            pbar.update(1)
    
    pbar.close()
    return tracker

# ============================================
# Setup
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Device: {DEVICE}")

trace_dir = CONFIG['trace_dir']
loader = TraceLoader(trace_dir=trace_dir)

env = ContentAwareEnvV2(
    trace_dir=trace_dir,
    features_file=CONFIG['features_file'],
    vmaf_file=CONFIG['vmaf_file']
)

print(f"📊 Testing configuration:")
print(f"   Videos: {len(VIDEO_IDS)}")
print(f"   Episodes per video: {CONFIG['episodes_per_video']}")
print(f"   Total episodes: {len(VIDEO_IDS) * CONFIG['episodes_per_video']}")

# Dictionary to store all trackers
all_trackers = {}

# ============================================
# 1. Your Model
# ============================================
print("\n" + "="*100)
print("📊 Evaluating: Your Model (Content-Aware)")
print("="*100)

try:
    model = ContentAwareActor(state_dim=(6,8), action_dim=6, content_dim=2).to(DEVICE)
    checkpoint = torch.load('results/fcc_training_low_lr/checkpoint_best.pth',
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    buffer_policy = BufferAwarePolicy(model)
    your_policy = SmoothPolicy(buffer_policy, max_jump=2)
    
    all_trackers['Ours'] = evaluate_method(your_policy, env, "Ours", DEVICE)
    print("✅ Your Model completed")
except Exception as e:
    print(f"❌ Error with Your Model: {e}")

# ============================================
# 2. Pensieve
# ============================================
print("\n" + "="*100)
print("📊 Evaluating: Pensieve")
print("="*100)

try:
    # Load Pensieve model
    pensieve_model = ContentAwareActor(state_dim=(6,8), action_dim=6, content_dim=2).to(DEVICE)
    
    # Try to load Pensieve checkpoint
    # Adjust path if needed
    pensieve_checkpoint = torch.load('results/pensieve/pensieve_cooked.pth',
                                    map_location=DEVICE, weights_only=False)
    pensieve_model.load_state_dict(pensieve_checkpoint['model_state_dict'])
    pensieve_model.eval()
    
    all_trackers['Pensieve'] = evaluate_method(pensieve_model, env, "Pensieve", DEVICE)
    print("✅ Pensieve completed")
except Exception as e:
    print(f"⚠️  Pensieve not available: {e}")
    print("   Skipping Pensieve...")

# ============================================
# 3. MPC (if available)
# ============================================
print("\n" + "="*100)
print("📊 Evaluating: MPC")
print("="*100)

try:
    from scripts.baselines.mpc_policy import MPCPolicy
    mpc_policy = MPCPolicy()
    all_trackers['MPC'] = evaluate_method(mpc_policy, env, "MPC")
    print("✅ MPC completed")
except Exception as e:
    print(f"⚠️  MPC not available: {e}")
    print("   Skipping MPC...")

# ============================================
# 4. Comyco (if available)
# ============================================
print("\n" + "="*100)
print("📊 Evaluating: Comyco")
print("="*100)

try:
    from scripts.baselines.comyco_policy import ComycoPolicy
    comyco_policy = ComycoPolicy()
    all_trackers['Comyco'] = evaluate_method(comyco_policy, env, "Comyco")
    print("✅ Comyco completed")
except Exception as e:
    print(f"⚠️  Comyco not available: {e}")
    print("   Skipping Comyco...")

# ============================================
# Print Individual Summaries
# ============================================
for method_name, tracker in all_trackers.items():
    print(f"\n{'='*100}")
    tracker.print_summary(title=f"{method_name} - Per-Video Results")
    
    # Save individual results
    tracker.save_to_json(f'results/per_video_{method_name.lower()}.json')
    tracker.save_to_csv(f'results/per_video_{method_name.lower()}.csv')

# ============================================
# Comparison Table
# ============================================
print("\n" + "="*120)
print("📊 COMPARISON TABLE - Reward per Video")
print("="*120)

# Header
header = f"{'Video':<15}"
for method in all_trackers.keys():
    header += f" {method:>12}"
header += f" {'Best':>12}"
print(header)
print("-"*120)

# Rows
for video in sorted(VIDEO_NAMES.values()):
    row = f"{video:<15}"
    
    video_rewards = []
    for method_name, tracker in all_trackers.items():
        if video in tracker.results:
            reward = np.mean(tracker.results[video]['rewards'])
            row += f" {reward:>+11.2f}"
            video_rewards.append((reward, method_name))
        else:
            row += f" {'N/A':>12}"
    
    # Mark best
    if video_rewards:
        best_reward, best_method = max(video_rewards, key=lambda x: x[0])
        row += f" {best_method:>12}"
    
    print(row)

print("="*120)

# ============================================
# Overall Comparison
# ============================================
print("\n" + "="*120)
print("📊 OVERALL COMPARISON")
print("="*120)

for method_name, tracker in all_trackers.items():
    stats = tracker.get_overall_stats()
    print(f"\n{method_name}:")
    print(f"   Mean Reward:      {stats['mean_reward']:+.2f} ± {stats['std_reward']:.2f}")
    print(f"   Mean Rebuffering: {stats['mean_rebuffer']:.2f}s ± {stats['std_rebuffer']:.2f}s")
    print(f"   Mean Bitrate:     {stats['mean_bitrate']:.0f} ± {stats['std_bitrate']:.0f} kbps")

# ============================================
# Save Comparison CSV
# ============================================
print("\n💾 Saving comparison CSV...")
comparison_data = []

for video in sorted(VIDEO_NAMES.values()):
    row = {'video': video}
    for method_name, tracker in all_trackers.items():
        if video in tracker.results:
            row[f'{method_name}_reward'] = np.mean(tracker.results[video]['rewards'])
            row[f'{method_name}_rebuffer'] = np.mean(tracker.results[video]['rebuffering'])
            row[f'{method_name}_bitrate'] = np.mean(tracker.results[video]['bitrates'])
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)
df.to_csv('results/per_video_comparison_all.csv', index=False)
print("   ✅ Saved: results/per_video_comparison_all.csv")

# ============================================
# Insights
# ============================================
print("\n" + "="*120)
print("💡 KEY INSIGHTS")
print("="*120)

# Find best method per video
print("\n🏆 Best Method per Video:")
for video in sorted(VIDEO_NAMES.values()):
    best_reward = -float('inf')
    best_method = None
    
    for method_name, tracker in all_trackers.items():
        if video in tracker.results:
            reward = np.mean(tracker.results[video]['rewards'])
            if reward > best_reward:
                best_reward = reward
                best_method = method_name
    
    if best_method:
        print(f"   {video:<12}: {best_method:<12} ({best_reward:+.2f})")

# Find hardest videos
print("\n⚠️  Most Challenging Videos (lowest rewards):")
video_avg_rewards = {}
for video in sorted(VIDEO_NAMES.values()):
    rewards = []
    for tracker in all_trackers.values():
        if video in tracker.results:
            rewards.append(np.mean(tracker.results[video]['rewards']))
    if rewards:
        video_avg_rewards[video] = np.mean(rewards)

sorted_videos = sorted(video_avg_rewards.items(), key=lambda x: x[1])
for video, avg_reward in sorted_videos[:3]:
    print(f"   {video:<12}: {avg_reward:+.2f} (average across methods)")

print("\n" + "="*120)
print("✅ Complete! All results saved in results/ directory")
print("="*120)