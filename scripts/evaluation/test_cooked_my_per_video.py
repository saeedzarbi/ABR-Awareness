"""
scripts/evaluation/test_per_video.py
=====================================
Test Your Model with Per-Video Analysis
FIXED: Ensures each video gets equal number of episodes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.trace_loader import TraceLoader
from models.policy_wrapper import BufferAwarePolicy, SmoothPolicy
from models.video_tracker import PerVideoTracker

print("="*80)
print("🏆 EVALUATING YOUR MODEL with PER-VIDEO ANALYSIS")
print("="*80)

# ============================================
# Configuration
# ============================================
CONFIG = {
    'checkpoint': 'results/fcc_training_low_lr/checkpoint_best.pth',
    'trace_dir': 'data/network_traces/cooked_traces',
    'features_file': 'data/features/si_ti_features.json',
    'vmaf_file': 'data/vmaf/vmaf_table.json',
    'output_json': 'results/per_video_your_model.json',
    'output_csv': 'results/per_video_your_model.csv',
    
    # ✅ NEW: Episodes per video
    'episodes_per_video': 10  # هر ویدیو 10 episode
}

# ============================================
# Setup
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Device: {DEVICE}")

# Load model
print("\n🧠 Loading model...")
model = ContentAwareActor(state_dim=(6,8), action_dim=6, content_dim=2).to(DEVICE)

try:
    checkpoint = torch.load(CONFIG['checkpoint'], map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✅ Loaded: {CONFIG['checkpoint']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

model.eval()

# Setup policy wrapper
buffer_policy = BufferAwarePolicy(model)
policy = SmoothPolicy(buffer_policy, max_jump=2)
print("   ✅ Policy wrapper enabled (BufferAware + Smooth)")

# Load environment
print("\n🌍 Setting up environment...")
trace_dir = CONFIG['trace_dir']
loader = TraceLoader(trace_dir=trace_dir)

env = ContentAwareEnvV2(
    trace_dir=trace_dir,
    features_file=CONFIG['features_file'],
    vmaf_file=CONFIG['vmaf_file']
)

num_test_traces = len(loader.test_traces)
print(f"   ✅ Loaded {num_test_traces} test traces")

# ============================================
# Create Video Tracker
# ============================================
tracker = PerVideoTracker()

# ============================================
# Evaluation Loop - FIXED VERSION
# ============================================
print("\n🧪 Running evaluation...")
print(f"   Testing each video with {CONFIG['episodes_per_video']} episodes")
print(f"   Total episodes: {6 * CONFIG['episodes_per_video']}")
print("-"*80)

rewards, rebuffers, bitrates_list = [], [], []

# ✅ NEW: Test each video separately with equal episodes
video_ids = [1, 2, 3, 4, 5, 6]
video_names_map = {
    1: 'sports',
    2: 'animation', 
    3: 'news',
    4: 'nature',
    5: 'game',
    6: 'movie'
}

total_episodes = len(video_ids) * CONFIG['episodes_per_video']
pbar = tqdm(total=total_episodes, desc="Evaluating")

for video_id in video_ids:
    video_name = video_names_map[video_id]
    
    for ep in range(CONFIG['episodes_per_video']):
        policy.reset()
        
        # ✅ Reset with specific video
        state = env.reset(video_id=video_id, split='test')
        if state is None:
            continue
        
        # Verify video
        assert env.get_video_name() == video_name, "Video mismatch!"
        
        # Episode tracking
        ep_reward = 0
        ep_rebuffer = 0
        ep_bitrates = []
        done = False
        recent_rebuffer = 0.0
        
        # Episode loop
        while not done:
            # Select action
            action = policy.select_action(state, env.buffer, recent_rebuffer)
            
            # Step
            state, reward, done, info = env.step(action)
            
            # Track metrics
            recent_rebuffer = info['rebuffer_time']
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_bitrates.append(info['bitrate'])
        
        # Calculate average bitrate
        avg_bitrate = np.mean(ep_bitrates) if ep_bitrates else 0
        
        # Add to per-video tracker
        tracker.add_episode(video_name, ep_reward, ep_rebuffer, avg_bitrate)
        
        # Add to overall tracking
        rewards.append(ep_reward)
        rebuffers.append(ep_rebuffer)
        bitrates_list.append(avg_bitrate)
        
        pbar.update(1)

pbar.close()

# ============================================
# Results
# ============================================

# Print per-video breakdown
tracker.print_summary(title="YOUR MODEL - Per-Video Results (Balanced)")

# Save results
print("\n💾 Saving results...")
tracker.save_to_json(CONFIG['output_json'])
tracker.save_to_csv(CONFIG['output_csv'])

# Print overall summary
print("\n" + "="*80)
print("📊 OVERALL RESULTS (60 episodes balanced across 6 videos)")
print("="*80)
print(f"  Mean Reward:      {np.mean(rewards):+8.2f} ± {np.std(rewards):.2f}")
print(f"  Mean Rebuffering: {np.mean(rebuffers):8.2f}s ± {np.std(rebuffers):.2f}s")
print(f"  Mean Bitrate:     {np.mean(bitrates_list):8.0f} ± {np.std(bitrates_list):.0f} kbps")
print("="*80)

# ✅ Per-video comparison
print("\n📊 Per-Video Comparison:")
print("-"*80)
df = tracker.get_summary_df()
for _, row in df.iterrows():
    improvement = row['mean_reward'] - df['mean_reward'].mean()
    symbol = "✅" if improvement > 0 else "⚠️"
    print(f"  {symbol} {row['video']:<12}: Reward {row['mean_reward']:+7.2f} "
          f"(Δ {improvement:+5.2f} vs avg)")

print("\n✅ Done! Check the following files:")
print(f"   📄 {CONFIG['output_json']}")
print(f"   📄 {CONFIG['output_csv']}")