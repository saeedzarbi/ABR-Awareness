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
from models.video_tracker import PerVideoTracker  # ✅ New import

print("=" * 80)
print("🏆 FINAL TEST with PER-VIDEO ANALYSIS")
print("=" * 80)

# --- Load model ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContentAwareActor(state_dim=(6,8), action_dim=6, content_dim=2).to(DEVICE)
checkpoint_path = 'results/fcc_training_low_lr/checkpoint_best.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded: {checkpoint_path}")
except Exception as e:
    print(f"❌ Error loading: {e}")
    sys.exit(1)
model.eval()

# --- Policy wrapper ---
buffer_policy = BufferAwarePolicy(model)
policy = SmoothPolicy(buffer_policy, max_jump=2)
print("✅ Policy Wrapper enabled")

# --- Load environment ---
trace_dir = 'data/network_traces/cooked_traces'
loader = TraceLoader(trace_dir=trace_dir)

env = ContentAwareEnvV2(
    trace_dir=trace_dir,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json'
)

num_test_episodes = len(loader.test_traces)
print(f"🧪 Testing on {num_test_episodes} test traces")
print("-"*80)

# --- ✅ Create tracker ---
tracker = PerVideoTracker()

# --- Test loop ---
rewards, rebuffers, bitrates_list = [], [], []

for ep in tqdm(range(num_test_episodes), desc="Eval (Your Model)"):
    policy.reset()
    
    # ✅ Reset with random video (environment will choose)
    state = env.reset(split='test')  # video_id=None by default
    if state is None:
        continue
    
    # ✅ Get video name from environment
    video_name = env.get_video_name()
    
    ep_reward, ep_rebuffer, ep_bitrates = 0, 0, []
    done = False
    recent_rebuffer = 0.0
    
    while not done:
        action = policy.select_action(state, env.buffer, recent_rebuffer)
        state, reward, done, info = env.step(action)
        
        recent_rebuffer = info['rebuffer_time']
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
    
    avg_bitrate = np.mean(ep_bitrates) if ep_bitrates else 0
    
    # ✅ Track per video
    tracker.add_episode(video_name, ep_reward, ep_rebuffer, avg_bitrate)
    
    # Original tracking
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(avg_bitrate)

# --- ✅ Print per-video breakdown ---
tracker.print_summary()

# --- ✅ Save detailed results ---
tracker.save_to_json('results/per_video_your_model.json')

# --- Overall results ---
print("="*80)
print("📊 OVERALL RESULTS")
print("="*80)
print(f"  Mean Reward:      {np.mean(rewards):+8.2f}")
print(f"  Mean Rebuffering: {np.mean(rebuffers):8.2f}s")
print(f"  Mean Bitrate:     {np.mean(bitrates_list):8.0f} kbps")
print("="*80)