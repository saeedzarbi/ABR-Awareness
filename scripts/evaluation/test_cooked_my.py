"""
تست بهترین مدل شما (low_lr) روی مجموعه داده Cooked (TEST SET)
همراه با Wrapper پیشرفته
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_v2 import ContentAwareEnvV2 
from models.trace_loader import TraceLoader 
from models.policy_wrapper import BufferAwarePolicy, SmoothPolicy

print("=" * 80)
print("🏆 FINAL TEST on COOKED_TRACES SET (Your Model + Advanced Wrapper) 🏆")
print("   (Model: 'fcc_training_low_lr/checkpoint_best.pth')")
print("=" * 80)

# --- بارگذاری مدل ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2).to(DEVICE)
checkpoint_path = 'results/fcc_training/checkpoint_400.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded: {checkpoint_path}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)
model.eval()

# --- ساخت Wrapper ---
buffer_policy = BufferAwarePolicy(model)
policy = SmoothPolicy(buffer_policy, max_jump=2) 
print("✅ Advanced Policy Wrapper (BufferAware + Smooth) enabled.")

# --- بارگذاری محیط ---
trace_dir = 'data/network_traces/cooked_traces'
loader = TraceLoader(trace_dir=trace_dir)

mode = 'test'
# ✅✅✅ اصلاح شد: 'reward_mode' حذف شد. 
# ContentAwareEnvV2 به طور پیش‌فرض از پاداش VMAF استفاده می‌کند.
env = ContentAwareEnvV2(
    trace_dir=trace_dir,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json'
)
# ✅✅✅ اصلاح شد: استفاده از متد صحیح get_trace_files
num_test_episodes = len(loader.get_trace_files(split=mode))
print(f"🧪 Testing on: {mode} set ({num_test_episodes} traces)...")
print("-" * 80)

# --- حلقه تست ---
rewards, rebuffers, bitrates_list = [], [], []
for ep in tqdm(range(num_test_episodes), desc="Eval (Your Model Cooked)"):
    policy.reset()
    state = env.reset(split=mode) 
    if state is None: continue
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
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates) if ep_bitrates else 0)

print("\n" + "=" * 80)
print("📊 FINAL RESULTS (Your Model + Wrapper on COOKED_TRACES Set)")
print("=" * 80)
mean_reward = np.mean(rewards)
mean_rebuffer = np.mean(rebuffers)
mean_bitrate = np.mean(bitrates_list)
print(f"  Mean Reward:        {mean_reward:+8.2f}")
print(f"  Mean Rebuffering:   {mean_rebuffer:8.2f}s")
print(f"  Mean Bitrate:       {mean_bitrate:8.0f} kbps")
print("=" * 80)