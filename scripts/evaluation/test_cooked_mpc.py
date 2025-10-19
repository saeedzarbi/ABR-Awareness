"""
تست نهایی الگوریتم MPC روی مجموعه داده Cooked (TEST SET)
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from tqdm import tqdm

from models.mpc_model import MPC #
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.trace_loader import TraceLoader

print("=" * 80)
print("🧪 FINAL TEST on COOKED_TRACES SET (MPC Model) 🧪")
print("=" * 80)

# --- ساخت ایجنت MPC ---
mpc_agent = MPC(future_chunks=5)
print("✅ MPC Agent created.")

# --- بارگذاری محیط ---
trace_dir = 'data/network_traces/cooked_traces'
loader = TraceLoader(trace_dir=trace_dir)

mode = 'test'
# ✅ استفاده از ContentAwareEnvV2 (با پاداش VMAF برای مقایسه عادلانه)
env = ContentAwareEnvV2(
    trace_dir=trace_dir,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    reward_mode='vmaf_aware' # ارزیابی با معیار VMAF
)
num_test_episodes = len(loader.get_trace_files(split=mode))
print(f"🧪 Testing on: {mode} set ({num_test_episodes} traces)...")
print("-" * 80)

# --- حلقه تست ---
rewards, rebuffers, bitrates_list = [], [], []
for ep in tqdm(range(num_test_episodes), desc="Eval (MPC Cooked)"):
    state = env.reset(split=mode)
    if state is None: continue
    ep_reward, ep_rebuffer, ep_bitrates = 0, 0, []
    done = False
    while not done:
        action = mpc_agent.select_action(state)
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates) if ep_bitrates else 0)

print("\n" + "=" * 80)
print("📊 FINAL RESULTS (MPC Model on COOKED_TRACES Set)")
print("=" * 80)
mean_reward = np.mean(rewards)
mean_rebuffer = np.mean(rebuffers)
mean_bitrate = np.mean(bitrates_list)
print(f"  Mean Reward (VMAF-based): {mean_reward:+8.2f}")
print(f"  Mean Rebuffering:         {mean_rebuffer:8.2f}s")
print(f"  Mean Bitrate:             {mean_bitrate:8.0f} kbps")
print("=" * 80)