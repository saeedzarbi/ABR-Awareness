"""
تست الگوریتم MPC روی مجموعه اعتبارسنجی (Validation Set)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) # فرض می‌کنیم در scripts/evaluation/ است

import numpy as np
from tqdm import tqdm

from models.mpc_model import MPC # <-- ایمپورت MPC
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🧪 Testing MPC Model on VALIDATION Set")
print("=" * 80)
print()

# --- ساخت ایجنت MPC ---
# (از فایل mpc_model.py شما)
mpc_agent = MPC(future_chunks=5)
print("✅ MPC Agent created.")
print()

# --- بارگذاری محیط ---
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

# ✅ ارزیابی روی مجموعه اعتبارسنجی
mode = 'val'

# از محیط اصلی (با پاداش VMAF) برای مقایسه عادلانه پاداش استفاده می‌کنیم
env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode=mode
)

num_test_episodes = len(loader.val_traces)
print(f"🧪 Testing on: {mode} set ({num_test_episodes} episodes)...")
print("-" * 80)

# --- حلقه تست ---
rewards = []
rebuffers = []
bitrates_list = []

for ep in tqdm(range(num_test_episodes), desc="Evaluating MPC (Val)"):
    state = env.reset()
    if state is None:
        continue
    
    ep_reward = 0
    ep_rebuffer = 0
    ep_bitrates = []
    done = False
    
    while not done:
        # انتخاب اکشن توسط MPC
        action = mpc_agent.select_action(state)
        
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
    
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates) if ep_bitrates else 0)

print()
print("=" * 80)
print("📊 FINAL RESULTS - MPC Model (on Validation Set)")
print("=" * 80)
print()

mean_reward = np.mean(rewards)
mean_rebuffer = np.mean(rebuffers)
mean_bitrate = np.mean(bitrates_list)

print(f"  Mean Reward (VMAF-based): {mean_reward:+8.2f}")
print(f"  Mean Rebuffering:         {mean_rebuffer:8.2f}s")
print(f"  Mean Bitrate:             {mean_bitrate:8.0f} kbps")
print()

print("=" * 80)
print("🎯 COMPARISON (on Validation Set)")
print("=" * 80)

# مقایسه با بیس‌لاین‌های مجموعه اعتبارسنجی
bba = 68.24
print(f"{'Method':<30} {'Reward':>12}")
print("-" * 80)
print(f"{'BBA Baseline (Val)':<30} {bba:>+12.2f}")
print(f"{'MPC Model (Val)':<30} {mean_reward:>+12.2f}")
print("=" * 80)