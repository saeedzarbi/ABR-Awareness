# scripts/evaluation/evaluate_mpc.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from models.mpc_model import MPC # ✅ ایمپورت ایجنت MPC
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 70)
print("🧪 Evaluating MPC (Model Predictive Control) on TEST Set")
print("=" * 70)

# ✅ ۱. ساخت ایجنت MPC (بدون نیاز به لود کردن مدل)
mpc_agent = MPC(future_chunks=5) # lookahead=5 یک مقدار استاندارد است

# Load environment
print("\n📦 Loading test environment...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

# از محیط اصلی خودتان استفاده می‌کنیم تا پاداش‌ها یکسان ارزیابی شوند
env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'
)
print("   ✅ Environment loaded")

# Evaluate
print("\n🧪 Running evaluation on 50 test episodes...")
print("-" * 70)

episode_rewards = []
episode_rebuffers = []
episode_bitrates = []
episode_vmafs = []

for ep in range(50):
    state = env.reset()
    episode_reward = 0
    episode_rebuffer = 0
    bitrates = []
    vmafs = []
    done = False
    
    while not done:
        # ✅ ۲. انتخاب اکشن با استفاده از ایجنت MPC
        action = mpc_agent.select_action(state)
        
        state, reward, done, info = env.step(action)
        
        # چون از محیط اصلی استفاده می‌کنیم، پاداش بر اساس VMAF است
        # که مقایسه را با مدل شما عادلانه می‌کند.
        episode_reward += reward
        episode_rebuffer += info['rebuffer_time']
        bitrates.append(info['bitrate'])
        vmafs.append(info.get('vmaf', 0)) # vmaf را برای ارجاع لاگ می‌کنیم
    
    episode_rewards.append(episode_reward)
    episode_rebuffers.append(episode_rebuffer)
    episode_bitrates.append(np.mean(bitrates))
    episode_vmafs.append(np.mean(vmafs))
    
    if (ep + 1) % 10 == 0:
        print(f"  Episode {ep+1:2d}/50: Reward={episode_reward:+7.2f}, "
              f"Rebuffer={episode_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(bitrates):6.0f}kbps, "
              f"VMAF={np.mean(vmafs):5.1f}")

print("\n" + "=" * 70)
print("📊 MPC TEST SET RESULTS (VMAF-based Reward)")
print("=" * 70)
print(f"Reward (VMAF-based):")
print(f"  Mean:  {np.mean(episode_rewards):+7.2f}")
print(f"  Std:   {np.std(episode_rewards):7.2f}")
print()
print(f"Rebuffering:")
print(f"  Mean:  {np.mean(episode_rebuffers):7.2f}s")
print()
print(f"Bitrate:")
print(f"  Mean:  {np.mean(episode_bitrates):7.0f} kbps")
print()
print(f"VMAF:")
print(f"  Mean:  {np.mean(episode_vmafs):7.1f}")
print("=" * 70)