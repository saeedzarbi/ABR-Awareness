# scripts/evaluation/evaluate_pensieve.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

# ✅ تغییر ۱: ایمپورت مدل و محیط جدید
from models.pensieve_actor_compatible import PensieveActorCompatible
from models.pensieve_env_fcc import PensieveEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 70)
# ✅ تغییر ۲: تغییر عنوان
print("🧪 Evaluating Pensieve (Content-BLIND) Model on TEST Set")
print("=" * 70)

# ✅ تغییر ۳: استفاده از مدل Pensieve
print("\n📦 Loading model...")
model = PensieveActorCompatible(state_dim=(6, 8), action_dim=6, content_dim=2)

try:
    # ✅ تغییر ۴: لود کردن چک‌پوینت از مسیر جدید Pensieve
    checkpoint_path = 'results/pensieve_fcc_training/checkpoint_400.pth' # یا هر چک‌پوینتی
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    update_num = checkpoint.get('update', 'N/A')
    print(f"   ✅ Loaded checkpoint from {checkpoint_path} (Update {update_num})")
except Exception as e:
    print(f"   ❌ No checkpoint found at {checkpoint_path}! Error: {e}")
    sys.exit(1)

model.eval()

# Load environment
print("\n📦 Loading test environment (Pensieve)...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

# ✅ تغییر ۵: استفاده از محیط Pensieve
env = PensieveEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'  # Test set!
)
print("   ✅ Environment loaded")


# --- بقیه اسکریپت را از فایل اصلی کپی کنید ---
# (حلقه ارزیابی و نمایش نتایج)
print("\n🧪 Running evaluation on 50 test episodes...")
print("-" * 70)

episode_rewards = []
episode_rebuffers = []
episode_bitrates = []
episode_vmafs = [] # ! بیایید VMAF را هم لاگ کنیم

for ep in range(50):
    state = env.reset()
    episode_reward = 0
    episode_rebuffer = 0
    bitrates = []
    vmafs = [] # !
    done = False
    
    while not done:
        network_state = torch.FloatTensor(state['network']).unsqueeze(0)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf_features = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            # مدل Pensieve ما هر سه ورودی را می‌پذیرد (اما فقط از اولی استفاده می‌کند)
            action_probs, _ = model(network_state, content_features, vmaf_features)
            action = action_probs.argmax(dim=1).item()
        
        # محیط Pensieve ما پاداش مبتنی بر بیت‌ریت را برمی‌گرداند
        state, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_rebuffer += info['rebuffer_time']
        bitrates.append(info['bitrate'])
        vmafs.append(info['vmaf']) # !
    
    episode_rewards.append(episode_reward)
    episode_rebuffers.append(episode_rebuffer)
    episode_bitrates.append(np.mean(bitrates))
    episode_vmafs.append(np.mean(vmafs)) # !
    
    if (ep + 1) % 10 == 0:
        print(f"  Episode {ep+1:2d}/50: Reward={episode_reward:+7.2f}, "
              f"Rebuffer={episode_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(bitrates):6.0f}kbps, "
              f"VMAF={np.mean(vmafs):5.1f}") # !

print("\n" + "=" * 70)
# ✅ تغییر ۶: تغییر عنوان نتایج
print("📊 PENSIVE (Bitrate-based Reward) TEST SET RESULTS")
print("=" * 70)
print(f"Reward (Bitrate-based):")
print(f"  Mean:  {np.mean(episode_rewards):+7.2f}")
print(f"  Std:   {np.std(episode_rewards):7.2f}")
print()
print(f"Rebuffering:")
print(f"  Mean:  {np.mean(episode_rebuffers):7.2f}s")
print()
print(f"Bitrate:")
print(f"  Mean:  {np.mean(episode_bitrates):7.0f} kbps")
print()
print(f"VMAF (for reference):") # !
print(f"  Mean:  {np.mean(episode_vmafs):7.1f}") # !
print("=" * 70)