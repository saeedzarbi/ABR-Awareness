"""
تست مدل Comyco (تقلیدی) روی مجموعه اعتبارسنجی (Validation Set)
(بارگذاری مدل از train_comyco.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) # فرض می‌کنیم در scripts/evaluation/ است

import torch
import numpy as np
from tqdm import tqdm

# مدل Comyco از همان معماری مدل اصلی شما استفاده می‌کند
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🧪 Testing Comyco-style Model on VALIDATION Set")
print("=" * 80)
print()

# --- بارگذاری مدل ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2).to(DEVICE)

# این مدل از اسکریپت train_comyco.py می‌آید
checkpoint_path = 'results/comyco_model.pth' 

try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"✅ Loaded: {checkpoint_path}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("   آیا 'train_comyco.py' اجرا و تمام شده است؟")
    sys.exit(1)

model.eval()
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

for ep in tqdm(range(num_test_episodes), desc="Evaluating Comyco (Val)"):
    state = env.reset()
    if state is None:
        continue
    
    ep_reward = 0
    ep_rebuffer = 0
    ep_bitrates = []
    done = False
    
    while not done:
        net = torch.FloatTensor(state['network']).unsqueeze(0).to(DEVICE)
        cont = torch.FloatTensor(state['content']).unsqueeze(0).to(DEVICE)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            probs, _ = model(net, cont, vmaf)
            action = probs.argmax(dim=1).item()
        
        state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
    
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates) if ep_bitrates else 0)

print()
print("=" * 80)
print("📊 FINAL RESULTS - Comyco Model (on Validation Set)")
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
print(f"{'Comyco Model (Val)':<30} {mean_reward:>+12.2f}")
print("=" * 80)