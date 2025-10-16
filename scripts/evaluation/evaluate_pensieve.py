# scripts/evaluation/evaluate_pensieve.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm

# ایمپورت مدل و محیط
from models.pensieve_actor_compatible import PensieveActorCompatible
from models.pensieve_env_fcc import PensieveEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 70)
print("🧪 Evaluating Pensieve (Content-BLIND) Model on TEST Set")
print("=" * 70)

# ۱. لود کردن مدل Pensieve
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PensieveActorCompatible(state_dim=(6, 8), action_dim=6, content_dim=2).to(device)
checkpoint_path = 'results/pensieve_fcc_training/checkpoint_400.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    update_num = checkpoint.get('update', 'N/A')
    print(f"   ✅ Loaded checkpoint from {checkpoint_path} (Update {update_num})")
except Exception as e:
    print(f"   ❌ Could not load model! Have you run 'train_pensieve.py' first? Error: {e}")
    sys.exit(1)

model.eval()

# ۲. لود کردن محیط تست Pensieve
print("\n📦 Loading test environment (Pensieve)...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env = PensieveEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'
)
print("   ✅ Environment loaded")

# ۳. اجرای ارزیابی
print("\n🧪 Running evaluation on all test episodes...")
print("-" * 70)

episode_rewards = []
episode_rebuffers = []
episode_bitrates = []
episode_vmafs = []

num_test_episodes = len(loader.test_traces)

for ep in tqdm(range(num_test_episodes), desc="Evaluating Pensieve Model"):
    state = env.reset(video_id=np.random.randint(1, 7))
    episode_reward = 0
    episode_rebuffer = 0
    bitrates = []
    vmafs = []
    done = False
    
    while not done:
        network_state = torch.FloatTensor(state['network']).unsqueeze(0).to(device)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0).to(device)
        vmaf_features = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, _ = model(network_state, content_features, vmaf_features)
            action = action_probs.argmax(dim=1).item()
        
        state, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_rebuffer += info['rebuffer_time']
        bitrates.append(info['bitrate'])
        
        # =======================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # !!           تغییر اصلی و رفع خطا اینجاست           !!
        #
        # از .get() استفاده می‌کنیم تا اگر کلید 'vmaf' وجود نداشت،
        # برنامه متوقف نشود و مقدار 0 را برگرداند.
        vmafs.append(info.get('vmaf', 0.0))
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # =======================================================

    episode_rewards.append(episode_reward)
    episode_rebuffers.append(episode_rebuffer)
    episode_bitrates.append(np.mean(bitrates) if bitrates else 0)
    episode_vmafs.append(np.mean(vmafs) if vmafs else 0)

# ۴. نمایش نتایج نهایی
print("\n" + "=" * 70)
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
print(f"V