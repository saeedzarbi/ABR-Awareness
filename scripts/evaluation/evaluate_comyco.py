# scripts/evaluation/evaluate_comyco.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm

# مدل ما همان ContentAwareActor است که با روش تقلیدی آموزش دیده
from models.content_aware_model import ContentAwareActor
# برای ارزیابی عادلانه، از محیط اصلی (با پاداش VMAF) استفاده می‌کنیم
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 70)
print("🧪 Evaluating Comyco-style Model on TEST Set")
print("=" * 70)

# ۱. لود کردن مدل آموزش‌دیده با یادگیری تقلیدی
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2).to(device)
model_path = 'results/comyco_model.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"   ✅ Loaded trained Comyco model from '{model_path}'")
except Exception as e:
    print(f"   ❌ Could not load model! Have you run 'train_comyco.py' first? Error: {e}")
    sys.exit(1)

model.eval()

# ۲. لود کردن محیط تست (محیط اصلی شما با پاداش VMAF)
print("\n📦 Loading test environment (VMAF-based reward)...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env = ContentAwareEnvFCC(
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

for ep in tqdm(range(num_test_episodes), desc="Evaluating Comyco Model"):
    state = env.reset(video_id=np.random.randint(1, 7)) # ویدئوهای مختلف را تست می‌کنیم
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
            action = action_probs.argmax(dim=1).item() # انتخاب قطعی بهترین اکشن
        
        state, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_rebuffer += info['rebuffer_time']
        bitrates.append(info['bitrate'])
        vmafs.append(info.get('vmaf', 0))
    
    episode_rewards.append(episode_reward)
    episode_rebuffers.append(episode_rebuffer)
    episode_bitrates.append(np.mean(bitrates) if bitrates else 0)
    episode_vmafs.append(np.mean(vmafs) if vmafs else 0)

# ۴. نمایش نتایج نهایی
print("\n" + "=" * 70)
print("📊 COMYCO-STYLE MODEL TEST SET RESULTS")
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