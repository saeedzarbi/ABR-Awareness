"""
تست نهایی مدل Pensieve روی مجموعه داده Cooked (TEST SET)
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from models.pensieve_actor_compatible import PensieveActorCompatible 
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.trace_loader import TraceLoader 

print("=" * 80)
print("🧪 FINAL TEST on COOKED_TRACES SET (Pensieve Model) 🧪")
print("=" * 80)

# --- بارگذاری مدل ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PensieveActorCompatible(state_dim=(6, 8), action_dim=6, content_dim=2).to(DEVICE)
checkpoint_path = 'results/pensieve_fcc_training/checkpoint_400.pth' #

try:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded: {checkpoint_path}")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)
model.eval()

# --- بارگذاری محیط ---
# --- بارگذاری محیط ---
trace_dir = 'data/network_traces/cooked_traces'
loader = TraceLoader(trace_dir=trace_dir)

mode = 'test'
env = ContentAwareEnvV2(
    trace_dir=trace_dir,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json'
)

num_test_episodes = len(loader.test_traces)
print(f"🧪 Testing on: {mode} set ({num_test_episodes} traces)...")
print("-" * 80)

# --- حلقه تست ---
rewards, rebuffers, bitrates_list = [], [], []
for ep in tqdm(range(num_test_episodes), desc="Eval (Pensieve Cooked)"):
    state = env.reset(split=mode)
    if state is None: continue
    ep_reward, ep_rebuffer, ep_bitrates = 0, 0, []
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

print("\n" + "=" * 80)
print("📊 FINAL RESULTS (Pensieve Model on COOKED_TRACES Set)")
print("=" * 80)
mean_reward = np.mean(rewards)
mean_rebuffer = np.mean(rebuffers)
mean_bitrate = np.mean(bitrates_list)
print(f"  Mean Reward (VMAF-based): {mean_reward:+8.2f}")
print(f"  Mean Rebuffering:         {mean_rebuffer:8.2f}s")
print(f"  Mean Bitrate:             {mean_bitrate:8.0f} kbps")
print("=" * 80)