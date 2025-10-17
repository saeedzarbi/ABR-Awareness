import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🧪 Test checkpoint_100")
print("=" * 80)
print()

# بارگذاری مدل
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)

try:
    checkpoint = torch.load('results/fcc_training/checkpoint_100.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint_100")
    print(f"   Update: {checkpoint['update']}")
except FileNotFoundError:
    print("❌ checkpoint_100 not found!")
    print("\nTrying checkpoint_400...")
    try:
        checkpoint = torch.load('results/fcc_training/checkpoint_400.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint_400")
        print(f"   Update: {checkpoint['update']}")
    except:
        print("❌ No checkpoints found!")
        sys.exit(1)

model.eval()
print()

# بارگذاری environment
print("📦 Loading environment...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

print(f"✅ Traces loaded:")
print(f"   Train: {len(loader.train_traces)}")
print(f"   Val: {len(loader.val_traces)}")
print(f"   Test: {len(loader.test_traces)}")
print()

# استفاده از val اگر test کم باشه
if len(loader.test_traces) < 20:
    print("⚠️  Using validation set (test < 20)")
    mode = 'val'
else:
    mode = 'test'

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode=mode
)

print(f"Mode: {mode}")
print()

# تست
print("🧪 Running 20 test episodes...")
print("-" * 80)

rewards = []
rebuffers = []
bitrates_list = []

for ep in range(20):
    # Reset با بررسی
    state = env.reset()
    
    if state is None:
        print(f"❌ Episode {ep+1}: reset() returned None!")
        continue
    
    ep_reward = 0
    ep_rebuffer = 0
    ep_bitrates = []
    done = False
    steps = 0
    
    while not done:
        try:
            # State to tensors
            net = torch.FloatTensor(state['network']).unsqueeze(0)
            cont = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            # Action
            with torch.no_grad():
                probs, _ = model(net, cont, vmaf)
                action = probs.argmax(dim=1).item()
            
            # Step
            state, reward, done, info = env.step(action)
            
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_bitrates.append(info['bitrate'])
            steps += 1
            
            if steps > 100:  # safety limit
                print(f"⚠️  Episode {ep+1}: exceeded 100 steps, breaking")
                break
                
        except Exception as e:
            print(f"❌ Episode {ep+1} error at step {steps}: {e}")
            break
    
    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates) if ep_bitrates else 0)
    
    if (ep + 1) % 5 == 0:
        print(f"Episode {ep+1:2d}/20: Reward={ep_reward:+8.2f}, "
              f"Rebuffer={ep_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(ep_bitrates) if ep_bitrates else 0:6.0f}kbps")

print()
print("=" * 80)
print("📊 Results")
print("=" * 80)
print(f"Reward:      {np.mean(rewards):+8.2f} ± {np.std(rewards):.2f}")
print(f"Rebuffering: {np.mean(rebuffers):8.2f}s")
print(f"Bitrate:     {np.mean(bitrates_list):8.0f} kbps")
print()

baseline = 102.16
improvement = ((np.mean(rewards) - baseline) / baseline) * 100
print(f"vs BBA: {improvement:+.1f}%")
print("=" * 80)