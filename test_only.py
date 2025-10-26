# فایل جدید: test_only.py

import torch
import numpy as np
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🎯 Test Only - 30 Episodes")
print("=" * 80)

# Load data
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

# Create test environment
env_test = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'
)

# Load model
model = create_content_aware_model()
checkpoint = torch.load('results/fcc_training_improved_s/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded checkpoint (update {checkpoint['update']})")

# Evaluate
def evaluate(env, model, n_episodes=30):
    rewards = []
    for i in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            net = torch.FloatTensor(state['network']).unsqueeze(0)
            cont = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = model(net, cont, vmaf)
            
            action = action_probs.argmax(dim=1).item()
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        if (i+1) % 5 == 0:
            print(f"  [{i+1:2d}/30] Mean: {np.mean(rewards):+.2f}")
    
    return np.mean(rewards), np.std(rewards)

test_mean, test_std = evaluate(env_test, model, n_episodes=30)

print(f"\n📊 Final Result:")
print(f"   Test:     {test_mean:+.2f} ± {test_std:.2f}")
print(f"   Val:      {checkpoint['val_reward']:+.2f}")
print(f"   Baseline: +102.16")

if test_mean > 102.16:
    print(f"\n✅ EXCELLENT!")
elif test_mean > 100:
    print(f"\n✅ GOOD!")
else:
    print(f"\n⚠️  ACCEPTABLE")

print("=" * 80)