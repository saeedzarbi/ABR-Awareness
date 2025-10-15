import torch
import numpy as np
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

# Load model
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)
checkpoint = torch.load('results/fcc_training/checkpoint_400.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load environment
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
    mode='test'  # Test set!
)

# Evaluate
print("🧪 Evaluating on TEST set...")
episode_rewards = []

for ep in range(50):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        network_state = torch.FloatTensor(state['network']).unsqueeze(0)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf_features = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = model(network_state, content_features, vmaf_features)
            action = action_probs.argmax(dim=1).item()
        
        state, reward, done, info = env.step(action)
        episode_reward += reward
    
    episode_rewards.append(episode_reward)
    if (ep + 1) % 10 == 0:
        print(f"  Episode {ep+1}/50: {episode_reward:+.2f}")

print(f"\n📊 Test Results:")
print(f"  Mean: {np.mean(episode_rewards):+.2f}")
print(f"  Std:  {np.std(episode_rewards):.2f}")
print(f"  Min:  {np.min(episode_rewards):+.2f}")
print(f"  Max:  {np.max(episode_rewards):+.2f}")