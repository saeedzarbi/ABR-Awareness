import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)
checkpoint = torch.load('results/fcc_training/checkpoint_100.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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

print("Testing checkpoint 100...")
rewards = []
for ep in range(20):
    state = env.reset()
    ep_reward = 0
    done = False
    
    while not done:
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = model(net, cont, vmaf)
            action = probs.argmax(dim=1).item()
        
        state, reward, done, info = env.step(action)
        ep_reward += reward
    
    rewards.append(ep_reward)

print(f"Checkpoint 100: Mean={np.mean(rewards):+.2f}, Std={np.std(rewards):.2f}")