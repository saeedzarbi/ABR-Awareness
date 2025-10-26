# # فایل: test_with_safety.py

# import torch
# import numpy as np
# from models.content_aware_model import create_content_aware_model
# from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
# from models.fcc_trace_loader import FCCTraceLoader

# loader = FCCTraceLoader(
#     fcc_trace_dir='data/fcc_traces',
#     train_file='data/network_traces/fcc/splits/fcc_train.txt',
#     val_file='data/network_traces/fcc/splits/fcc_val.txt',
#     test_file='data/network_traces/fcc/splits/fcc_test.txt'
# )

# env_test = ContentAwareEnvFCC(
#     fcc_trace_loader=loader,
#     features_file='data/features/si_ti_features.json',
#     vmaf_file='data/vmaf/vmaf_table.json',
#     video_dir='data/videos',
#     mode='test'
# )

# model = create_content_aware_model()
# checkpoint = torch.load('results/fcc_training_improved_s/checkpoint_best.pth')
# model.load_state_dict(checkpoint['model_state_dict'])

# def evaluate_with_safety(env, model, n_episodes=30):
#     rewards = []
#     for i in range(n_episodes):
#         state = env.reset()
#         episode_reward = 0
#         done = False
        
#         while not done:
#             net = torch.FloatTensor(state['network']).unsqueeze(0)
#             cont = torch.FloatTensor(state['content']).unsqueeze(0)
#             vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
#             with torch.no_grad():
#                 action_probs, _ = model(net, cont, vmaf)
            
#             action = action_probs.argmax(dim=1).item()
            
#             # ✅ Safety Wrapper
#             buffer = env.buffer
#             if buffer < 5.0:
#                 action = min(action, 1)
#             elif buffer < 10.0:
#                 action = min(action, 2)
#             elif buffer < 20.0:
#                 action = min(action, 3)
            
#             state, reward, done, info = env.step(action)
#             episode_reward += reward
        
#         rewards.append(episode_reward)
#         if (i+1) % 5 == 0:
#             print(f"  [{i+1:2d}/30] Mean: {np.mean(rewards):+.2f}")
    
#     return np.mean(rewards), np.std(rewards)

# print("🧪 Test با Safety Wrapper...")
# test_mean, test_std = evaluate_with_safety(env_test, model, 30)

# print(f"\n📊 نتایج:")
# print(f"   بدون Safety: +77.76")
# print(f"   با Safety:    {test_mean:+.2f} ± {test_std:.2f}")
# print(f"   Baseline:     +102.16")
# test_checkpoint.py

import torch
import numpy as np
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env_test = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'
)

def quick_test(checkpoint_path):
    model = create_content_aware_model()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    rewards = []
    for _ in range(10):  # سریع: فقط 10 episode
        state = env_test.reset()
        ep_reward = 0
        done = False
        
        while not done:
            net = torch.FloatTensor(state['network']).unsqueeze(0)
            cont = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = model(net, cont, vmaf)
            action = action_probs.argmax(dim=1).item()
            state, reward, done, _ = env.step(action)
            ep_reward += reward
        
        rewards.append(ep_reward)
    
    return np.mean(rewards)

# تست checkpoints مختلف
checkpoints = [
    'results/fcc_training_improved/checkpoint_best.pth',
    'results/fcc_training_improved/checkpoint_100.pth',
    'results/fcc_training_improved_s/checkpoint_best.pth',
]

print("🔍 Testing different checkpoints:\n")
for ckpt in checkpoints:
    try:
        result = quick_test(ckpt)
        print(f"{ckpt:60s} → {result:+.2f}")
    except:
        print(f"{ckpt:60s} → NOT FOUND")

print("\n✅ استفاده کنید از checkpoint با بهترین test result")