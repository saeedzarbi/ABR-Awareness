"""
تست تمام checkpoint ها
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import os
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 80)
print("🔍 Testing ALL Checkpoints")
print("=" * 80)
print()

# Safety Wrapper
class SafetyWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def select_action(self, state, buffer):
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = self.model(net, cont, vmaf)
            action = probs.argmax(dim=1).item()
        
        original = action
        
        if buffer < 5.0:
            action = min(action, 1)
        elif buffer < 10.0:
            action = min(action, 2)
        elif buffer < 20.0:
            action = min(action, 3)
        
        return action, original

# Load environment
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

mode = 'test' if len(loader.test_traces) >= 20 else 'val'

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode=mode
)

# تست تمام checkpoint ها
checkpoint_dir = 'results/fcc_training'
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')])

print(f"Found {len(checkpoints)} checkpoints")
print()

results = []

for ckpt_name in checkpoints:
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    
    try:
        # Load model
        model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        policy = SafetyWrapper(model)
        
        # Test (20 episodes برای سرعت)
        rewards = []
        rebuffers = []
        
        for ep in range(20):
            state = env.reset()
            if state is None:
                continue
            
            ep_reward = 0
            ep_rebuffer = 0
            done = False
            
            while not done:
                action, _ = policy.select_action(state, env.buffer)
                state, reward, done, info = env.step(action)
                ep_reward += reward
                ep_rebuffer += info['rebuffer_time']
            
            rewards.append(ep_reward)
            rebuffers.append(ep_rebuffer)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_rebuffer = np.mean(rebuffers)
        
        results.append({
            'checkpoint': ckpt_name,
            'update': checkpoint.get('update', 0),
            'reward_mean': mean_reward,
            'reward_std': std_reward,
            'rebuffer_mean': mean_rebuffer
        })
        
        print(f"{ckpt_name:25s}: Reward={mean_reward:+7.2f}±{std_reward:5.2f}, Rebuffer={mean_rebuffer:5.2f}s")
        
    except Exception as e:
        print(f"{ckpt_name:25s}: Error - {e}")

print()
print("=" * 80)
print("📊 Summary")
print("=" * 80)
print()

# مرتب کردن بر اساس reward
results_sorted = sorted(results, key=lambda x: x['reward_mean'], reverse=True)

print("Top 5 Checkpoints:")
for i, result in enumerate(results_sorted[:5]):
    marker = "🏆" if i == 0 else f"{i+1}. "
    print(f"{marker} {result['checkpoint']:20s}: "
          f"Reward={result['reward_mean']:+7.2f}, "
          f"Rebuffer={result['rebuffer_mean']:5.2f}s")

print()
print("=" * 80)

baseline = 102.16
best = results_sorted[0]
improvement = ((best['reward_mean'] - baseline) / baseline) * 100

print(f"Best Checkpoint: {best['checkpoint']}")
print(f"  Reward:      {best['reward_mean']:+.2f}")
print(f"  vs BBA:      {improvement:+.1f}%")
print()

if improvement > 10:
    print("🏆 Excellent! Use this checkpoint!")
elif improvement > 0:
    print("✅ Better than baseline!")
else:
    print("⚠️  All checkpoints below baseline...")
    print("    Suggestion: Try checkpoint from old training (checkpoint_100 from fcc_training/)")

print("=" * 80)
