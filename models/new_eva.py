"""
Evaluate best model vs baselines
"""
import torch
import numpy as np
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("="*80)
print("EVALUATION: Best Model vs Baselines")
print("="*80)

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ContentAwareActor(
    state_dim=(6, 8),
    action_dim=6,
    content_dim=2
).to(device)

# Load checkpoint
checkpoint = torch.load('results/optimized_training_v2/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded best model from Update {checkpoint['update']}")
print(f"   Training reward: {checkpoint['reward']:+.2f}")
print(f"   Training rebuffer: {checkpoint['rebuffer']:.2f}s")

# Create test environment
fcc_loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

test_env = ContentAwareEnvFCC(
    fcc_trace_loader=fcc_loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'  # TEST SET
)

def evaluate_policy(policy_fn, policy_name, n_episodes=20):
    """Evaluate a policy"""
    print(f"\n{policy_name}:")
    print("-" * 80)
    
    all_rewards = []
    all_rebuffers = []
    all_vmafs = []
    all_bitrates = []
    
    for ep in range(n_episodes):
        state = test_env.reset()
        ep_reward = 0
        ep_rebuffer = 0
        ep_vmafs = []
        ep_bitrates = []
        done = False
        step = 0
        
        while not done:
            action = policy_fn(state, step, test_env)
            state, reward, done, info = test_env.step(action)
            
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_vmafs.append(info['vmaf'])
            ep_bitrates.append(info['bitrate'])
            step += 1
        
        all_rewards.append(ep_reward)
        all_rebuffers.append(ep_rebuffer)
        all_vmafs.append(np.mean(ep_vmafs))
        all_bitrates.append(np.mean(ep_bitrates))
    
    results = {
        'reward': np.mean(all_rewards),
        'reward_std': np.std(all_rewards),
        'rebuffer': np.mean(all_rebuffers),
        'rebuffer_std': np.std(all_rebuffers),
        'vmaf': np.mean(all_vmafs),
        'vmaf_std': np.std(all_vmafs),
        'bitrate': np.mean(all_bitrates),
        'bitrate_std': np.std(all_bitrates)
    }
    
    print(f"  Reward:    {results['reward']:+7.2f} ± {results['reward_std']:5.2f}")
    print(f"  Rebuffer:  {results['rebuffer']:7.2f}s ± {results['rebuffer_std']:5.2f}s")
    print(f"  VMAF:      {results['vmaf']:7.1f} ± {results['vmaf_std']:5.1f}")
    print(f"  Bitrate:   {results['bitrate']:7.0f} ± {results['bitrate_std']:5.0f} kbps")
    
    return results

# Baseline 1: Your DRL Model
def drl_policy(state, step, env):
    with torch.no_grad():
        net = torch.FloatTensor(state['network']).unsqueeze(0).to(device)
        cont = torch.FloatTensor(state['content']).unsqueeze(0).to(device)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(device)
        
        action_probs, _ = model(net, cont, vmaf)
        action = action_probs.argmax(dim=1).item()
    return action

# Baseline 2: BBA (Buffer-Based)
def bba_policy(state, step, env):
    buffer = env.buffer
    if buffer < 5:
        return 0
    elif buffer < 15:
        return 1
    elif buffer < 25:
        return 2
    elif buffer < 35:
        return 3
    else:
        return 4

# Baseline 3: Hybrid (Best simple strategy)
def hybrid_policy(state, step, env):
    buffer = env.buffer
    
    if len(env.past_throughput) == 0:
        return 1
    
    recent_tp = np.mean(env.past_throughput[-3:]) if len(env.past_throughput) >= 3 else env.past_throughput[-1]
    
    if buffer < 8:
        return min(1, int(recent_tp / 1000))
    
    if recent_tp < 600:
        return 0
    elif recent_tp < 1200:
        return 1
    elif recent_tp < 2200:
        return 2
    elif recent_tp < 3500:
        return 3
    else:
        return 4

# Run evaluations
print("\n" + "="*80)
print("TEST SET EVALUATION (20 episodes)")
print("="*80)

results_drl = evaluate_policy(drl_policy, "1. Your DRL Model (Content-Aware)")
results_bba = evaluate_policy(bba_policy, "2. BBA (Buffer-Based)")
results_hybrid = evaluate_policy(hybrid_policy, "3. Hybrid Baseline")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n{'Method':<30} {'Reward':<12} {'Rebuffer':<12} {'VMAF':<10} {'Bitrate':<10}")
print("-" * 80)
print(f"{'Your DRL Model':<30} {results_drl['reward']:+7.2f}     {results_drl['rebuffer']:7.2f}s    {results_drl['vmaf']:6.1f}    {results_drl['bitrate']:6.0f}")
print(f"{'BBA Baseline':<30} {results_bba['reward']:+7.2f}     {results_bba['rebuffer']:7.2f}s    {results_bba['vmaf']:6.1f}    {results_bba['bitrate']:6.0f}")
print(f"{'Hybrid Baseline':<30} {results_hybrid['reward']:+7.2f}     {results_hybrid['rebuffer']:7.2f}s    {results_hybrid['vmaf']:6.1f}    {results_hybrid['bitrate']:6.0f}")

# Improvements
print(f"\n{'Improvement vs BBA:':<30}")
reward_gain = results_drl['reward'] - results_bba['reward']
rebuffer_reduction = ((results_bba['rebuffer'] - results_drl['rebuffer']) / results_bba['rebuffer'] * 100) if results_bba['rebuffer'] > 0 else 0
print(f"  Reward:    {reward_gain:+.2f} ({reward_gain/abs(results_bba['reward'])*100:+.1f}%)")
print(f"  Rebuffer:  {rebuffer_reduction:+.1f}% reduction")

print("\n" + "="*80)