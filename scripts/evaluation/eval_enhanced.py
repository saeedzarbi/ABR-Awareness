"""
Evaluate Enhanced Policy (Buffer-Aware + Smoothing)
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.policy_wrapper import BufferAwarePolicy, SmoothPolicy
import numpy as np


def evaluate_policy(policy, env, num_episodes=20, split='test'):
    """Evaluate policy on episodes"""
    
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    episode_details = []
    
    for ep in range(num_episodes):
        state = env.reset(video_id=(ep % 6) + 1, split=split)
        policy.reset()
        
        episode_reward = 0
        episode_rebuffer = 0
        bitrates = []
        actions = []
        
        done = False
        step = 0
        
        while not done:
            # Get current buffer
            buffer = env.buffer
            
            # Select action
            action = policy.select_action(
                state, 
                buffer,
                recent_rebuffer_time=0  # Will track internally
            )
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_rebuffer += info['rebuffer_time']
            bitrates.append(info['bitrate'])
            actions.append(action)
            
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(episode_rebuffer)
        episode_bitrates.append(np.mean(bitrates))
        
        episode_details.append({
            'reward': episode_reward,
            'rebuffer': episode_rebuffer,
            'bitrate': np.mean(bitrates),
            'actions': actions,
            'video_id': (ep % 6) + 1
        })
        
        print(f"  Episode {ep+1:2d}: Reward={episode_reward:+7.2f}, "
              f"Rebuffer={episode_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(bitrates):4.0f}kbps")
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_rebuffer': np.mean(episode_rebuffers),
        'avg_bitrate': np.mean(episode_bitrates),
        'details': episode_details
    }


def main():
    print("=" * 70)
    print("Evaluating Enhanced Policy")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = create_content_aware_model()
    checkpoint = torch.load('results/checkpoints/best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded best model (reward: {checkpoint['best_reward']:+.2f})")
    
    # Create environment
    env = ContentAwareEnvV2(use_real_traces=True)
    
    print("\n" + "=" * 70)
    print("1. Baseline Model (No Enhancements)")
    print("=" * 70)
    
    from models.policy_wrapper import BufferAwarePolicy
    
    # Just the model (no enhancements)
    class SimplePolicy:
        def __init__(self, model):
            self.model = model
            self.model.eval()
        
        def select_action(self, state, buffer=None, recent_rebuffer_time=0):
            network_state = torch.FloatTensor(state['network']).unsqueeze(0)
            content_features = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = self.model(network_state, content_features, vmaf_predictions)
                action = action_probs.argmax(dim=1).item()
            return action
        
        def reset(self):
            pass
    
    baseline_policy = SimplePolicy(model)
    baseline_results = evaluate_policy(baseline_policy, env, num_episodes=20)
    
    print(f"\n  Avg Reward:      {baseline_results['avg_reward']:+7.2f} ± {baseline_results['std_reward']:.2f}")
    print(f"  Avg Rebuffering:  {baseline_results['avg_rebuffer']:6.2f}s")
    print(f"  Avg Bitrate:      {baseline_results['avg_bitrate']:6.0f}kbps")
    
    print("\n" + "=" * 70)
    print("2. Buffer-Aware Policy")
    print("=" * 70)
    
    buffer_policy = BufferAwarePolicy(model)
    buffer_results = evaluate_policy(buffer_policy, env, num_episodes=20)
    
    print(f"\n  Avg Reward:      {buffer_results['avg_reward']:+7.2f} ± {buffer_results['std_reward']:.2f}")
    print(f"  Avg Rebuffering:  {buffer_results['avg_rebuffer']:6.2f}s")
    print(f"  Avg Bitrate:      {buffer_results['avg_bitrate']:6.0f}kbps")
    
    improvement_buffer = buffer_results['avg_reward'] - baseline_results['avg_reward']
    print(f"  Improvement: {improvement_buffer:+.2f}")
    
    print("\n" + "=" * 70)
    print("3. Buffer-Aware + Smoothing Policy")
    print("=" * 70)
    
    smooth_policy = SmoothPolicy(BufferAwarePolicy(model), max_jump=2)
    smooth_results = evaluate_policy(smooth_policy, env, num_episodes=20)
    
    print(f"\n  Avg Reward:      {smooth_results['avg_reward']:+7.2f} ± {smooth_results['std_reward']:.2f}")
    print(f"  Avg Rebuffering:  {smooth_results['avg_rebuffer']:6.2f}s")
    print(f"  Avg Bitrate:      {smooth_results['avg_bitrate']:6.0f}kbps")
    
    improvement_smooth = smooth_results['avg_reward'] - baseline_results['avg_reward']
    print(f"  Improvement: {improvement_smooth:+.2f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline:              {baseline_results['avg_reward']:+7.2f}")
    print(f"  + Buffer-Aware:        {buffer_results['avg_reward']:+7.2f} ({improvement_buffer:+.2f})")
    print(f"  + Smoothing:           {smooth_results['avg_reward']:+7.2f} ({improvement_smooth:+.2f})")
    print()
    print(f"  Buffer-Based Baseline: +102.16")
    
    if smooth_results['avg_reward'] > 102.16:
        print(f"  ✅ BEAT BASELINE! (+{smooth_results['avg_reward'] - 102.16:.2f})")
    else:
        gap = 102.16 - smooth_results['avg_reward']
        print(f"  Gap to baseline: {gap:.2f}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
