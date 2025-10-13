"""
Final Evaluation: Smoothing Only (max_jump=1)
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
import numpy as np


class SimplePolicy:
    """Base policy - just the model"""
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def select_action(self, state):
        network_state = torch.FloatTensor(state['network']).unsqueeze(0)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = self.model(network_state, content_features, vmaf_predictions)
            action = action_probs.argmax(dim=1).item()
        return action


class SmoothPolicy:
    """Bitrate smoothing with max jump constraint"""
    def __init__(self, base_policy, max_jump=1):
        self.base_policy = base_policy
        self.max_jump = max_jump
        self.last_action = None
    
    def select_action(self, state):
        action = self.base_policy.select_action(state)
        
        # Apply smoothing
        if self.last_action is not None:
            if abs(action - self.last_action) > self.max_jump:
                if action > self.last_action:
                    action = self.last_action + self.max_jump
                else:
                    action = self.last_action - self.max_jump
        
        self.last_action = action
        return action
    
    def reset(self):
        self.last_action = None


def evaluate(policy, env, num_episodes=20):
    """Evaluate policy"""
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    
    for ep in range(num_episodes):
        state = env.reset(video_id=(ep % 6) + 1, split='test')
        policy.reset() if hasattr(policy, 'reset') else None
        
        episode_reward = 0
        episode_rebuffer = 0
        bitrates = []
        
        done = False
        while not done:
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_rebuffer += info['rebuffer_time']
            bitrates.append(info['bitrate'])
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(episode_rebuffer)
        episode_bitrates.append(np.mean(bitrates))
        
        print(f"  Episode {ep+1:2d}: Reward={episode_reward:+7.2f}, "
              f"Rebuffer={episode_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(bitrates):4.0f}kbps")
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_rebuffer': np.mean(episode_rebuffers),
        'avg_bitrate': np.mean(episode_bitrates)
    }


def main():
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    # Load model
    model = create_content_aware_model()
    checkpoint = torch.load('results/checkpoints/best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded best model (reward: {checkpoint['best_reward']:+.2f})")
    
    env = ContentAwareEnvV2(use_real_traces=True)
    
    # Test baseline
    print("\n" + "=" * 70)
    print("1. Baseline (No Enhancement)")
    print("=" * 70)
    baseline = SimplePolicy(model)
    baseline_results = evaluate(baseline, env)
    
    print(f"\n  Avg Reward:      {baseline_results['avg_reward']:+7.2f} ± {baseline_results['std_reward']:.2f}")
    print(f"  Avg Rebuffering:  {baseline_results['avg_rebuffer']:6.2f}s")
    print(f"  Avg Bitrate:      {baseline_results['avg_bitrate']:6.0f}kbps")
    
    # Test with smoothing (max_jump=1)
    print("\n" + "=" * 70)
    print("2. With Smoothing (max_jump=1)")
    print("=" * 70)
    smooth = SmoothPolicy(SimplePolicy(model), max_jump=1)
    smooth_results = evaluate(smooth, env)
    
    print(f"\n  Avg Reward:      {smooth_results['avg_reward']:+7.2f} ± {smooth_results['std_reward']:.2f}")
    print(f"  Avg Rebuffering:  {smooth_results['avg_rebuffer']:6.2f}s")
    print(f"  Avg Bitrate:      {smooth_results['avg_bitrate']:6.0f}kbps")
    
    improvement = smooth_results['avg_reward'] - baseline_results['avg_reward']
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Baseline Model:        {baseline_results['avg_reward']:+7.2f}")
    print(f"  + Smoothing:           {smooth_results['avg_reward']:+7.2f}")
    print(f"  Improvement:           {improvement:+.2f}")
    print()
    print(f"  Buffer-Based:          +102.16  (target)")
    
    gap = 102.16 - smooth_results['avg_reward']
    if gap <= 0:
        print(f"  ✅ BEAT BASELINE! (+{-gap:.2f})")
    else:
        print(f"  Gap:                   {gap:.2f}")
        pct = (smooth_results['avg_reward'] / 102.16) * 100
        print(f"  Performance:           {pct:.1f}% of baseline")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
