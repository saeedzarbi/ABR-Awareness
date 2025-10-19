"""
Evaluate the best saved model
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
import numpy as np


def main():
    print("=" * 70)
    print("Evaluating BEST MODEL")
    print("=" * 70)
    
    # Load best model
    model = create_content_aware_model()
    checkpoint = torch.load('results/fcc_training/checkpoint_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✓ Loaded best model:")
    print(f"  Update: {checkpoint['update']}")
    print(f"  Timesteps: {checkpoint['timesteps']:,}")
    print(f"  Validation Reward: {checkpoint['best_reward']:+.2f}")
    print()
    
    # Test on test set
    env = ContentAwareEnvV2(use_real_traces=True)
    
    print("Testing on TEST set (20 episodes):")
    print("-" * 70)
    
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    
    for ep in range(20):
        state = env.reset(video_id=(ep % 6) + 1, split='test')
        episode_reward = 0
        episode_rebuffer = 0
        bitrates = []
        
        done = False
        while not done:
            network_state = torch.FloatTensor(state['network']).unsqueeze(0)
            content_features = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = model(network_state, content_features, vmaf_predictions)
                action = action_probs.argmax(dim=1).item()
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_rebuffer += info['rebuffer_time']
            bitrates.append(info['bitrate'])
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(episode_rebuffer)
        episode_bitrates.append(np.mean(bitrates))
        
        print(f"  Episode {ep+1:2d}: Reward={episode_reward:+7.2f}, "
              f"Rebuffer={episode_rebuffer:5.2f}s, "
              f"Bitrate={np.mean(bitrates):4.0f}kbps")
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"  Avg Reward:      {np.mean(episode_rewards):+7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg Rebuffering:  {np.mean(episode_rebuffers):6.2f}s")
    print(f"  Avg Bitrate:      {np.mean(episode_bitrates):6.0f}kbps")
    print("=" * 70)
    
    # Compare با baselines
    print("\nCOMPARISON:")
    print("  Buffer-Based:     +102.16  (baseline)")
    print(f"  Our Best Model:   {np.mean(episode_rewards):+7.2f}")
    
    if np.mean(episode_rewards) > 102:
        print("\n  ✅ BEATS BASELINE!")
    else:
        print(f"\n  Close to baseline (difference: {np.mean(episode_rewards) - 102.16:+.2f})")


if __name__ == '__main__':
    main()
