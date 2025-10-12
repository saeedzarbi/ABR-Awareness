"""
Evaluate trained PPO model
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
import numpy as np


def evaluate_model(model, env, num_episodes=10, split='test'):
    """Evaluate trained model"""
    
    model.eval()
    
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    
    for ep in range(num_episodes):
        state = env.reset(video_id=(ep % 6) + 1, split=split)
        episode_reward = 0
        episode_rebuffer = 0
        bitrates = []
        
        done = False
        while not done:
            # Get action from model
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
        
        print(f"  Episode {ep+1:2d}: Reward={episode_reward:7.2f}, "
              f"Rebuffer={episode_rebuffer:5.2f}s, "
              f"Avg Bitrate={np.mean(bitrates):6.0f}kbps")
    
    print("\n" + "=" * 70)
    print(f"Results on {split.upper()} set:")
    print("=" * 70)
    print(f"  Avg Reward:      {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg Rebuffering:  {np.mean(episode_rebuffers):6.2f}s")
    print(f"  Avg Bitrate:      {np.mean(episode_bitrates):6.0f}kbps")
    
    return {
        'rewards': episode_rewards,
        'rebuffers': episode_rebuffers,
        'bitrates': episode_bitrates
    }


if __name__ == '__main__':
    print("=" * 70)
    print("Evaluating Trained PPO Model")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = create_content_aware_model()
    checkpoint = torch.load('results/models/content_aware_ppo.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")
    
    # Create environment
    print("\nCreating environment...")
    env = ContentAwareEnvV2(use_real_traces=True)
    print("✓ Environment created")
    
    # Evaluate on test set
    print("\nEvaluating on TEST set:")
    print("-" * 70)
    test_results = evaluate_model(model, env, num_episodes=20, split='test')
    
    print("\n" + "=" * 70)
    print("✓ Evaluation complete!")
    print("=" * 70)
