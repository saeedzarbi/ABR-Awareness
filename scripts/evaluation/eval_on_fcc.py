"""
Evaluate model on FCC traces
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.trace_loader import TraceLoader
import numpy as np


def evaluate_on_fcc(num_episodes=50):
    print("=" * 70)
    print("Evaluating on FCC Traces")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = create_content_aware_model()
    checkpoint = torch.load('results/checkpoints/best_model.pth', 
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded best model (reward: {checkpoint['best_reward']:+.2f})")
    
    # Create FCC trace loader
    print("\nLoading FCC traces...")
    fcc_loader = TraceLoader(
        train_dir='data/fcc_traces/cooked',
        val_dir='data/fcc_traces/cooked',
        test_dir='data/fcc_traces/cooked'
    )
    print(f"✓ Loaded {len(fcc_loader.test_traces)} FCC traces")
    
    # Create environment with FCC traces
    env = ContentAwareEnvV2(use_real_traces=True)
    env.trace_loader = fcc_loader
    
    # Evaluate
    print(f"\nTesting on {num_episodes} episodes...")
    
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    
    for ep in range(num_episodes):
        video_id = (ep % 6) + 1
        state = env.reset(video_id=video_id, split='test')
        
        episode_reward = 0
        episode_rebuffer = 0
        bitrates = []
        
        done = False
        while not done:
            # Select action
            network_state = torch.FloatTensor(state['network']).unsqueeze(0)
            content_features = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = model(network_state, content_features, vmaf_predictions)
                action = action_probs.argmax(dim=1).item()
            
            # Step
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_rebuffer += info['rebuffer_time']
            bitrates.append(info['bitrate'])
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(episode_rebuffer)
        episode_bitrates.append(np.mean(bitrates))
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Reward={episode_reward:+7.2f}, "
                  f"Rebuffer={episode_rebuffer:5.2f}s")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Pensieve traces:  +98.95  (original)")
    print(f"  FCC traces:      {np.mean(episode_rewards):+7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Rebuffering:      {np.mean(episode_rebuffers):6.2f}s")
    print(f"  Bitrate:          {np.mean(episode_bitrates):6.0f}kbps")
    print()
    
    gap = 98.95 - np.mean(episode_rewards)
    pct = (np.mean(episode_rewards) / 98.95) * 100
    
    print(f"  Generalization gap: {gap:.2f} ({100-pct:.1f}% drop)")
    print(f"  Performance:        {pct:.1f}% of original")
    print("=" * 70)
    
    # Interpretation
    print("\nInterpretation:")
    if pct >= 90:
        print("  ✅ Excellent generalization!")
    elif pct >= 80:
        print("  ✅ Good generalization")
    elif pct >= 70:
        print("  ⚠️  Moderate generalization")
    else:
        print("  ❌ Poor generalization - needs improvement")


if __name__ == '__main__':
    evaluate_on_fcc()