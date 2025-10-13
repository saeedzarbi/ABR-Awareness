"""
Evaluate model generalization
Test vs Validation split
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
import numpy as np


def evaluate_split(model, env, split='test', num_episodes=50):
    """Evaluate on a specific split"""
    
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    
    print(f"\nTesting on {split} split ({num_episodes} episodes)...")
    
    for ep in range(num_episodes):
        video_id = (ep % 6) + 1
        state = env.reset(video_id=video_id, split=split)
        
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
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_rebuffer': np.mean(episode_rebuffers),
        'avg_bitrate': np.mean(episode_bitrates),
        'all_rewards': episode_rewards
    }


def main():
    print("=" * 70)
    print("Generalization Evaluation")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = create_content_aware_model()
    checkpoint = torch.load('results/checkpoints/best_model.pth', 
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded best model")
    
    # Create environment
    env = ContentAwareEnvV2(use_real_traces=True)
    
    print(f"\nDataset info:")
    print(f"  Train traces: {len(env.trace_loader.train_traces)}")
    print(f"  Val traces:   {len(env.trace_loader.val_traces)}")
    print(f"  Test traces:  {len(env.trace_loader.test_traces)}")
    
    # Evaluate on validation (seen during training)
    print("\n" + "=" * 70)
    print("1. Validation Set (seen during training)")
    print("=" * 70)
    val_results = evaluate_split(model, env, split='val', num_episodes=20)
    
    # Evaluate on test (completely unseen)
    print("\n" + "=" * 70)
    print("2. Test Set (completely unseen)")
    print("=" * 70)
    test_results = evaluate_split(model, env, split='test', num_episodes=20)
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERALIZATION RESULTS")
    print("=" * 70)
    print(f"  Validation (seen):       {val_results['avg_reward']:+7.2f} ± {val_results['std_reward']:.2f}")
    print(f"  Test (unseen):           {test_results['avg_reward']:+7.2f} ± {test_results['std_reward']:.2f}")
    print()
    
    gap = val_results['avg_reward'] - test_results['avg_reward']
    if val_results['avg_reward'] != 0:
        gap_pct = (gap / abs(val_results['avg_reward'])) * 100
    else:
        gap_pct = 0
    
    print(f"  Generalization gap:      {gap:+.2f} ({gap_pct:.1f}%)")
    print()
    print(f"  Val rebuffering:         {val_results['avg_rebuffer']:.2f}s")
    print(f"  Test rebuffering:        {test_results['avg_rebuffer']:.2f}s")
    print()
    print(f"  Val bitrate:             {val_results['avg_bitrate']:.0f}kbps")
    print(f"  Test bitrate:            {test_results['avg_bitrate']:.0f}kbps")
    print("=" * 70)
    
    # Interpretation
    print("\nInterpretation:")
    if abs(gap_pct) < 10:
        print("  ✅ Excellent generalization (gap < 10%)")
    elif abs(gap_pct) < 20:
        print("  ✅ Good generalization (gap < 20%)")
    elif abs(gap_pct) < 30:
        print("  ⚠️  Moderate generalization (gap < 30%)")
    else:
        print("  ❌ Poor generalization (gap > 30%)")
    
    print("\nFor paper:")
    print('  "Our model achieves {:.1f} reward on validation and'.format(val_results['avg_reward']))
    print('   {:.1f} on completely unseen test traces, demonstrating'.format(test_results['avg_reward']))
    print('   {:.1f}% generalization gap."'.format(abs(gap_pct)))


if __name__ == '__main__':
    main()