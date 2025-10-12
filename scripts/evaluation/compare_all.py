"""
Compare Content-Aware model with baselines
FIXED: JSON serialization and better evaluation
"""

import torch
import numpy as np
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env import ContentAwareEnv
from scripts.evaluation.baselines import get_baseline_policy


# JSON Encoder for numpy types (FIX)
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_policy(policy, env, num_episodes=10, video_ids=[1,2,3,4,5,6], policy_name="Policy"):
    """
    Evaluate any policy
    
    Args:
        policy: policy object with select_action(state) method
        env: environment
        num_episodes: number of episodes
        video_ids: list of video IDs to test
        policy_name: name for printing
    
    Returns:
        dict with statistics
    """
    
    episode_rewards = []
    all_stats = {
        'bitrates': [],
        'rebuffer_times': [],
        'vmaf_scores': [],
        'buffer_levels': [],
        'smoothness': []
    }
    
    print(f"\nEvaluating {policy_name}...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        video_id = video_ids[episode % len(video_ids)]
        state = env.reset(video_id=video_id)
        
        episode_reward = 0
        prev_bitrate = None
        
        done = False
        while not done:
            # Select action
            if hasattr(policy, 'model'):  # Neural network policy
                network_state = torch.FloatTensor(state['network']).unsqueeze(0)
                content_features = torch.FloatTensor(state['content']).unsqueeze(0)
                vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs, _ = policy.model(network_state, content_features, vmaf_predictions)
                    action = action_probs.argmax(dim=1).item()
            else:  # Baseline policy
                action = policy.select_action(state)
            
            # Step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Collect stats (convert to native Python types)
            all_stats['bitrates'].append(float(info['bitrate']))
            all_stats['rebuffer_times'].append(float(info['rebuffer_time']))
            all_stats['vmaf_scores'].append(float(state['vmaf'][action]))
            all_stats['buffer_levels'].append(float(info['buffer']))
            
            # Smoothness (bitrate changes)
            if prev_bitrate is not None:
                all_stats['smoothness'].append(float(abs(info['bitrate'] - prev_bitrate)))
            prev_bitrate = info['bitrate']
            
            state = next_state
        
        episode_rewards.append(float(episode_reward))
    
    # Compute statistics (all as native Python types)
    stats = {
        'policy_name': str(policy_name),
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_bitrate': float(np.mean(all_stats['bitrates'])),
        'avg_vmaf': float(np.mean(all_stats['vmaf_scores'])),
        'total_rebuffer': float(np.sum(all_stats['rebuffer_times'])),
        'rebuffer_ratio': float(np.sum(all_stats['rebuffer_times']) / (num_episodes * 48 * 4) * 100),
        'avg_smoothness': float(np.mean(all_stats['smoothness'])) if all_stats['smoothness'] else 0.0,
        'num_switches': int(len([s for s in all_stats['smoothness'] if s > 0])) if all_stats['smoothness'] else 0,
    }
    
    # Print summary
    print(f"  Avg Reward:        {stats['avg_reward']:7.2f} ± {stats['std_reward']:.2f}")
    print(f"  Avg Bitrate:       {stats['avg_bitrate']:7.0f} kbps")
    print(f"  Avg VMAF:          {stats['avg_vmaf']:7.2f}")
    print(f"  Total Rebuffering: {stats['total_rebuffer']:7.2f}s ({stats['rebuffer_ratio']:.2f}%)")
    print(f"  Avg Smoothness:    {stats['avg_smoothness']:7.0f} kbps")
    print(f"  Bitrate Switches:  {stats['num_switches']}")
    
    return stats


class ContentAwarePolicy:
    """Wrapper for content-aware model"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()


def main():
    print("\n" + "=" * 70)
    print("ABR Algorithm Comparison (Improved)")
    print("=" * 70)
    
    # Create environment
    print("\nCreating environment...")
    env = ContentAwareEnv()
    print("✓ Environment created")
    
    # Test parameters
    num_episodes = 12  # 2 episodes per video
    video_ids = [1, 2, 3, 4, 5, 6]
    
    print(f"\nTest configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Videos: {video_ids}")
    
    # Policies to compare
    all_results = []
    
    # 1. Content-Aware (our model)
    print("\n" + "=" * 70)
    try:
        model = create_content_aware_model()
        
        # Try to load improved model first, fall back to simple
        try:
            checkpoint = torch.load('results/models/content_aware_improved.pth', 
                                   weights_only=False)
            model_name = "Content-Aware Improved (Ours)"
        except:
            checkpoint = torch.load('results/models/content_aware_simple.pth', 
                                   weights_only=False)
            model_name = "Content-Aware (Ours)"
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        policy = ContentAwarePolicy(model)
        stats = evaluate_policy(policy, env, num_episodes, video_ids, model_name)
        all_results.append(stats)
    except Exception as e:
        print(f"✗ Could not load Content-Aware model: {e}")
    
    # 2. Fixed Low
    policy = get_baseline_policy('fixed_low')
    stats = evaluate_policy(policy, env, num_episodes, video_ids, "Fixed Low (300 kbps)")
    all_results.append(stats)
    
    # 3. Fixed Medium
    policy = get_baseline_policy('fixed_mid')
    stats = evaluate_policy(policy, env, num_episodes, video_ids, "Fixed Medium (1850 kbps)")
    all_results.append(stats)
    
    # 4. Fixed High
    policy = get_baseline_policy('fixed_high')
    stats = evaluate_policy(policy, env, num_episodes, video_ids, "Fixed High (6000 kbps)")
    all_results.append(stats)
    
    # 5. Buffer-Based
    policy = get_baseline_policy('buffer_based')
    stats = evaluate_policy(policy, env, num_episodes, video_ids, "Buffer-Based (BBA-like)")
    all_results.append(stats)
    
    # 6. Throughput-Based
    policy = get_baseline_policy('throughput_based')
    stats = evaluate_policy(policy, env, num_episodes, video_ids, "Throughput-Based")
    all_results.append(stats)
    
    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Algorithm':<30} {'Reward':>10} {'VMAF':>8} {'Bitrate':>10} {'Rebuffer':>10} {'Switches':>10}")
    print("-" * 90)
    
    for result in all_results:
        print(f"{result['policy_name']:<30} "
              f"{result['avg_reward']:>10.2f} "
              f"{result['avg_vmaf']:>8.2f} "
              f"{result['avg_bitrate']:>10.0f} "
              f"{result['rebuffer_ratio']:>9.2f}% "
              f"{result['num_switches']:>10}")
    
    print()
    
    # Improvement analysis
    if len(all_results) > 1:
        content_aware_reward = all_results[0]['avg_reward']
        
        print("Improvement over baselines:")
        for i, result in enumerate(all_results[1:], 1):
            if result['avg_reward'] != 0:
                improvement = ((content_aware_reward - result['avg_reward']) / abs(result['avg_reward']) * 100)
            else:
                improvement = 0.0
            print(f"  vs {result['policy_name']:<28}: {improvement:+6.1f}%")
    
    print("\n✓ Comparison complete!")
    
    # Save results (with custom encoder)
    import os
    os.makedirs('results', exist_ok=True)
    output_file = 'results/comparison_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
