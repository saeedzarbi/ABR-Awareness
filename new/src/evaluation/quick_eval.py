"""
Quick evaluation of trained PPO model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env import ABREnv
from configs.paths import get_paths
import numpy as np

PATHS = get_paths()


def evaluate_model(model_path: str, num_episodes: int = 5):
    """
    Evaluate trained model.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to evaluate
    """
    print("\n🎯 Evaluating Trained Model\n")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}\n")
    
    # Load model
    model = PPO.load(model_path)
    print("✓ Model loaded\n")
    
    # Create environment
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48,
        random_seed=42
    )
    
    # Evaluate
    episode_rewards = []
    episode_rebuffers = []
    episode_qualities = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(info['total_rebuffer'])
        episode_qualities.append(info['avg_quality'])
        
        print(f"Episode {ep+1}: "
              f"Reward={episode_reward:.2f}, "
              f"Rebuffer={info['total_rebuffer']:.2f}s, "
              f"Avg Quality={info['avg_quality']:.2f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print(f"  Avg Reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg Rebuffer:   {np.mean(episode_rebuffers):.2f}s ± {np.std(episode_rebuffers):.2f}s")
    print(f"  Avg Quality:    {np.mean(episode_qualities):.2f} ± {np.std(episode_qualities):.2f}")
    print(f"{'='*60}\n")


def compare_with_random():
    """Compare trained model with random policy."""
    print("\n📊 Comparing with Random Policy\n")
    
    env = ABREnv(
        video_name='sample1',
        trace_dir=str(PATHS['processed_traces']),
        vmaf_dir=str(PATHS['vmaf_scores']),
        siti_dir=str(PATHS['content_features']),
        max_chunks=48,
        random_seed=999
    )
    
    # Random policy
    random_rewards = []
    random_rebuffers = []
    
    for _ in range(5):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        random_rewards.append(episode_reward)
        random_rebuffers.append(info['total_rebuffer'])
    
    print("Random Policy:")
    print(f"  Avg Reward:   {np.mean(random_rewards):.2f}")
    print(f"  Avg Rebuffer: {np.mean(random_rebuffers):.2f}s\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='results/models/ppo_abr/best_model/best_model.zip',
        help='Path to model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with random policy'
    )
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.episodes)
    
    if args.compare:
        compare_with_random()