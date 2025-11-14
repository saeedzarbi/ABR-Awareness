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


def find_best_model():
    """Find the best available model."""
    model_dir = PATHS['models'] / 'ppo_abr'
    
    # Priority order
    candidates = [
        model_dir / 'best_model' / 'best_model',
        model_dir / 'final_model',
    ]
    
    # Add checkpoints
    checkpoint_dir = model_dir / 'checkpoints'
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob('ppo_abr_*_steps.zip'))
        if checkpoints:
            # Get latest checkpoint (remove .zip)
            candidates.append(checkpoints[-1].with_suffix(''))
    
    # Find first existing model
    for candidate in candidates:
        # Try with and without .zip
        if candidate.with_suffix('.zip').exists():
            return str(candidate)
        elif candidate.exists():
            return str(candidate)
    
    raise FileNotFoundError(
        f"No model found in {model_dir}\n"
        f"Train a model first: python src/training/train_ppo.py --quick-test"
    )


def evaluate_model(model_path: str, num_episodes: int = 5):
    """
    Evaluate trained model.
    
    Args:
        model_path: Path to saved model (without .zip)
        num_episodes: Number of episodes to evaluate
    """
    print("\n🎯 Evaluating Trained Model\n")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}\n")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nAvailable models:")
        model_dir = PATHS['models'] / 'ppo_abr'
        for f in model_dir.rglob('*.zip'):
            print(f"  - {f}")
        return
    
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
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_rebuffers.append(info['total_rebuffer'])
        episode_qualities.append(info['avg_quality'])
        
        print(f"Episode {ep+1}/{num_episodes}: "
              f"Reward={episode_reward:7.2f}, "
              f"Rebuffer={info['total_rebuffer']:5.2f}s, "
              f"Quality={info['avg_quality']:.3f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print(f"  Avg Reward:     {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg Rebuffer:   {np.mean(episode_rebuffers):5.2f}s ± {np.std(episode_rebuffers):.2f}s")
    print(f"  Avg Quality:    {np.mean(episode_qualities):.3f} ± {np.std(episode_qualities):.3f}")
    print(f"{'='*60}\n")
    
    return {
        'rewards': episode_rewards,
        'rebuffers': episode_rebuffers,
        'qualities': episode_qualities
    }


def compare_with_random(num_episodes: int = 5):
    """Compare with random policy."""
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
    random_qualities = []
    
    for ep in range(num_episodes):
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
        random_qualities.append(info['avg_quality'])
    
    print("Random Policy Results:")
    print(f"  Avg Reward:   {np.mean(random_rewards):7.2f} ± {np.std(random_rewards):.2f}")
    print(f"  Avg Rebuffer: {np.mean(random_rebuffers):5.2f}s ± {np.std(random_rebuffers):.2f}s")
    print(f"  Avg Quality:  {np.mean(random_qualities):.3f} ± {np.std(random_qualities):.3f}\n")
    
    return {
        'rewards': random_rewards,
        'rebuffers': random_rebuffers,
        'qualities': random_qualities
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model (auto-detect if not provided)'
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
    
    # Find model
    if args.model is None:
        try:
            args.model = find_best_model()
            print(f"✓ Auto-detected model: {args.model}\n")
        except FileNotFoundError as e:
            print(f"✗ {e}")
            exit(1)
    
    # Evaluate
    ppo_results = evaluate_model(args.model, args.episodes)
    
    if args.compare:
        random_results = compare_with_random(args.episodes)
        
        # Improvement
        reward_improvement = (
            np.mean(ppo_results['rewards']) - np.mean(random_results['rewards'])
        )
        rebuffer_reduction = (
            np.mean(random_results['rebuffers']) - np.mean(ppo_results['rebuffers'])
        )
        
        print(f"{'='*60}")
        print("PPO vs Random:")
        print(f"  Reward improvement:  {reward_improvement:+.2f}")
        print(f"  Rebuffer reduction:  {rebuffer_reduction:+.2f}s")
        print(f"{'='*60}\n")