"""
Train Content-Aware ABR model using PPO
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.ppo_trainer import PPOTrainer
import numpy as np


def main():
    print("\n" + "=" * 70)
    print("Content-Aware ABR Training with PPO")
    print("=" * 70)
    
    # Create model
    print("\nCreating model...")
    model = create_content_aware_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created ({total_params:,} parameters)")
    
    # Create environment
    print("\nCreating environment...")
    env = ContentAwareEnvV2(use_real_traces=True)
    print("✓ Environment created")
    
    # Create trainer
    print("\nCreating PPO trainer...")
    trainer = PPOTrainer(
        model=model,
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64
    )
    print("✓ Trainer created")
    
    # Train
    print("\nStarting training...")
    print("=" * 70)
    
    episode_rewards = trainer.train(
        total_timesteps=50000,
        rollout_length=2048,
        log_interval=5
    )
    
    # Save model
    print("\nSaving model...")
    import os
    os.makedirs('results/models', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'episode_rewards': episode_rewards,
    }, 'results/models/content_aware_ppo.pth')
    
    print("✓ Model saved to results/models/content_aware_ppo.pth")
    
    # Plot learning curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, alpha=0.6)
        
        # Moving average
        window = 20
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (20)')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('PPO Training - Learning Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('results/ppo_learning_curve.png', dpi=150)
        print("✓ Learning curve saved to results/ppo_learning_curve.png")
    except Exception as e:
        print(f"Could not plot learning curve: {e}")
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("Training Statistics:")
    print("=" * 70)
    if len(episode_rewards) >= 20:
        final_avg = np.mean(episode_rewards[-20:])
        initial_avg = np.mean(episode_rewards[:20])
        print(f"Initial avg reward (first 20 episodes): {initial_avg:7.2f}")
        print(f"Final avg reward (last 20 episodes):    {final_avg:7.2f}")
        print(f"Improvement:                             {final_avg - initial_avg:+7.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
