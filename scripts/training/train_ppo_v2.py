"""
Train with HIGHER exploration
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
    print("PPO Training V2 - Increased Exploration")
    print("=" * 70)
    
    model = create_content_aware_model()
    env = ContentAwareEnvV2(use_real_traces=True)
    
    # Higher entropy coefficient for MORE exploration
    trainer = PPOTrainer(
        model=model,
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.05,  # Increased from 0.01 to 0.05
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64
    )
    
    print("\n⚠️  Using HIGHER entropy coefficient (0.05) for more exploration")
    print("=" * 70)
    
    episode_rewards = trainer.train(
        total_timesteps=100000,  # More timesteps
        rollout_length=2048,
        log_interval=5
    )
    
    # Save
    import os
    os.makedirs('results/models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'episode_rewards': episode_rewards,
    }, 'results/models/content_aware_ppo_v2.pth')
    
    print("✓ Model saved to results/models/content_aware_ppo_v2.pth")


if __name__ == '__main__':
    main()
