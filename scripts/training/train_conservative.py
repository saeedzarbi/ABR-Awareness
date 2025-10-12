"""
Train to be VERY conservative (prioritize low bitrate)
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.ppo_trainer import PPOTrainer


def main():
    print("\n" + "=" * 70)
    print("Conservative Training - Prioritize Low Bitrate")
    print("=" * 70)
    
    model = create_content_aware_model()
    env = ContentAwareEnvV2(use_real_traces=True)
    
    # MORE conservative settings
    trainer = PPOTrainer(
        model=model,
        env=env,
        lr=1e-4,  # Lower learning rate
        gamma=0.95,  # Less focus on future
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.1,  # High exploration
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64
    )
    
    episode_rewards = trainer.train(
        total_timesteps=100000,
        rollout_length=2048,
        log_interval=5
    )
    
    # Save
    import os
    os.makedirs('results/models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'episode_rewards': episode_rewards,
    }, 'results/models/content_aware_conservative.pth')
    
    print("✓ Model saved")


if __name__ == '__main__':
    main()
