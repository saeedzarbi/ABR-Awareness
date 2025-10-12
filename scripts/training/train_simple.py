"""
Simple Training Script for Content-Aware ABR
Single-agent training (no A3C yet) for testing
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env import ContentAwareEnv


class SimpleTrainer:
    """Simple trainer for testing"""
    
    def __init__(
        self,
        model,
        env,
        learning_rate=1e-4,
        gamma=0.99,
        entropy_coef=0.01
    ):
        self.model = model
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def compute_returns(self, rewards, gamma):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    def train_episode(self, video_id=1):
        """Train on one episode"""
        
        # Reset environment
        state = self.env.reset(video_id=video_id)
        
        # Storage for trajectory
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        episode_reward = 0
        episode_length = 0
        
        done = False
        
        while not done:
            # Convert state to tensors
            network_state = torch.FloatTensor(state['network']).unsqueeze(0)  # (1, 6, 8)
            content_features = torch.FloatTensor(state['content']).unsqueeze(0)  # (1, 2)
            vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)  # (1, 6)
            
            # Forward pass
            action_probs, value = self.model(
                network_state,
                content_features,
                vmaf_predictions
            )
            
            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            # Store log prob and entropy
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value)
            
            # Take step
            next_state, reward, done, info = self.env.step(action.item())
            
            rewards.append(reward)
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        # Compute returns
        returns = self.compute_returns(rewards, self.gamma)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute losses
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()
        entropies = torch.stack(entropies)
        
        # Advantage
        advantages = returns - values.detach()
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss (value function)
        critic_loss = advantages.pow(2).mean()
        
        # Entropy bonus (for exploration)
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        
        self.optimizer.step()
        
        # Statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropies.mean().item()
        }
    
    def train(self, num_episodes=100, video_ids=[1, 2, 3]):
        """Train for multiple episodes"""
        
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Videos: {video_ids}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print()
        
        for episode in range(num_episodes):
            # Randomly select video
            video_id = np.random.choice(video_ids)
            
            # Train one episode
            stats = self.train_episode(video_id=video_id)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                
                print(f"Episode {episode+1:4d} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Length: {avg_length:4.1f} | "
                      f"Actor: {stats['actor_loss']:6.3f} | "
                      f"Critic: {stats['critic_loss']:6.3f} | "
                      f"Entropy: {stats['entropy']:5.3f}")
        
        print()
        print("=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Average reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")
        print(f"Average length (last 10): {np.mean(self.episode_lengths[-10:]):.1f}")
        
        return self.episode_rewards, self.episode_lengths


def main():
    """Main training function"""
    
    print("\n" + "=" * 70)
    print("Content-Aware ABR - Simple Training")
    print("=" * 70)
    print()
    
    # Create model
    print("Creating model...")
    model = create_content_aware_model()
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create environment
    print("\nCreating environment...")
    env = ContentAwareEnv()
    print("✓ Environment created")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = SimpleTrainer(
        model=model,
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        entropy_coef=0.01
    )
    print("✓ Trainer created")
    print()
    
    # Train
    episode_rewards, episode_lengths = trainer.train(
        num_episodes=100,
        video_ids=[1, 2, 3]
    )
    
    # Save model
    save_path = 'results/models/content_aware_simple.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }, save_path)
    print(f"\n✓ Model saved to {save_path}")
    
    # Plot results (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Rewards
        ax1.plot(episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Moving average
        window = 10
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg')
            ax1.legend()
        
        # Lengths
        ax2.plot(episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = 'results/plots/training_simple.png'
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Plot saved to {plot_path}")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
