"""
Long-term Training با Real-time Monitoring
"""

import torch
import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger
import numpy as np


def evaluate_current_model(model, env, num_episodes=10):
    """Evaluate current model performance"""
    
    model.eval()
    
    episode_rewards = []
    episode_rebuffers = []
    episode_bitrates = []
    
    for ep in range(num_episodes):
        state = env.reset(video_id=(ep % 6) + 1, split='val')
        episode_reward = 0
        episode_rebuffer = 0
        bitrates = []
        
        done = False
        while not done:
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
    
    model.train()
    
    return {
        'avg_reward': float(np.mean(episode_rewards)),
        'avg_rebuffer': float(np.mean(episode_rebuffers)),
        'avg_bitrate': float(np.mean(episode_bitrates))
    }


class LongTermPPOTrainer(PPOTrainer):
    """Extended PPO Trainer with monitoring"""
    
    def __init__(self, model, env, logger, eval_env, **kwargs):
        super().__init__(model, env, **kwargs)
        
        # CRITICAL: Set external logger
        self.external_logger = logger
        
        self.eval_env = eval_env
        self.best_reward = -float('inf')
    
    def train_with_monitoring(
        self,
        total_timesteps=1_000_000,
        rollout_length=2048,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
        checkpoint_dir='results/checkpoints'
    ):
        """Train with monitoring"""
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("=" * 70)
        print("🚀 Long-term Training Started")
        print("=" * 70)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Rollout length: {rollout_length:,}")
        print(f"Evaluation every {eval_interval} updates")
        print(f"Checkpoints every {save_interval} updates")
        print("=" * 70)
        print()
        
        timesteps = 0
        update = 0
        
        while timesteps < total_timesteps:
            # Collect rollout
            buffer = self.collect_rollout(n_steps=rollout_length)
            timesteps += len(buffer)
            
            # Update policy
            stats = self.update_policy(buffer)
            update += 1
            
            # Log
            log_entry = self.external_logger.log_update(update, timesteps, stats)
            
            if update % log_interval == 0:
                self.external_logger.print_update(log_entry)
                
                # Time estimate
                elapsed = time.time() - self.external_logger.start_time
                steps_per_sec = timesteps / elapsed if elapsed > 0 else 0
                remaining_steps = total_timesteps - timesteps
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600
                
                print(f"         | Speed: {steps_per_sec:.1f} steps/s | "
                      f"ETA: {eta_hours:.1f}h")
                print()
            
            # Periodic evaluation
            if update % eval_interval == 0:
                print(f"\n{'='*70}")
                print(f"🔍 Evaluation at update {update} ({timesteps:,} steps)")
                print(f"{'='*70}")
                
                eval_results = evaluate_current_model(
                    self.model, 
                    self.eval_env, 
                    num_episodes=10
                )
                
                print(f"  Val Reward:      {eval_results['avg_reward']:+7.2f}")
                print(f"  Val Rebuffering:  {eval_results['avg_rebuffer']:6.2f}s")
                print(f"  Val Bitrate:      {eval_results['avg_bitrate']:6.0f}kbps")
                
                # Save best model
                if eval_results['avg_reward'] > self.best_reward:
                    self.best_reward = eval_results['avg_reward']
                    torch.save({
                        'update': update,
                        'timesteps': timesteps,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_reward': self.best_reward,
                        'eval_results': eval_results
                    }, f'{checkpoint_dir}/best_model.pth')
                    print(f"  ✓ New best model saved! (reward: {self.best_reward:+.2f})")
                
                print(f"{'='*70}\n")
            
            # Periodic checkpoint
            if update % save_interval == 0:
                checkpoint_path = f'{checkpoint_dir}/checkpoint_{update}.pth'
                torch.save({
                    'update': update,
                    'timesteps': timesteps,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode_rewards': self.episode_rewards,
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        print("\n" + "=" * 70)
        print("✓ Training Complete!")
        print("=" * 70)
        
        # Save final
        self.external_logger.save_summary()
        
        return self.episode_rewards


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🎯 Content-Aware ABR - Long-term Training")
    print("=" * 70)
    
    # Create model
    print("\nCreating model...")
    model = create_content_aware_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created ({total_params:,} parameters)")
    
    # Create environments
    print("\nCreating environments...")
    train_env = ContentAwareEnvV2(use_real_traces=True)
    eval_env = ContentAwareEnvV2(use_real_traces=True)
    print("✓ Environments created")
    
    # Create logger
    logger = TrainingLogger(run_name=args.run_name)
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = LongTermPPOTrainer(
        model=model,
        env=train_env,
        logger=logger,
        eval_env=eval_env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.05,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64
    )
    print("✓ Trainer created")
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Resumed from update {checkpoint.get('update', 0)}")
    
    # Train
    episode_rewards = trainer.train_with_monitoring(
        total_timesteps=args.timesteps,
        rollout_length=2048,
        log_interval=10,
        eval_interval=50,
        save_interval=100
    )
    
    # Final save
    print("\nSaving final model...")
    os.makedirs('results/models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'episode_rewards': episode_rewards,
        'final_timesteps': args.timesteps
    }, 'results/models/content_aware_longterm.pth')
    
    print("✓ Final model saved")
    
    print("\n" + "=" * 70)
    print("🎉 All Done!")
    print("=" * 70)
    print(f"\nLogs: {logger.log_file}")
    print(f"Best model: results/checkpoints/best_model.pth")


if __name__ == '__main__':
    main()
