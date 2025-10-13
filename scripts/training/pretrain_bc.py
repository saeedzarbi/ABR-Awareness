"""
Behavioral Cloning Pre-training
Train model to imitate buffer-based policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
import numpy as np


class BufferBasedExpert:
    """Buffer-based heuristic as expert"""
    
    def __init__(self, bitrate_levels=[300, 750, 1850, 2850, 4300, 6000]):
        self.bitrate_levels = bitrate_levels
    
    def select_action(self, buffer):
        """Select action based on buffer level"""
        if buffer < 5:
            return 0  # 300 kbps
        elif buffer < 10:
            return 1  # 750 kbps
        elif buffer < 20:
            return 2  # 1850 kbps
        elif buffer < 30:
            return 3  # 2850 kbps
        elif buffer < 40:
            return 4  # 4300 kbps
        else:
            return 5  # 6000 kbps


def collect_expert_data(env, expert, num_episodes=500):
    """Collect expert demonstrations"""
    
    print(f"Collecting {num_episodes} expert demonstrations...")
    
    dataset = {
        'states_network': [],
        'states_content': [],
        'states_vmaf': [],
        'actions': []
    }
    
    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"  Episode {ep}/{num_episodes}")
        
        video_id = (ep % 6) + 1
        state = env.reset(video_id=video_id, split='train')
        
        done = False
        while not done:
            # Expert action
            action = expert.select_action(env.buffer)
            
            # Store transition
            dataset['states_network'].append(state['network'])
            dataset['states_content'].append(state['content'])
            dataset['states_vmaf'].append(state['vmaf'])
            dataset['actions'].append(action)
            
            # Step
            state, _, done, _ = env.step(action)
    
    # Convert to numpy
    for key in dataset:
        dataset[key] = np.array(dataset[key])
    
    print(f"✓ Collected {len(dataset['actions'])} samples")
    return dataset


def pretrain_model(model, dataset, epochs=10, batch_size=64):
    """Pre-train model via behavioral cloning"""
    
    print(f"\nPre-training model for {epochs} epochs...")
    
    # Prepare data
    states_network = torch.FloatTensor(dataset['states_network'])
    states_content = torch.FloatTensor(dataset['states_content'])
    states_vmaf = torch.FloatTensor(dataset['states_vmaf'])
    actions = torch.LongTensor(dataset['actions'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_samples = len(actions)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Shuffle
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            # Batch data
            batch_network = states_network[batch_idx]
            batch_content = states_content[batch_idx]
            batch_vmaf = states_vmaf[batch_idx]
            batch_actions = actions[batch_idx]
            
            # Forward
            action_probs, _ = model(batch_network, batch_content, batch_vmaf)
            
            # Loss
            loss = criterion(action_probs, batch_actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            pred = action_probs.argmax(dim=1)
            correct += (pred == batch_actions).sum().item()
        
        accuracy = correct / num_samples * 100
        avg_loss = total_loss / (num_samples / batch_size)
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
    
    print("✓ Pre-training complete!")


def main():
    print("=" * 70)
    print("Behavioral Cloning Pre-training")
    print("=" * 70)
    
    # Create environment
    env = ContentAwareEnvV2(use_real_traces=True)
    
    # Create expert
    expert = BufferBasedExpert()
    
    # Collect data
    dataset = collect_expert_data(env, expert, num_episodes=500)
    
    # Create model
    model = create_content_aware_model()
    
    # Pre-train
    pretrain_model(model, dataset, epochs=10)
    
    # Save
    save_path = 'results/models/pretrained_bc.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'method': 'behavioral_cloning'
    }, save_path)
    
    print(f"\n✓ Saved pre-trained model to: {save_path}")
    print("\nNext step:")
    print("  python scripts/training/train_longterm.py \\")
    print("      --pretrained results/models/pretrained_bc.pth \\")
    print("      --run-name bc_finetuned")
    print("=" * 70)


if __name__ == '__main__':
    main()
