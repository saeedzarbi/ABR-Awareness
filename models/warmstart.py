"""
Warmstart Model Initialization
Pre-train model with simple heuristic policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


def generate_heuristic_dataset(n_samples: int = 10000) -> Tuple[List, List]:
    """
    Generate dataset from buffer-based heuristic (BBA-like)
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        states: List of state dictionaries
        actions: List of actions
    """
    print("\n" + "="*80)
    print("🎯 WARMSTART: Generating Heuristic Dataset")
    print("="*80)
    
    states = []
    actions = []
    
    bitrate_levels = [300, 750, 1850, 2850, 4300, 6000]
    
    for i in range(n_samples):
        # Random buffer level and throughput
        buffer = np.random.uniform(0, 60)
        throughput = np.random.uniform(300, 6000)
        
        # BBA-like heuristic
        if buffer < 5:
            action = 0
        elif buffer < 12:
            action = 1
        elif buffer < 20:
            # Use throughput
            if throughput < 1000:
                action = 1
            else:
                action = 2
        elif buffer < 30:
            if throughput < 1500:
                action = 2
            elif throughput < 3000:
                action = 3
            else:
                action = 4
        else:
            if throughput < 2000:
                action = 2
            elif throughput < 3500:
                action = 3
            else:
                action = 4
        
        # Create synthetic state
        state = {
            'network': np.random.randn(6, 8).astype(np.float32) * 0.1,
            'content': np.random.randn(2).astype(np.float32) * 0.1,
            'vmaf': np.random.randn(6).astype(np.float32) * 0.1
        }
        
        # Set key features
        state['network'][2, -1] = buffer / 60.0  # Buffer
        state['network'][0, -1] = throughput / 6000.0  # Last throughput
        
        # Set past throughput
        for j in range(min(5, 8)):
            state['network'][0, -(j+1)] = np.clip(
                (throughput + np.random.randn() * 500) / 6000.0,
                0, 1
            )
        
        states.append(state)
        actions.append(action)
    
    # Print action distribution
    action_counts = np.bincount(actions, minlength=6)
    print(f"\nGenerated {n_samples} samples")
    print(f"Action distribution:")
    for i, count in enumerate(action_counts):
        pct = count / n_samples * 100
        print(f"  Action {i} ({bitrate_levels[i]:4d} kbps): {count:5d} ({pct:5.1f}%)")
    
    return states, actions


def warmstart_model(model, n_samples: int = 10000, n_epochs: int = 15, lr: float = 1e-3):
    """
    Pre-train model with heuristic policy
    
    Args:
        model: ContentAwareActor model
        n_samples: Number of training samples
        n_epochs: Training epochs
        lr: Learning rate
    
    Returns:
        model: Warmstarted model
    """
    print("\n" + "="*80)
    print("🔥 WARMSTART: Pre-training Model")
    print("="*80)
    
    device = next(model.parameters()).device
    
    # Generate dataset
    states, actions = generate_heuristic_dataset(n_samples)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    print(f"\nPreparing tensors...")
    network_states = torch.FloatTensor([s['network'] for s in states]).to(device)
    content_states = torch.FloatTensor([s['content'] for s in states]).to(device)
    vmaf_states = torch.FloatTensor([s['vmaf'] for s in states]).to(device)
    action_labels = torch.LongTensor(actions).to(device)
    
    dataset_size = len(states)
    batch_size = 128
    
    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {batch_size}")
    print(f"Training for {n_epochs} epochs\n")
    
    best_accuracy = 0
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        n_batches = 0
        
        # Shuffle
        indices = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:min(i+batch_size, dataset_size)]
            
            # Batch
            net = network_states[batch_idx]
            cont = content_states[batch_idx]
            vmaf = vmaf_states[batch_idx]
            labels = action_labels[batch_idx]
            
            # Forward
            action_probs, _ = model(net, cont, vmaf)
            loss = criterion(action_probs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            predictions = action_probs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            n_batches += 1
        
        # Epoch stats
        avg_loss = total_loss / n_batches
        accuracy = correct / dataset_size * 100
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    print(f"\n✅ Warmstart Complete")
    print(f"   Best accuracy: {best_accuracy:.2f}%")
    print(f"   Model is now initialized with buffer-based heuristic")
    print("="*80)
    
    return model


if __name__ == '__main__':
    from models.content_aware_model import ContentAwareActor
    
    print("Testing Warmstart")
    print("="*60)
    
    # Create model
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    )
    
    # Warmstart
    model = warmstart_model(model, n_samples=5000, n_epochs=10)
    
    print("\n✓ Warmstart tests passed!")