"""
Imitation Learning - Train Policy to Mimic RobustMPC
Uses supervised learning (behavior cloning)
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# Dataset
# ============================================================================

class ExpertDataset(Dataset):
    """Dataset for expert demonstrations"""
    
    def __init__(self, demonstrations):
        self.states = []
        self.actions = []
        
        for demo in demonstrations:
            self.states.append(demo['state'])
            self.actions.append(demo['action'])
        
        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)
        
        print(f"Dataset: {len(self.states)} samples")
        print(f"State shape: {self.states.shape}")
        print(f"Action distribution: {np.bincount(self.actions)}")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ============================================================================
# Policy Network
# ============================================================================

class ImitationPolicy(nn.Module):
    """
    Policy network for imitation learning
    Same architecture as PPO will use
    """
    
    def __init__(self, state_dim=18, action_dim=6, hidden_size=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(self, state):
        """Returns action logits"""
        return self.network(state)
    
    def predict(self, state):
        """Returns action (for evaluation)"""
        with torch.no_grad():
            logits = self.forward(state)
            action = torch.argmax(logits, dim=-1)
        return action

# ============================================================================
# Training
# ============================================================================

def train_imitation_policy(
    demonstrations_file='expert_demonstrations.pkl',
    output_model='imitation_policy.pth',
    epochs=50,
    batch_size=256,
    learning_rate=1e-3,
    val_split=0.1,
    device='cuda'
):
    """
    Train imitation policy using behavior cloning
    """
    
    print("="*70)
    print("🎓 Training Imitation Policy")
    print("="*70)
    
    # Load demonstrations
    print(f"Loading: {demonstrations_file}")
    with open(demonstrations_file, 'rb') as f:
        data = pickle.load(f)
    
    demonstrations = data['demonstrations']
    metadata = data['metadata']
    
    print(f"Total demonstrations: {len(demonstrations)}")
    print(f"Expert performance:")
    print(f"  Avg VMAF: {metadata['avg_vmaf']:.2f}")
    print(f"  Avg Rebuffer: {metadata['avg_rebuffer']:.3f}s")
    print()
    
    # Create dataset
    dataset = ExpertDataset(demonstrations)
    
    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    print()
    
    # Create model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ImitationPolicy().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    
    print("Starting training...")
    print("-"*70)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for states, actions in pbar:
            states = states.to(device)
            actions = actions.to(device)
            
            # Forward
            logits = model(states)
            loss = criterion(logits, actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * states.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == actions).sum().item()
            train_total += states.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                
                logits = model(states)
                loss = criterion(logits, actions)
                
                val_loss += loss.item() * states.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == actions).sum().item()
                val_total += states.size(0)
        
        # Compute averages
        train_loss /= train_total
        train_acc = 100 * train_correct / train_total
        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        
        # Learning rate schedule
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best model! Val Acc: {val_acc:.2f}%")
        
        # Early stopping check
        if val_acc > 85.0:
            print(f"\n✓ Target accuracy reached! Stopping early.")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*70)
    print("📊 Final Evaluation")
    print("="*70)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    
    # Per-class accuracy
    model.eval()
    class_correct = [0] * 6
    class_total = [0] * 6
    
    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(device)
            actions = actions.to(device)
            
            logits = model(states)
            _, predicted = torch.max(logits, 1)
            
            for i in range(6):
                mask = (actions == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += ((predicted == actions) & mask).sum().item()
    
    print("\nPer-action accuracy:")
    bitrates = [300, 750, 1200, 1850, 2850, 6000]
    for i in range(6):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  Bitrate {i} ({bitrates[i]} kbps): {acc:.2f}% ({class_total[i]} samples)")
    
    # Save model
    print(f"\n✅ Saving model to: {output_model}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_accuracy': best_val_acc,
        'metadata': metadata
    }, output_model)
    
    print("="*70)
    print("\n✓ Imitation learning complete!")
    print(f"✓ Model ready for PPO fine-tuning")
    
    return model

# ============================================================================
# Convert to Stable-Baselines3 format
# ============================================================================

def convert_to_sb3_format(imitation_model_path, output_path):
    """
    Convert PyTorch imitation model to SB3 compatible format
    """
    print("\n" + "="*70)
    print("🔄 Converting to Stable-Baselines3 format")
    print("="*70)
    
    # Load imitation model
    checkpoint = torch.load(imitation_model_path)
    state_dict = checkpoint['model_state_dict']
    
    # Create SB3-compatible structure
    sb3_params = {
        'policy': state_dict,
        'val_accuracy': checkpoint['val_accuracy'],
        'source': 'imitation_learning'
    }
    
    torch.save(sb3_params, output_path)
    print(f"✅ Saved SB3 format to: {output_path}")
    print("="*70)

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train imitation policy')
    parser.add_argument('--data', type=str, default='expert_demonstrations.pkl')
    parser.add_argument('--output', type=str, default='imitation_policy.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Train
    model = train_imitation_policy(
        demonstrations_file=args.data,
        output_model=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Convert to SB3 format
    sb3_output = args.output.replace('.pth', '_sb3.pth')
    convert_to_sb3_format(args.output, sb3_output)
