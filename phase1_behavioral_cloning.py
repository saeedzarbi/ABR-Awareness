"""
Phase 1: Behavioral Cloning from Hybrid Expert
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader


def hybrid_expert(env):
    """
    Hybrid expert policy (best simple baseline)
    """
    buffer = env.buffer
    
    if len(env.past_throughput) == 0:
        return 1
    
    recent_tp = np.mean(env.past_throughput[-3:]) if len(env.past_throughput) >= 3 else env.past_throughput[-1]
    
    # Buffer-based conservativeness
    if buffer < 8:
        return min(1, int(recent_tp / 1000))
    
    # Throughput-based selection
    if recent_tp < 600:
        return 0
    elif recent_tp < 1200:
        return 1
    elif recent_tp < 2200:
        return 2
    elif recent_tp < 3500:
        return 3
    else:
        return 4


def collect_expert_demonstrations(n_episodes=200):
    """
    Collect expert demonstrations
    """
    print("="*80)
    print("🎓 PHASE 1: Collecting Expert Demonstrations")
    print("="*80)
    
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    
    env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        video_dir='data/videos',
        mode='train'
    )
    
    states = []
    actions = []
    total_reward = 0
    
    print(f"\nCollecting {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # Expert action
            action = hybrid_expert(env)
            
            # Store state-action pair
            states.append({
                'network': state['network'].copy(),
                'content': state['content'].copy(),
                'vmaf': state['vmaf'].copy()
            })
            actions.append(action)
            
            # Step
            state, reward, done, info = env.step(action)
            ep_reward += reward
        
        total_reward += ep_reward
        
        if (ep + 1) % 50 == 0:
            avg_reward = total_reward / (ep + 1)
            print(f"  Episodes: {ep+1}/{n_episodes} | "
                  f"Transitions: {len(states)} | "
                  f"Avg Reward: {avg_reward:+.2f}")
    
    avg_reward = total_reward / n_episodes
    
    print(f"\n✅ Collection Complete:")
    print(f"   Total transitions: {len(states)}")
    print(f"   Expert avg reward: {avg_reward:+.2f}")
    print(f"   Actions distribution:")
    
    action_counts = np.bincount(actions, minlength=6)
    for i, count in enumerate(action_counts):
        percentage = count / len(actions) * 100
        print(f"      Action {i}: {count:5d} ({percentage:5.1f}%)")
    
    return states, actions


def train_behavioral_cloning(states, actions, n_epochs=30):
    """
    Train model via behavioral cloning
    """
    print("\n" + "="*80)
    print("🧠 PHASE 1: Behavioral Cloning Training")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    print(f"\nPreparing dataset ({len(states)} transitions)...")
    network_states = torch.FloatTensor([s['network'] for s in states]).to(device)
    content_states = torch.FloatTensor([s['content'] for s in states]).to(device)
    vmaf_states = torch.FloatTensor([s['vmaf'] for s in states]).to(device)
    action_labels = torch.LongTensor(actions).to(device)
    
    dataset_size = len(states)
    batch_size = 128
    
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
        
        # Save best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs('results', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            }, 'results/bc_pretrained.pth')
            print(f"   → Saved (best accuracy: {best_accuracy:.2f}%)")
    
    print(f"\n✅ BC Training Complete")
    print(f"   Best accuracy: {best_accuracy:.2f}%")
    print(f"   Model saved to: results/bc_pretrained.pth")
    
    return model


if __name__ == '__main__':
    print("="*80)
    print("🚀 PHASE 1: BEHAVIORAL CLONING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Collect demonstrations
        states, actions = collect_expert_demonstrations(n_episodes=200)
        
        # Train BC
        model = train_behavioral_cloning(states, actions, n_epochs=30)
        
        print("\n" + "="*80)
        print("✅ PHASE 1 COMPLETE")
        print("="*80)
        print("Next: Run phase2_rl_finetuning.py")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()