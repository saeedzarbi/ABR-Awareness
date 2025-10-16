# scripts/training/train_comyco.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from tqdm import tqdm

from models.content_aware_model import ContentAwareActor

def train_comyco():
    print("="*60)
    print("🎓 Training Comyco-style Model via Imitation Learning")
    print("="*60)
    
    # ۱. لود کردن داده‌های متخصص
    data_path = 'data/expert_data.json'
    print(f"📦 Loading expert data from '{data_path}'...")
    with open(data_path, 'r') as f:
        expert_data = json.load(f)
    print(f"   Loaded {len(expert_data):,} samples.")
    
    # ۲. ساخت مدل (از همان مدل آگاه از محتوا استفاده می‌کنیم)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # ۳. حلقه آموزش
    epochs = 5
    batch_size = 128
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        np.random.shuffle(expert_data)
        
        total_loss = 0
        
        for i in tqdm(range(0, len(expert_data), batch_size), desc=f"Epoch {epoch+1}"):
            batch = expert_data[i:i+batch_size]
            
            # آماده‌سازی batch
            network_states = torch.FloatTensor([item['state']['network'] for item in batch]).to(device)
            content_features = torch.FloatTensor([item['state']['content'] for item in batch]).to(device)
            vmaf_predictions = torch.FloatTensor([item['state']['vmaf'] for item in batch]).to(device)
            expert_actions = torch.LongTensor([item['action'] for item in batch]).to(device)
            
            # آموزش
            optimizer.zero_grad()
            action_probs, _ = model(network_states, content_features, vmaf_predictions)
            
            loss = criterion(action_probs, expert_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"   Average Loss: {total_loss / (len(expert_data)/batch_size):.4f}")

    # ۴. ذخیره مدل آموزش‌دیده
    output_path = 'results/comyco_model.pth'
    torch.save(model.state_dict(), output_path)
    print(f"\n✅ Comyco model trained and saved to '{output_path}'")
    print("="*60)

if __name__ == '__main__':
    train_comyco()