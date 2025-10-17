import torch
import os

print("=" * 70)
print("🔍 Checking Checkpoints")
print("=" * 70)

checkpoint_dir = 'results/fcc_training/'

if os.path.exists(checkpoint_dir):
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    
    print(f"\nFound {len(files)} checkpoint files:\n")
    
    for f in files:
        path = os.path.join(checkpoint_dir, f)
        try:
            ckpt = torch.load(path, map_location='cpu')
            update = ckpt.get('update', 'N/A')
            train_info = ckpt.get('train_info', {})
            reward = train_info.get('mean_reward', 'N/A')
            
            print(f"📦 {f}")
            print(f"   Update: {update}")
            print(f"   Train Reward: {reward}")
            
            if 'model_state_dict' in ckpt:
                n_params = sum(p.numel() for p in ckpt['model_state_dict'].values())
                print(f"   Parameters: {n_params:,}")
            print()
            
        except Exception as e:
            print(f"❌ {f}: Error - {e}\n")
else:
    print("❌ Directory not found!")