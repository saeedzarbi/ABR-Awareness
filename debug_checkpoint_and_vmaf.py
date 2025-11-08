"""
Debug Script: Check Checkpoint and VMAF Calculation
Find out why evaluation results are bad
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader


def check_checkpoints():
    """Check all available checkpoints"""
    
    print("="*80)
    print("🔍 CHECKPOINT INVESTIGATION")
    print("="*80)
    
    checkpoint_paths = [
        'results/advanced_training/best_model.pth',
        'results/advanced_training/checkpoint_250.pth',
        'results/advanced_training/checkpoint_240.pth',
        'results/advanced_training/checkpoint_260.pth',
        'results/advanced_trainings/best_model.pth',
        'results/fcc_training_improved_new/checkpoint_best.pth',
    ]
    
    print("\n📦 Searching for checkpoints...")
    print("-"*80)
    
    found_checkpoints = []
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✅ FOUND: {path}")
            
            try:
                ckpt = torch.load(path, map_location='cpu')
                
                print(f"   Update: {ckpt.get('update', 'N/A')}")
                print(f"   Timestep: {ckpt.get('timestep', 'N/A')}")
                print(f"   Val Reward: {ckpt.get('reward', 'N/A')}")
                print(f"   Val Rebuffer: {ckpt.get('rebuffer', 'N/A')}")
                
                found_checkpoints.append({
                    'path': path,
                    'update': ckpt.get('update'),
                    'reward': ckpt.get('reward'),
                    'data': ckpt
                })
                
            except Exception as e:
                print(f"   ❌ Error loading: {e}")
            
            print()
        else:
            print(f"❌ NOT FOUND: {path}")
    
    if not found_checkpoints:
        print("\n⚠️  No checkpoints found!")
        return None
    
    # Find best checkpoint
    print("-"*80)
    print("🏆 Best Checkpoint by Val Reward:")
    print("-"*80)
    
    valid_ckpts = [c for c in found_checkpoints if c['reward'] is not None]
    
    if valid_ckpts:
        best = max(valid_ckpts, key=lambda x: float(x['reward']))
        print(f"   Path: {best['path']}")
        print(f"   Update: {best['update']}")
        print(f"   Reward: {best['reward']:+.2f}")
    else:
        print("   No checkpoints with reward info!")
        best = found_checkpoints[0]
    
    return best


def test_vmaf_calculation():
    """Test if VMAF is calculated correctly"""
    
    print("\n" + "="*80)
    print("🧪 VMAF CALCULATION TEST")
    print("="*80)
    
    # Create environment
    print("\n📦 Creating environment...")
    
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
        mode='test'
    )
    
    print("   ✅ Environment created")
    
    # Test VMAF for different videos and bitrates
    print("\n🎬 Testing VMAF values...")
    print("-"*80)
    print(f"{'Video':<12} {'Bitrate':<10} {'VMAF (State)':<15} {'VMAF (Info)'}")
    print("-"*80)
    
    test_videos = [1, 3, 5]  # sports, news, game
    test_actions = [0, 2, 4, 5]  # Different bitrates
    
    for video_id in test_videos:
        state = env.reset(video_id=video_id)
        video_name = env.get_video_name()
        
        # Check VMAF in state
        vmaf_state = state['vmaf'] * 100.0  # Convert back to 0-100
        
        for action in test_actions:
            bitrate = env.bitrate_levels[action]
            vmaf_prediction = vmaf_state[action]
            
            # Take a step
            next_state, reward, done, info = env.step(action)
            vmaf_info = info.get('vmaf', 0)
            
            print(f"{video_name:<12} {bitrate:<10} {vmaf_prediction:>12.1f}   {vmaf_info:>12.1f}")
            
            if done:
                break
        
        print()
    
    print("-"*80)
    
    # Analysis
    print("\n📊 Analysis:")
    
    # Reset and check one episode
    state = env.reset(video_id=1)
    episode_vmafs = []
    
    for _ in range(10):
        action = 3  # Medium-high bitrate (2850 kbps)
        state, reward, done, info = env.step(action)
        episode_vmafs.append(info.get('vmaf', 0))
        
        if done:
            break
    
    mean_vmaf = np.mean(episode_vmafs)
    
    print(f"   10 chunks with 2850 kbps bitrate:")
    print(f"   Mean VMAF: {mean_vmaf:.1f}")
    
    if mean_vmaf < 50:
        print(f"   ⚠️  WARNING: VMAF is very low ({mean_vmaf:.1f})")
        print(f"   Expected: 70-80 for 2850 kbps")
        print(f"\n   Possible issues:")
        print(f"   1. VMAF normalization problem (0-1 vs 0-100)")
        print(f"   2. VMAF table has wrong values")
        print(f"   3. Wrong video/bitrate lookup")
    else:
        print(f"   ✅ VMAF values look reasonable")


def compare_evaluation_results(checkpoint):
    """Quick evaluation to compare with full results"""
    
    print("\n" + "="*80)
    print("🎯 QUICK EVALUATION (5 episodes)")
    print("="*80)
    
    # Load model
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    )
    model.load_state_dict(checkpoint['data']['model_state_dict'])
    model.eval()
    
    print(f"   ✅ Model loaded from: {checkpoint['path']}")
    
    # Create environment
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
        mode='test'
    )
    
    # Run 5 episodes
    print("\n   Running 5 test episodes...")
    
    results = []
    
    for ep in range(5):
        state = env.reset(video_id=(ep % 6) + 1)
        
        ep_reward = 0
        ep_rebuffer = 0
        ep_vmafs = []
        ep_bitrates = []
        done = False
        
        while not done:
            with torch.no_grad():
                net = torch.FloatTensor(state['network']).unsqueeze(0)
                cont = torch.FloatTensor(state['content']).unsqueeze(0)
                vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
                
                action_probs, _ = model(net, cont, vmaf)
                action = action_probs.argmax(dim=1).item()
            
            state, reward, done, info = env.step(action)
            
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_vmafs.append(info['vmaf'])
            ep_bitrates.append(info['bitrate'])
        
        results.append({
            'reward': ep_reward,
            'rebuffer': ep_rebuffer,
            'vmaf': np.mean(ep_vmafs),
            'bitrate': np.mean(ep_bitrates)
        })
        
        print(f"   Episode {ep+1}: R={ep_reward:+.1f}, "
              f"Rebuf={ep_rebuffer:.2f}s, "
              f"VMAF={np.mean(ep_vmafs):.1f}, "
              f"BR={np.mean(ep_bitrates):.0f}kbps")
    
    # Summary
    print("\n   Quick Evaluation Summary:")
    print(f"   Mean Reward: {np.mean([r['reward'] for r in results]):+.2f}")
    print(f"   Mean Rebuffer: {np.mean([r['rebuffer'] for r in results]):.2f}s")
    print(f"   Mean VMAF: {np.mean([r['vmaf'] for r in results]):.1f}")
    print(f"   Mean Bitrate: {np.mean([r['bitrate'] for r in results]):.0f}kbps")


def main():
    """Main debug function"""
    
    try:
        # Step 1: Check checkpoints
        best_ckpt = check_checkpoints()
        
        if best_ckpt is None:
            print("\n❌ Cannot proceed without checkpoints!")
            return
        
        # Step 2: Test VMAF calculation
        test_vmaf_calculation()
        
        # Step 3: Quick evaluation
        compare_evaluation_results(best_ckpt)
        
        print("\n" + "="*80)
        print("✅ DEBUG COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()