"""
Reproduce Validation Score from Training
Try to match the +111 validation reward
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


def evaluate_exactly_like_training(checkpoint_path: str, n_episodes: int = 10):
    """
    Evaluate exactly the same way as during training
    To reproduce the +111 validation reward
    """
    
    print("="*80)
    print("🔄 REPRODUCING TRAINING VALIDATION")
    print("="*80)
    
    # Load checkpoint
    print(f"\n📦 Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    print(f"   Update: {checkpoint['update']}")
    print(f"   Stored Val Reward: {checkpoint['reward']:+.2f}")
    print(f"   Stored Val Rebuffer: {checkpoint.get('rebuffer', 'N/A')}")
    
    # Load model
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create environment
    print(f"\n🏗️  Creating Validation Environment...")
    
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    
    val_env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        video_dir='data/videos',
        mode='val'
    )
    
    print(f"   ✅ Environment ready")
    
    # Test 1: Exact reproduction (greedy policy, val set)
    print(f"\n" + "="*80)
    print(f"TEST 1: Greedy Policy on Validation ({n_episodes} episodes)")
    print("="*80)
    
    val_rewards = []
    val_rebuffers = []
    val_vmafs = []
    val_bitrates = []
    
    for ep in range(n_episodes):
        state = val_env.reset()
        
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
                action = action_probs.argmax(dim=1).item()  # Greedy
            
            state, reward, done, info = val_env.step(action)
            
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_vmafs.append(info.get('vmaf', 0))
            ep_bitrates.append(info['bitrate'])
        
        val_rewards.append(ep_reward)
        val_rebuffers.append(ep_rebuffer)
        val_vmafs.append(np.mean(ep_vmafs))
        val_bitrates.append(np.mean(ep_bitrates))
        
        print(f"   Ep {ep+1:2d}: R={ep_reward:+7.1f}, "
              f"Rebuf={ep_rebuffer:5.2f}s, "
              f"VMAF={np.mean(ep_vmafs):5.1f}, "
              f"BR={np.mean(ep_bitrates):4.0f}kbps")
    
    mean_reward = np.mean(val_rewards)
    mean_rebuffer = np.mean(val_rebuffers)
    mean_vmaf = np.mean(val_vmafs)
    mean_bitrate = np.mean(val_bitrates)
    
    print(f"\n   Mean Reward:    {mean_reward:+.2f}")
    print(f"   Mean Rebuffer:  {mean_rebuffer:.2f}s")
    print(f"   Mean VMAF:      {mean_vmaf:.1f}")
    print(f"   Mean Bitrate:   {mean_bitrate:.0f}kbps")
    
    # Compare with stored
    stored_reward = checkpoint['reward']
    difference = mean_reward - stored_reward
    difference_pct = (difference / abs(stored_reward)) * 100
    
    print(f"\n   📊 Comparison:")
    print(f"   Stored Val Reward: {stored_reward:+.2f}")
    print(f"   Current Eval:      {mean_reward:+.2f}")
    print(f"   Difference:        {difference:+.2f} ({difference_pct:+.1f}%)")
    
    if abs(difference) < 5:
        print(f"   ✅ MATCH! Evaluation reproduces training validation")
    elif abs(difference) < 15:
        print(f"   ⚠️  CLOSE but not exact (acceptable variance)")
    else:
        print(f"   ❌ MISMATCH! Something changed in environment")
    
    # Test 2: Check individual episode variance
    print(f"\n" + "="*80)
    print(f"TEST 2: Episode Variance Analysis")
    print("="*80)
    
    print(f"\n   Reward Range: {min(val_rewards):+.1f} to {max(val_rewards):+.1f}")
    print(f"   Reward Std:   {np.std(val_rewards):.1f}")
    print(f"   Rebuffer Range: {min(val_rebuffers):.2f}s to {max(val_rebuffers):.2f}s")
    
    # Check for outliers
    outliers = [r for r in val_rewards if r < mean_reward - 2*np.std(val_rewards)]
    if outliers:
        print(f"\n   ⚠️  Found {len(outliers)} outlier episodes (very low reward)")
        print(f"       This suggests some traces are extremely difficult")
    
    # Test 3: Environment consistency check
    print(f"\n" + "="*80)
    print(f"TEST 3: Environment State Consistency")
    print("="*80)
    
    # Reset and check state
    state1 = val_env.reset(video_id=1)
    state2 = val_env.reset(video_id=1)
    
    # Check if states are different (should be, due to different traces)
    net_diff = np.abs(state1['network'] - state2['network']).sum()
    
    print(f"\n   Two resets of same video:")
    print(f"   Network state difference: {net_diff:.4f}")
    
    if net_diff > 0:
        print(f"   ✅ Environment is stochastic (different traces each time)")
    else:
        print(f"   ⚠️  Environment might be deterministic")
    
    # Test 4: Check VMAF calculation
    print(f"\n" + "="*80)
    print(f"TEST 4: VMAF Calculation Check")
    print("="*80)
    
    state = val_env.reset(video_id=1)
    vmaf_state = state['vmaf'] * 100.0  # Convert to 0-100
    
    print(f"\n   VMAF predictions for video 1, chunk 0:")
    for i, vmaf in enumerate(vmaf_state):
        bitrate = val_env.bitrate_levels[i]
        print(f"      Bitrate {bitrate:4d} kbps → VMAF {vmaf:.1f}")
    
    # Take action and check info
    _, _, _, info = val_env.step(3)  # Medium-high bitrate
    print(f"\n   Selected bitrate: {info['bitrate']} kbps")
    print(f"   VMAF in info: {info.get('vmaf', 'MISSING')}")
    
    if 'vmaf' in info:
        print(f"   ✅ VMAF is in info dict")
        if info['vmaf'] > 0 and info['vmaf'] <= 100:
            print(f"   ✅ VMAF value looks reasonable")
        else:
            print(f"   ⚠️  VMAF value looks wrong")
    else:
        print(f"   ❌ VMAF missing from info dict!")
    
    print("\n" + "="*80)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*80)
    
    return {
        'mean_reward': mean_reward,
        'stored_reward': stored_reward,
        'difference': difference,
        'mean_vmaf': mean_vmaf
    }


def main():
    """Main function"""
    
    checkpoint_path = 'results/advanced_training/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        result = evaluate_exactly_like_training(checkpoint_path, n_episodes=10)
        
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS")
        print("="*80)
        
        if abs(result['difference']) > 15:
            print("\n⚠️  Large mismatch detected!")
            print("\nPossible causes:")
            print("   1. Environment changed (VMAF addition to info)")
            print("   2. Different trace sampling")
            print("   3. Random seed effects")
            
            print("\nSolutions:")
            print("   1. Use validation reward from checkpoint: +111.00")
            print("   2. Re-evaluate on multiple seeds and average")
            print("   3. Check if environment code changed")
        else:
            print("\n✅ Results are consistent!")
            print(f"\nYou can confidently use:")
            print(f"   Validation Reward: {result['mean_reward']:+.2f}")
            print(f"   VMAF: {result['mean_vmaf']:.1f}")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()