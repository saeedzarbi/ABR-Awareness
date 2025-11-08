"""
Evaluate with Multiple Seeds and Report Statistics
This will give us a more reliable estimate of performance
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader


def set_seed(seed: int):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_with_seed(
    model,
    fcc_loader,
    seed: int,
    n_episodes: int = 10
):
    """Evaluate with a specific seed"""
    
    set_seed(seed)
    
    # Create fresh environment
    env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        video_dir='data/videos',
        mode='val'
    )
    
    episode_results = []
    
    for ep in range(n_episodes):
        state = env.reset()
        
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
        
        episode_results.append({
            'reward': ep_reward,
            'rebuffer': ep_rebuffer,
            'vmaf': np.mean(ep_vmafs),
            'bitrate': np.mean(ep_bitrates)
        })
    
    return {
        'mean_reward': np.mean([r['reward'] for r in episode_results]),
        'mean_rebuffer': np.mean([r['rebuffer'] for r in episode_results]),
        'mean_vmaf': np.mean([r['vmaf'] for r in episode_results]),
        'mean_bitrate': np.mean([r['bitrate'] for r in episode_results])
    }


def main():
    """Main evaluation with multiple seeds"""
    
    print("="*80)
    print("🎲 MULTI-SEED EVALUATION")
    print("="*80)
    
    # Load checkpoint
    checkpoint_path = 'results/advanced_training/best_model.pth'
    
    print(f"\n📦 Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    print(f"   Stored Val Reward: {checkpoint['reward']:+.2f}")
    print(f"   Stored Val Rebuffer: {checkpoint.get('rebuffer'):.2f}s")
    
    # Load model
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load FCC traces
    print(f"\n📦 Loading FCC Traces...")
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    
    # Evaluate with multiple seeds
    seeds = [42, 123, 456, 789, 2024]
    n_episodes = 20  # More episodes for better statistics
    
    print(f"\n🎯 Running evaluation with {len(seeds)} different seeds")
    print(f"   Episodes per seed: {n_episodes}")
    print(f"   Total episodes: {len(seeds) * n_episodes}")
    print("="*80)
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\nSeed {i+1}/{len(seeds)}: {seed}")
        print("-"*80)
        
        result = evaluate_with_seed(model, fcc_loader, seed, n_episodes)
        all_results.append(result)
        
        print(f"   Reward:    {result['mean_reward']:+7.2f}")
        print(f"   Rebuffer:  {result['mean_rebuffer']:7.2f}s")
        print(f"   VMAF:      {result['mean_vmaf']:7.1f}")
        print(f"   Bitrate:   {result['mean_bitrate']:7.0f}kbps")
    
    # Aggregate statistics
    print("\n" + "="*80)
    print("📊 AGGREGATE STATISTICS (across all seeds)")
    print("="*80)
    
    rewards = [r['mean_reward'] for r in all_results]
    rebuffers = [r['mean_rebuffer'] for r in all_results]
    vmafs = [r['mean_vmaf'] for r in all_results]
    bitrates = [r['mean_bitrate'] for r in all_results]
    
    print(f"\nReward:")
    print(f"   Mean:  {np.mean(rewards):+.2f}")
    print(f"   Std:   {np.std(rewards):.2f}")
    print(f"   Range: {min(rewards):+.2f} to {max(rewards):+.2f}")
    
    print(f"\nRebuffering:")
    print(f"   Mean:  {np.mean(rebuffers):.2f}s")
    print(f"   Std:   {np.std(rebuffers):.2f}s")
    print(f"   Range: {min(rebuffers):.2f}s to {max(rebuffers):.2f}s")
    
    print(f"\nVMAF:")
    print(f"   Mean:  {np.mean(vmafs):.1f}")
    print(f"   Std:   {np.std(vmafs):.1f}")
    print(f"   Range: {min(vmafs):.1f} to {max(vmafs):.1f}")
    
    print(f"\nBitrate:")
    print(f"   Mean:  {np.mean(bitrates):.0f} kbps")
    print(f"   Std:   {np.std(bitrates):.0f} kbps")
    print(f"   Range: {min(bitrates):.0f} to {max(bitrates):.0f} kbps")
    
    # Compare with stored validation
    print("\n" + "="*80)
    print("📊 COMPARISON WITH TRAINING VALIDATION")
    print("="*80)
    
    stored_reward = checkpoint['reward']
    current_mean = np.mean(rewards)
    difference = current_mean - stored_reward
    difference_pct = (difference / abs(stored_reward)) * 100
    
    print(f"\nStored Val Reward:  {stored_reward:+.2f}")
    print(f"Current Mean:       {current_mean:+.2f} ± {np.std(rewards):.2f}")
    print(f"Difference:         {difference:+.2f} ({difference_pct:+.1f}%)")
    
    if abs(difference) < 5:
        print(f"\n✅ EXCELLENT MATCH! Results are consistent")
    elif abs(difference) < 10:
        print(f"\n✅ GOOD MATCH! Results are within acceptable range")
    elif abs(difference) < 15:
        print(f"\n⚠️  ACCEPTABLE VARIANCE (likely due to trace selection)")
    else:
        print(f"\n❌ SIGNIFICANT MISMATCH!")
        print(f"\nPossible explanations:")
        print(f"   1. Training validation used different traces")
        print(f"   2. Training validation used different seed")
        print(f"   3. Environment behavior changed")
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS FOR PAPER")
    print("="*80)
    
    if abs(difference) < 15:
        print(f"\n✅ Use the multi-seed average:")
        print(f"\nValidation Performance:")
        print(f"   • QoE Reward:     {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
        print(f"   • Rebuffering:    {np.mean(rebuffers):.2f} ± {np.std(rebuffers):.2f} s")
        print(f"   • VMAF Score:     {np.mean(vmafs):.1f} ± {np.std(vmafs):.1f}")
        print(f"   • Average Bitrate: {np.mean(bitrates):.0f} ± {np.std(bitrates):.0f} kbps")
        print(f"\nNote: Report these numbers as validation performance")
    else:
        print(f"\n⚠️  Large variance detected!")
        print(f"\nOptions:")
        print(f"   1. Report training validation: +{stored_reward:.2f}")
        print(f"   2. Report current multi-seed:  +{np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"   3. Investigate environment changes")
    
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()