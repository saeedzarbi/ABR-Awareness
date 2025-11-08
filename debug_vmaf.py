"""
Debug Script: Test VMAF in Environment Info
Check if VMAF is properly tracked after the fix
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
import numpy as np


def test_vmaf_in_info():
    """Test if VMAF is properly included in info dict"""
    
    print("="*80)
    print("🔍 VMAF DEBUG TEST")
    print("="*80)
    
    # Load FCC traces
    print("\n📦 Loading FCC Traces...")
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    print("   ✅ Traces loaded")
    
    # Create environment
    print("\n🏗️  Creating Environment...")
    env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        video_dir='data/videos',
        mode='val'
    )
    print("   ✅ Environment created")
    
    # Reset environment
    print("\n🔄 Resetting Environment...")
    state = env.reset(video_id=1)
    print(f"   ✅ Reset complete")
    print(f"   Video ID: {env.video_id}")
    print(f"   Video Name: {env.get_video_name()}")
    
    # Display state structure
    print("\n📊 State Structure:")
    print(f"   Keys: {list(state.keys())}")
    print(f"   Network shape: {state['network'].shape}")
    print(f"   Content shape: {state['content'].shape}")
    print(f"   VMAF shape: {state['vmaf'].shape}")
    
    # Display VMAF predictions
    print(f"\n🎯 VMAF Predictions (normalized 0-1):")
    for i, vmaf_norm in enumerate(state['vmaf']):
        bitrate = env.bitrate_levels[i]
        vmaf_actual = vmaf_norm * 100.0  # Convert back to 0-100 scale
        print(f"   Action {i} ({bitrate:4d} kbps): {vmaf_norm:.4f} → VMAF {vmaf_actual:.1f}")
    
    # Test multiple actions
    print("\n" + "="*80)
    print("🧪 TESTING VMAF IN INFO DICT")
    print("="*80)
    
    test_actions = [0, 2, 4]  # Low, Medium, High bitrate
    
    for action in test_actions:
        print(f"\n▶️  Testing Action {action} ({env.bitrate_levels[action]} kbps)")
        print("-"*80)
        
        # Perform step
        next_state, reward, done, info = env.step(action)
        
        # Display all info keys
        print(f"   Info keys: {list(info.keys())}")
        
        # Check VMAF
        if 'vmaf' in info:
            print(f"   ✅ VMAF in info: YES")
            print(f"   📊 VMAF value: {info['vmaf']:.2f}")
        else:
            print(f"   ❌ VMAF in info: NO")
            print(f"   ⚠️  VMAF is missing from info dict!")
        
        # Display other metrics
        print(f"\n   Metrics:")
        print(f"   • Bitrate:        {info['bitrate']:.0f} kbps")
        print(f"   • VMAF:           {info.get('vmaf', 'MISSING'):.2f}")
        print(f"   • Rebuffer:       {info['rebuffer_time']:.2f} s")
        print(f"   • Buffer:         {info['buffer']:.2f} s")
        print(f"   • Throughput:     {info['throughput']:.0f} kbps")
        print(f"   • Reward:         {reward:+.2f}")
        
        if done:
            print("\n   ⚠️  Episode finished")
            break
    
    # Final validation
    print("\n" + "="*80)
    print("✅ VALIDATION")
    print("="*80)
    
    # Run one complete step and validate
    state = env.reset(video_id=2)
    next_state, reward, done, info = env.step(3)
    
    checks = {
        'VMAF in info': 'vmaf' in info,
        'VMAF > 0': info.get('vmaf', 0) > 0,
        'VMAF <= 100': info.get('vmaf', 101) <= 100,
        'Bitrate in info': 'bitrate' in info,
        'Rebuffer in info': 'rebuffer_time' in info
    }
    
    all_passed = all(checks.values())
    
    print("\nChecks:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print(f"   VMAF value: {info['vmaf']:.2f}")
        print(f"   Environment is working correctly")
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("   Please review the environment implementation")
    
    print("="*80)
    
    return all_passed, info


def test_episode_with_vmaf():
    """Test complete episode and track VMAF"""
    
    print("\n" + "="*80)
    print("🎬 FULL EPISODE TEST WITH VMAF TRACKING")
    print("="*80)
    
    # Setup
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
        mode='val'
    )
    
    # Run episode
    state = env.reset(video_id=3)
    print(f"\n📹 Video: {env.get_video_name()}")
    
    episode_vmafs = []
    episode_bitrates = []
    episode_rewards = []
    
    # Simple policy: medium bitrate (action 2)
    done = False
    step = 0
    
    print("\nRunning 10 chunks:")
    print("-"*80)
    
    while not done and step < 10:
        action = 2  # Medium bitrate
        next_state, reward, done, info = env.step(action)
        
        episode_vmafs.append(info.get('vmaf', 0))
        episode_bitrates.append(info['bitrate'])
        episode_rewards.append(reward)
        
        print(f"Chunk {info['chunk_idx']:2d}: "
              f"Bitrate={info['bitrate']:4.0f}kbps, "
              f"VMAF={info.get('vmaf', 0):5.1f}, "
              f"Reward={reward:+7.2f}, "
              f"Rebuffer={info['rebuffer_time']:5.2f}s")
        
        step += 1
        state = next_state
    
    # Summary
    print("\n" + "-"*80)
    print("Episode Summary:")
    print(f"   Mean VMAF:    {np.mean(episode_vmafs):.2f}")
    print(f"   Mean Bitrate: {np.mean(episode_bitrates):.0f} kbps")
    print(f"   Total Reward: {sum(episode_rewards):+.2f}")
    print(f"   VMAF Range:   {min(episode_vmafs):.1f} - {max(episode_vmafs):.1f}")
    print("="*80)


def main():
    """Main debug function"""
    
    try:
        # Test 1: Basic VMAF check
        passed, info = test_vmaf_in_info()
        
        if passed:
            # Test 2: Full episode
            test_episode_with_vmaf()
            
            print("\n" + "="*80)
            print("✅ VMAF DEBUG COMPLETE")
            print("="*80)
            print("\nKey Findings:")
            print("   • VMAF is properly tracked in info dict")
            print("   • VMAF values are in correct range (0-100)")
            print("   • Environment is ready for evaluation")
            print("\n💡 Next Step:")
            print("   Run: python evaluate_best_model.py")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ VMAF DEBUG FAILED")
            print("="*80)
            print("\nPossible Issues:")
            print("   • ContentAwareEnvV2.step() not updated")
            print("   • VMAF not added to info dict")
            print("   • Check models/content_aware_env_v2.py")
            print("="*80)
            
    except Exception as e:
        print(f"\n❌ Error during debug: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()