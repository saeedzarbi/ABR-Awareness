# """
# Debug Script: Check Checkpoint and VMAF Calculation
# Find out why evaluation results are bad
# """

# import os
# import sys
# import torch
# import numpy as np
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent))

# from models.content_aware_model import ContentAwareActor
# from models.content_aware_env_fcc import ContentAwareEnvFCC
# from models.fcc_trace_loader import FCCTraceLoader


# def check_checkpoints():
#     """Check all available checkpoints"""
    
#     print("="*80)
#     print("🔍 CHECKPOINT INVESTIGATION")
#     print("="*80)
    
#     checkpoint_paths = [
#         'results/advanced_training/best_model.pth',
#         'results/advanced_training/checkpoint_250.pth',
#         'results/advanced_training/checkpoint_240.pth',
#         'results/advanced_training/checkpoint_260.pth',
#         'results/advanced_trainings/best_model.pth',
#         'results/fcc_training_improved/checkpoint_best.pth',
#     ]
    
#     print("\n📦 Searching for checkpoints...")
#     print("-"*80)
    
#     found_checkpoints = []
    
#     for path in checkpoint_paths:
#         if os.path.exists(path):
#             print(f"✅ FOUND: {path}")
            
#             try:
#                 ckpt = torch.load(path, map_location='cpu')
                
#                 print(f"   Update: {ckpt.get('update', 'N/A')}")
#                 print(f"   Timestep: {ckpt.get('timestep', 'N/A')}")
#                 print(f"   Val Reward: {ckpt.get('reward', 'N/A')}")
#                 print(f"   Val Rebuffer: {ckpt.get('rebuffer', 'N/A')}")
                
#                 found_checkpoints.append({
#                     'path': path,
#                     'update': ckpt.get('update'),
#                     'reward': ckpt.get('reward'),
#                     'data': ckpt
#                 })
                
#             except Exception as e:
#                 print(f"   ❌ Error loading: {e}")
            
#             print()
#         else:
#             print(f"❌ NOT FOUND: {path}")
    
#     if not found_checkpoints:
#         print("\n⚠️  No checkpoints found!")
#         return None
    
#     # Find best checkpoint
#     print("-"*80)
#     print("🏆 Best Checkpoint by Val Reward:")
#     print("-"*80)
    
#     valid_ckpts = [c for c in found_checkpoints if c['reward'] is not None]
    
#     if valid_ckpts:
#         best = max(valid_ckpts, key=lambda x: float(x['reward']))
#         print(f"   Path: {best['path']}")
#         print(f"   Update: {best['update']}")
#         print(f"   Reward: {best['reward']:+.2f}")
#     else:
#         print("   No checkpoints with reward info!")
#         best = found_checkpoints[0]
    
#     return best


# def test_vmaf_calculation():
#     """Test if VMAF is calculated correctly"""
    
#     print("\n" + "="*80)
#     print("🧪 VMAF CALCULATION TEST")
#     print("="*80)
    
#     # Create environment
#     print("\n📦 Creating environment...")
    
#     fcc_loader = FCCTraceLoader(
#         fcc_trace_dir='data/fcc_traces',
#         train_file='data/network_traces/fcc/splits/fcc_train.txt',
#         val_file='data/network_traces/fcc/splits/fcc_val.txt',
#         test_file='data/network_traces/fcc/splits/fcc_test.txt'
#     )
    
#     env = ContentAwareEnvFCC(
#         fcc_trace_loader=fcc_loader,
#         features_file='data/features/si_ti_features.json',
#         vmaf_file='data/vmaf/vmaf_table.json',
#         video_dir='data/videos',
#         mode='test'
#     )
    
#     print("   ✅ Environment created")
    
#     # Test VMAF for different videos and bitrates
#     print("\n🎬 Testing VMAF values...")
#     print("-"*80)
#     print(f"{'Video':<12} {'Bitrate':<10} {'VMAF (State)':<15} {'VMAF (Info)'}")
#     print("-"*80)
    
#     test_videos = [1, 3, 5]  # sports, news, game
#     test_actions = [0, 2, 4, 5]  # Different bitrates
    
#     for video_id in test_videos:
#         state = env.reset(video_id=video_id)
#         video_name = env.get_video_name()
        
#         # Check VMAF in state
#         vmaf_state = state['vmaf'] * 100.0  # Convert back to 0-100
        
#         for action in test_actions:
#             bitrate = env.bitrate_levels[action]
#             vmaf_prediction = vmaf_state[action]
            
#             # Take a step
#             next_state, reward, done, info = env.step(action)
#             vmaf_info = info.get('vmaf', 0)
            
#             print(f"{video_name:<12} {bitrate:<10} {vmaf_prediction:>12.1f}   {vmaf_info:>12.1f}")
            
#             if done:
#                 break
        
#         print()
    
#     print("-"*80)
    
#     # Analysis
#     print("\n📊 Analysis:")
    
#     # Reset and check one episode
#     state = env.reset(video_id=1)
#     episode_vmafs = []
    
#     for _ in range(10):
#         action = 3  # Medium-high bitrate (2850 kbps)
#         state, reward, done, info = env.step(action)
#         episode_vmafs.append(info.get('vmaf', 0))
        
#         if done:
#             break
    
#     mean_vmaf = np.mean(episode_vmafs)
    
#     print(f"   10 chunks with 2850 kbps bitrate:")
#     print(f"   Mean VMAF: {mean_vmaf:.1f}")
    
#     if mean_vmaf < 50:
#         print(f"   ⚠️  WARNING: VMAF is very low ({mean_vmaf:.1f})")
#         print(f"   Expected: 70-80 for 2850 kbps")
#         print(f"\n   Possible issues:")
#         print(f"   1. VMAF normalization problem (0-1 vs 0-100)")
#         print(f"   2. VMAF table has wrong values")
#         print(f"   3. Wrong video/bitrate lookup")
#     else:
#         print(f"   ✅ VMAF values look reasonable")


# def compare_evaluation_results(checkpoint):
#     """Quick evaluation to compare with full results"""
    
#     print("\n" + "="*80)
#     print("🎯 QUICK EVALUATION (5 episodes)")
#     print("="*80)
    
#     # Load model
#     model = ContentAwareActor(
#         state_dim=(6, 8),
#         action_dim=6,
#         content_dim=2
#     )
#     model.load_state_dict(checkpoint['data']['model_state_dict'])
#     model.eval()
    
#     print(f"   ✅ Model loaded from: {checkpoint['path']}")
    
#     # Create environment
#     fcc_loader = FCCTraceLoader(
#         fcc_trace_dir='data/fcc_traces',
#         train_file='data/network_traces/fcc/splits/fcc_train.txt',
#         val_file='data/network_traces/fcc/splits/fcc_val.txt',
#         test_file='data/network_traces/fcc/splits/fcc_test.txt'
#     )
    
#     env = ContentAwareEnvFCC(
#         fcc_trace_loader=fcc_loader,
#         features_file='data/features/si_ti_features.json',
#         vmaf_file='data/vmaf/vmaf_table.json',
#         video_dir='data/videos',
#         mode='test'
#     )
    
#     # Run 5 episodes
#     print("\n   Running 5 test episodes...")
    
#     results = []
    
#     for ep in range(5):
#         state = env.reset(video_id=(ep % 6) + 1)
        
#         ep_reward = 0
#         ep_rebuffer = 0
#         ep_vmafs = []
#         ep_bitrates = []
#         done = False
        
#         while not done:
#             with torch.no_grad():
#                 net = torch.FloatTensor(state['network']).unsqueeze(0)
#                 cont = torch.FloatTensor(state['content']).unsqueeze(0)
#                 vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
                
#                 action_probs, _ = model(net, cont, vmaf)
#                 action = action_probs.argmax(dim=1).item()
            
#             state, reward, done, info = env.step(action)
            
#             ep_reward += reward
#             ep_rebuffer += info['rebuffer_time']
#             ep_vmafs.append(info['vmaf'])
#             ep_bitrates.append(info['bitrate'])
        
#         results.append({
#             'reward': ep_reward,
#             'rebuffer': ep_rebuffer,
#             'vmaf': np.mean(ep_vmafs),
#             'bitrate': np.mean(ep_bitrates)
#         })
        
#         print(f"   Episode {ep+1}: R={ep_reward:+.1f}, "
#               f"Rebuf={ep_rebuffer:.2f}s, "
#               f"VMAF={np.mean(ep_vmafs):.1f}, "
#               f"BR={np.mean(ep_bitrates):.0f}kbps")
    
#     # Summary
#     print("\n   Quick Evaluation Summary:")
#     print(f"   Mean Reward: {np.mean([r['reward'] for r in results]):+.2f}")
#     print(f"   Mean Rebuffer: {np.mean([r['rebuffer'] for r in results]):.2f}s")
#     print(f"   Mean VMAF: {np.mean([r['vmaf'] for r in results]):.1f}")
#     print(f"   Mean Bitrate: {np.mean([r['bitrate'] for r in results]):.0f}kbps")


# def main():
#     """Main debug function"""
    
#     try:
#         # Step 1: Check checkpoints
#         best_ckpt = check_checkpoints()
        
#         if best_ckpt is None:
#             print("\n❌ Cannot proceed without checkpoints!")
#             return
        
#         # Step 2: Test VMAF calculation
#         test_vmaf_calculation()
        
#         # Step 3: Quick evaluation
#         compare_evaluation_results(best_ckpt)
        
#         print("\n" + "="*80)
#         print("✅ DEBUG COMPLETE")
#         print("="*80)
        
#     except Exception as e:
#         print(f"\n❌ Error: {str(e)}")
#         import traceback
#         traceback.print_exc()


# if __name__ == '__main__':
#     main()
"""
Load and Compare All Checkpoints
Handles PyTorch 2.6 compatibility issues
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader


def safe_load_checkpoint(checkpoint_path: str) -> dict:
    """
    Safely load checkpoint with PyTorch 2.6 compatibility
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Checkpoint dict or None if failed
    """
    try:
        # Try with weights_only=True first (secure)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print(f"   ✅ Loaded with weights_only=True (secure)")
        return checkpoint
    except Exception as e1:
        try:
            # Add safe globals for numpy
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print(f"   ✅ Loaded with safe globals")
            return checkpoint
        except Exception as e2:
            try:
                # Last resort: weights_only=False (less secure but works)
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"   ⚠️  Loaded with weights_only=False (less secure)")
                return checkpoint
            except Exception as e3:
                print(f"   ❌ Failed to load: {str(e3)[:100]}")
                return None


def load_and_info(checkpoint_path: str) -> dict:
    """Load checkpoint and extract info"""
    
    if not os.path.exists(checkpoint_path):
        return None
    
    print(f"\n📦 Loading: {checkpoint_path}")
    
    checkpoint = safe_load_checkpoint(checkpoint_path)
    
    if checkpoint is None:
        return None
    
    # Extract info
    info = {
        'path': checkpoint_path,
        'checkpoint': checkpoint,
        'update': checkpoint.get('update', 'N/A'),
        'timestep': checkpoint.get('timestep', 'N/A'),
        'reward': checkpoint.get('reward') or checkpoint.get('best_val_reward'),
        'rebuffer': checkpoint.get('rebuffer', 'N/A')
    }
    
    print(f"   Update: {info['update']}")
    print(f"   Timestep: {info['timestep']}")
    print(f"   Val Reward: {info['reward']}")
    print(f"   Val Rebuffer: {info['rebuffer']}")
    
    return info


def quick_evaluate(checkpoint_info: dict, n_episodes: int = 5) -> dict:
    """
    Quick evaluation of checkpoint
    
    Args:
        checkpoint_info: Checkpoint info dict
        n_episodes: Number of episodes to test
        
    Returns:
        Evaluation results
    """
    print(f"\n🎯 Quick Evaluation ({n_episodes} episodes)")
    
    # Load model
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    )
    
    try:
        model.load_state_dict(checkpoint_info['checkpoint']['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"   ❌ Failed to load model weights: {e}")
        return None
    
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
        mode='val'  # Use validation set
    )
    
    # Run episodes
    results = []
    
    for ep in range(n_episodes):
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
        
        print(f"   Ep {ep+1}: R={ep_reward:+7.1f}, "
              f"Rebuf={ep_rebuffer:5.2f}s, "
              f"VMAF={np.mean(ep_vmafs):5.1f}, "
              f"BR={np.mean(ep_bitrates):4.0f}kbps")
    
    # Summary
    summary = {
        'mean_reward': np.mean([r['reward'] for r in results]),
        'mean_rebuffer': np.mean([r['rebuffer'] for r in results]),
        'mean_vmaf': np.mean([r['vmaf'] for r in results]),
        'mean_bitrate': np.mean([r['bitrate'] for r in results])
    }
    
    print(f"\n   Summary:")
    print(f"   Reward:    {summary['mean_reward']:+.2f}")
    print(f"   Rebuffer:  {summary['mean_rebuffer']:.2f}s")
    print(f"   VMAF:      {summary['mean_vmaf']:.1f}")
    print(f"   Bitrate:   {summary['mean_bitrate']:.0f}kbps")
    
    return summary


def compare_all_checkpoints():
    """Compare all available checkpoints"""
    
    print("="*80)
    print("🔍 COMPARING ALL CHECKPOINTS")
    print("="*80)
    
    # List of checkpoints to check
    checkpoint_paths = [
        'results/advanced_training/best_model.pth',
        'results/advanced_training/checkpoint_250.pth',
        'results/advanced_training/checkpoint_240.pth',
        'results/fcc_training_improved/checkpoint_best.pth',
        'results/fcc_training_improved_new/checkpoint_best.pth',
    ]
    
    loaded_checkpoints = []
    
    # Load all checkpoints
    for path in checkpoint_paths:
        info = load_and_info(path)
        if info is not None:
            loaded_checkpoints.append(info)
    
    if not loaded_checkpoints:
        print("\n❌ No checkpoints could be loaded!")
        return
    
    # Evaluate each
    print("\n" + "="*80)
    print("📊 EVALUATING ALL CHECKPOINTS")
    print("="*80)
    
    all_results = []
    
    for ckpt_info in loaded_checkpoints:
        print("\n" + "-"*80)
        print(f"Testing: {os.path.basename(ckpt_info['path'])}")
        print(f"Update {ckpt_info['update']}, Val Reward: {ckpt_info['reward']}")
        print("-"*80)
        
        eval_result = quick_evaluate(ckpt_info, n_episodes=5)
        
        if eval_result is not None:
            all_results.append({
                'name': os.path.basename(ckpt_info['path']),
                'update': ckpt_info['update'],
                'checkpoint_reward': ckpt_info['reward'],
                'eval': eval_result
            })
    
    # Final comparison
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON")
    print("="*80)
    
    if not all_results:
        print("\n❌ No successful evaluations!")
        return
    
    print(f"\n{'Checkpoint':<35} {'Ckpt R':>10} {'Eval R':>10} {'Rebuf':>10} {'VMAF':>8}")
    print("-"*73)
    
    for result in all_results:
        ckpt_r = result['checkpoint_reward']
        ckpt_r_str = f"{ckpt_r:+.1f}" if ckpt_r is not None else "N/A"
        
        print(f"{result['name']:<35} "
              f"{ckpt_r_str:>10} "
              f"{result['eval']['mean_reward']:>+10.1f} "
              f"{result['eval']['mean_rebuffer']:>9.2f}s "
              f"{result['eval']['mean_vmaf']:>8.1f}")
    
    print("-"*73)
    
    # Find best
    best = max(all_results, key=lambda x: x['eval']['mean_reward'])
    
    print(f"\n🏆 Best Checkpoint: {best['name']}")
    print(f"   Eval Reward: {best['eval']['mean_reward']:+.2f}")
    print(f"   Rebuffering: {best['eval']['mean_rebuffer']:.2f}s")
    print(f"   VMAF: {best['eval']['mean_vmaf']:.1f}")
    print(f"   Bitrate: {best['eval']['mean_bitrate']:.0f}kbps")
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)


def main():
    """Main function"""
    
    try:
        compare_all_checkpoints()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()