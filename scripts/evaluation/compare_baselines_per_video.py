import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.trace_loader import TraceLoader
from models.policy_wrapper import BufferAwarePolicy, SmoothPolicy
from models.video_tracker import PerVideoTracker

# Import baseline policies
from scripts.baselines.mpc_policy import MPCPolicy
from scripts.baselines.pensieve_policy import PensievePolicy  
from scripts.baselines.comyco_policy import ComycoPolicy

def evaluate_method(policy, env, num_episodes, method_name):
    """Evaluate one method and return tracker"""
    tracker = PerVideoTracker()
    
    for ep in tqdm(range(num_episodes), desc=f"Eval ({method_name})"):
        if hasattr(policy, 'reset'):
            policy.reset()
        
        state = env.reset(split='test')
        if state is None:
            continue
        
        video_name = getattr(env, 'current_video', 'unknown')
        
        ep_reward, ep_rebuffer, ep_bitrates = 0, 0, []
        done = False
        
        while not done:
            # Get action based on policy type
            if hasattr(policy, 'select_action'):
                action = policy.select_action(state, getattr(env, 'buffer', 25.0), 0.0)
            else:
                action = policy(state)
            
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_bitrates.append(info['bitrate'])
        
        avg_bitrate = np.mean(ep_bitrates) if ep_bitrates else 0
        tracker.add_episode(video_name, ep_reward, ep_rebuffer, avg_bitrate)
    
    return tracker

def main():
    print("="*100)
    print("🏆 COMPLETE BASELINE COMPARISON with PER-VIDEO BREAKDOWN")
    print("="*100)
    
    # Setup
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trace_dir = 'data/network_traces/cooked_traces'
    loader = TraceLoader(trace_dir=trace_dir)
    
    env = ContentAwareEnvV2(
        trace_dir=trace_dir,
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json'
    )
    
    num_episodes = len(loader.test_traces)
    
    # Dictionary to store all trackers
    all_trackers = {}
    
    # 1. Your Model
    print("\n📊 Evaluating: Your Model")
    model = ContentAwareActor(state_dim=(6,8), action_dim=6, content_dim=2).to(DEVICE)
    checkpoint = torch.load('results/fcc_training_low_lr/checkpoint_best.pth', 
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    buffer_policy = BufferAwarePolicy(model)
    your_policy = SmoothPolicy(buffer_policy, max_jump=2)
    
    all_trackers['Ours'] = evaluate_method(your_policy, env, num_episodes, "Ours")
    
    # 2. Pensieve
    print("\n📊 Evaluating: Pensieve")
    pensieve_policy = PensievePolicy()
    all_trackers['Pensieve'] = evaluate_method(pensieve_policy, env, num_episodes, "Pensieve")
    
    # 3. MPC
    print("\n📊 Evaluating: MPC")
    mpc_policy = MPCPolicy()
    all_trackers['MPC'] = evaluate_method(mpc_policy, env, num_episodes, "MPC")
    
    # 4. Comyco
    print("\n📊 Evaluating: Comyco")
    comyco_policy = ComycoPolicy()
    all_trackers['Comyco'] = evaluate_method(comyco_policy, env, num_episodes, "Comyco")
    
    # Print individual summaries
    for method_name, tracker in all_trackers.items():
        print(f"\n{'='*100}")
        print(f"📊 {method_name} - Per-Video Results")
        print(f"{'='*100}")
        tracker.print_summary()
    
    # Create comparison table
    print("\n" + "="*120)
    print("📊 COMPARISON TABLE (All Methods)")
    print("="*120)
    
    # Get all unique videos
    all_videos = set()
    for tracker in all_trackers.values():
        all_videos.update(tracker.results.keys())
    all_videos = sorted(all_videos)
    
    # Print header
    header = f"{'Video':<20}"
    for method in all_trackers.keys():
        header += f" {method:>15}"
    print(header)
    print("-"*120)
    
    # Print each video
    for video in all_videos:
        row = f"{video:<20}"
        for method_name, tracker in all_trackers.items():
            if video in tracker.results:
                reward = np.mean(tracker.results[video]['rewards'])
                row += f" {reward:>+14.2f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
    
    print("="*120)
    
    # Save comparison to CSV
    comparison_data = []
    for video in all_videos:
        row = {'video': video}
        for method_name, tracker in all_trackers.items():
            if video in tracker.results:
                row[f'{method_name}_reward'] = np.mean(tracker.results[video]['rewards'])
                row[f'{method_name}_rebuffer'] = np.mean(tracker.results[video]['rebuffering'])
                row[f'{method_name}_bitrate'] = np.mean(tracker.results[video]['bitrates'])
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df.to_csv('results/per_video_comparison.csv', index=False)
    print("\n💾 Comparison saved: results/per_video_comparison.csv")

if __name__ == '__main__':
    main()