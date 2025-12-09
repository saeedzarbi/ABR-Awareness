"""
Evaluate Hybrid Model
Simple evaluation script for hybrid trained model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env_3 import ABREnv
import numpy as np
import pandas as pd

# ============================================================================
# Evaluate Model
# ============================================================================

def evaluate_model(
    model_path='best_model',
    num_episodes=20,
    videos=['bigbuckbunny', 'parkjoy', 'tearsofsteel_short'],
    trace_dir='data/standardized/test_traces',
    vmaf_dir='data/vmaf_scores',
    siti_dir='data/content_features'
):
    """
    Evaluate trained model
    """
    
    print("="*70)
    print("🎯 Evaluating Hybrid Model")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Videos: {videos}")
    print(f"Episodes per video: {num_episodes}")
    print()
    
    # Load model
    print(f"Loading model from: {model_path}")
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create environment
    env = ABREnv(
        video_names=videos,
        trace_dir=trace_dir,
        vmaf_dir=vmaf_dir,
        siti_dir=siti_dir,
        max_chunks=48
    )
    
    # Evaluate
    all_results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        episode_reward = 0
        episode_vmaf = 0
        episode_rebuffer = 0
        episode_switches = 0
        last_action = 0
        steps = 0
        
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            vmaf = env.vmaf_scores[env.BITRATE_LEVELS[action]]
            episode_vmaf += vmaf
            episode_rebuffer += info.get('rebuffer', 0)
            
            if action != last_action:
                episode_switches += 1
            
            last_action = action
            steps += 1
        
        # Save episode results
        result = {
            'episode': episode + 1,
            'video': env.current_video_name,
            'reward': episode_reward,
            'avg_vmaf': episode_vmaf / steps,
            'total_rebuffer': episode_rebuffer,
            'rebuffer_pct': (episode_rebuffer / (steps * 4.0)) * 100,
            'switches': episode_switches,
            'steps': steps
        }
        
        all_results.append(result)
        
        # Print progress
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"VMAF={result['avg_vmaf']:.2f}, "
                  f"Rebuf={result['rebuffer_pct']:.2f}%, "
                  f"Switch={result['switches']}")
    
    # Calculate summary statistics
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("📊 Results Summary")
    print("="*70)
    
    # Overall
    print("\nOverall (all episodes):")
    print(f"  Avg VMAF:      {df['avg_vmaf'].mean():.2f}")
    print(f"  Avg Rebuffer:  {df['rebuffer_pct'].mean():.2f}%")
    print(f"  Avg Switches:  {df['switches'].mean():.1f}")
    print(f"  Avg Reward:    {df['reward'].mean():.2f}")
    
    # Per video
    print("\nPer Video:")
    for video in videos:
        video_df = df[df['video'] == video]
        if len(video_df) > 0:
            print(f"\n  {video}:")
            print(f"    VMAF:     {video_df['avg_vmaf'].mean():.2f} (±{video_df['avg_vmaf'].std():.2f})")
            print(f"    Rebuffer: {video_df['rebuffer_pct'].mean():.2f}%")
            print(f"    Switches: {video_df['switches'].mean():.1f}")
    
    # Calculate QoE (approximate)
    # QoE = sum(VMAF) - 50*rebuffer - switches
    total_vmaf = df['avg_vmaf'].sum()
    total_rebuffer = df['total_rebuffer'].sum()
    total_switches = df['switches'].sum()
    
    qoe = total_vmaf - (50 * total_rebuffer) - total_switches
    avg_qoe = qoe / len(df)
    
    print(f"\nApproximate QoE: {avg_qoe:.1f}")
    
    # Save results
    output_file = f'hybrid_evaluation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    print("="*70)
    
    return df

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate hybrid model')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per video')
    parser.add_argument('--videos', nargs='+', 
                       default=['bigbuckbunny', 'parkjoy', 'tearsofsteel_short'])
    parser.add_argument('--trace-dir', type=str, required=True)
    parser.add_argument('--vmaf-dir', type=str, required=True)
    parser.add_argument('--siti-dir', type=str, required=True)
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        videos=args.videos,
        trace_dir=args.trace_dir,
        vmaf_dir=args.vmaf_dir,
        siti_dir=args.siti_dir
    )