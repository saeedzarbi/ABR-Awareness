"""
Collect Expert Demonstrations from RobustMPC
Generates training data for imitation learning
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
import json

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from abr_multi_env_v13 import ABREnv

# ============================================================================
# RobustMPC Implementation
# ============================================================================

class RobustMPC:
    """
    RobustMPC algorithm for ABR
    Look-ahead optimization with robustness
    """
    
    BITRATE_LEVELS = [300, 750, 1200, 1850, 2850, 6000]
    CHUNK_DURATION = 4.0
    BUFFER_NORM = 10.0
    SMOOTH_PENALTY = 1.0
    REBUFFER_PENALTY = 4.3  # RobustMPC uses lower penalty
    
    def __init__(self, env):
        self.env = env
        self.future_chunk_count = 5  # Look ahead 5 chunks
        
    def predict_throughput(self, past_throughputs):
        """
        Harmonic mean of past throughputs (robust prediction)
        """
        if len(past_throughputs) == 0:
            return 1000  # Default
        
        # Convert normalized to kbps
        tp_kbps = [tp * 6000 for tp in past_throughputs[-8:]]
        
        # Harmonic mean (more conservative)
        harmonic_mean = len(tp_kbps) / sum(1.0 / (tp + 1e-6) for tp in tp_kbps)
        
        return harmonic_mean
    
    def compute_quality(self, bitrate_kbps):
        """Get VMAF for bitrate from environment"""
        return self.env.vmaf_scores.get(bitrate_kbps, 35.0)
    
    def simulate_download(self, bitrate_idx, buffer_level, predicted_tp):
        """
        Simulate downloading a chunk
        Returns: new_buffer, rebuffer_time, download_time
        """
        bitrate_kbps = self.BITRATE_LEVELS[bitrate_idx]
        chunk_size_bits = bitrate_kbps * 1000 * self.CHUNK_DURATION
        
        download_time = chunk_size_bits / (predicted_tp * 1000 + 1e-6)
        rebuffer_time = max(0, download_time - buffer_level)
        new_buffer = max(0, buffer_level - download_time) + self.CHUNK_DURATION
        new_buffer = min(new_buffer, 30.0)  # Buffer max
        
        return new_buffer, rebuffer_time, download_time
    
    def evaluate_action(self, action_idx, current_state, last_quality):
        """
        Evaluate an action with look-ahead
        Returns: total_reward
        """
        buffer_level = current_state['buffer']
        throughput_history = current_state['throughput_history']
        
        # Predict throughput
        predicted_tp = self.predict_throughput(throughput_history)
        
        # Simulate this action
        current_bitrate = self.BITRATE_LEVELS[action_idx]
        current_quality = self.compute_quality(current_bitrate)
        
        buffer, rebuffer, _ = self.simulate_download(
            action_idx, buffer_level, predicted_tp
        )
        
        # Immediate reward
        reward = current_quality \
                 - self.REBUFFER_PENALTY * rebuffer \
                 - self.SMOOTH_PENALTY * abs(current_quality - last_quality)
        
        # Look-ahead simulation (simple: assume same action)
        future_reward = 0
        future_buffer = buffer
        future_quality = current_quality
        
        for _ in range(self.future_chunk_count):
            future_buffer, future_rebuffer, _ = self.simulate_download(
                action_idx, future_buffer, predicted_tp
            )
            
            future_reward += current_quality \
                           - self.REBUFFER_PENALTY * future_rebuffer \
                           - self.SMOOTH_PENALTY * abs(current_quality - future_quality)
            
            future_quality = current_quality
        
        # Discount future reward
        total_reward = reward + 0.99 * future_reward / self.future_chunk_count
        
        return total_reward
    
    def select_action(self, observation, last_action=0):
        """
        Select best action using MPC
        """
        # Parse observation
        tp_history = observation[:8].tolist()
        buffer = observation[8] * 30.0  # Denormalize
        last_bitrate_idx = int(observation[9] * 5)  # Denormalize
        
        last_quality = self.compute_quality(self.BITRATE_LEVELS[last_bitrate_idx])
        
        current_state = {
            'buffer': buffer,
            'throughput_history': tp_history
        }
        
        # Evaluate all actions
        best_action = 0
        best_reward = -float('inf')
        
        for action_idx in range(len(self.BITRATE_LEVELS)):
            reward = self.evaluate_action(action_idx, current_state, last_quality)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action_idx
        
        return best_action

# ============================================================================
# Data Collection
# ============================================================================

def collect_demonstrations(
    num_episodes=1000,
    videos=['bigbuckbunny', 'tearsofsteel_short', 'parkjoy'],
    trace_dir='/path/to/train_traces',
    vmaf_dir='/path/to/vmaf_scores',
    siti_dir='/path/to/content_features',
    output_file='expert_demonstrations.pkl'
):
    """
    Collect expert demonstrations from RobustMPC
    
    Returns:
        demonstrations: list of (state, action) pairs
    """
    
    print("="*70)
    print("🎓 Collecting Expert Demonstrations from RobustMPC")
    print("="*70)
    print(f"Videos: {videos}")
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_file}")
    print()
    
    demonstrations = []
    episode_rewards = []
    episode_vmaf = []
    episode_rebuffer = []
    
    # Create environment
    env = ABREnv(
        video_names=videos,
        trace_dir=trace_dir,
        vmaf_dir=vmaf_dir,
        siti_dir=siti_dir,
        max_chunks=48
    )
    
    # Create RobustMPC agent
    agent = RobustMPC(env)
    
    # Collect episodes
    for episode in tqdm(range(num_episodes), desc="Collecting"):
        obs, info = env.reset()
        episode_data = []
        
        total_reward = 0
        total_vmaf = 0
        total_rebuffer = 0
        last_action = 0
        
        done = False
        step = 0
        
        while not done:
            # Get expert action
            action = agent.select_action(obs, last_action)
            
            # Save (state, action) pair
            episode_data.append({
                'state': obs.copy(),
                'action': action,
                'video': env.current_video_name
            })
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            total_vmaf += info.get('vmaf', env.vmaf_scores[env.BITRATE_LEVELS[action]])
            total_rebuffer += info.get('rebuffer', 0)
            
            last_action = action
            step += 1
        
        # Save episode
        demonstrations.extend(episode_data)
        episode_rewards.append(total_reward)
        episode_vmaf.append(total_vmaf / step)
        episode_rebuffer.append(total_rebuffer)
        
        # Log every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_vmaf = np.mean(episode_vmaf[-100:])
            avg_rebuf = np.mean(episode_rebuffer[-100:])
            
            print(f"\nEpisode {episode+1}:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg VMAF: {avg_vmaf:.2f}")
            print(f"  Avg Rebuffer: {avg_rebuf:.3f}s")
    
    # Summary
    print("\n" + "="*70)
    print("📊 Collection Summary")
    print("="*70)
    print(f"Total demonstrations: {len(demonstrations)}")
    print(f"Avg reward: {np.mean(episode_rewards):.2f}")
    print(f"Avg VMAF: {np.mean(episode_vmaf):.2f}")
    print(f"Avg rebuffer: {np.mean(episode_rebuffer):.3f}s")
    print()
    
    # Action distribution
    actions = [d['action'] for d in demonstrations]
    action_dist = np.bincount(actions, minlength=6) / len(actions)
    print("Action distribution:")
    for i, prob in enumerate(action_dist):
        print(f"  Bitrate {i} ({agent.BITRATE_LEVELS[i]} kbps): {prob*100:.1f}%")
    
    # Save
    data = {
        'demonstrations': demonstrations,
        'metadata': {
            'num_episodes': num_episodes,
            'total_demonstrations': len(demonstrations),
            'videos': videos,
            'avg_reward': float(np.mean(episode_rewards)),
            'avg_vmaf': float(np.mean(episode_vmaf)),
            'avg_rebuffer': float(np.mean(episode_rebuffer)),
            'action_distribution': action_dist.tolist()
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✅ Saved to: {output_file}")
    print("="*70)
    
    return demonstrations

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect expert demonstrations')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--videos', nargs='+', default=['bigbuckbunny', 'tearsofsteel_short', 'parkjoy'])
    parser.add_argument('--trace-dir', type=str, required=True)
    parser.add_argument('--vmaf-dir', type=str, required=True)
    parser.add_argument('--siti-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='expert_demonstrations.pkl')
    
    args = parser.parse_args()
    
    collect_demonstrations(
        num_episodes=args.episodes,
        videos=args.videos,
        trace_dir=args.trace_dir,
        vmaf_dir=args.vmaf_dir,
        siti_dir=args.siti_dir,
        output_file=args.output
    )
