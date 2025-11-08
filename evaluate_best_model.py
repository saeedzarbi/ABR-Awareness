"""
Evaluate Best Checkpoint (Reward +111) on Full Test Set
Complete evaluation for IEEE TCSVT paper submission
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader


class ModelEvaluator:
    """Evaluator for trained ABR model"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Initialize evaluator
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print("="*80)
        print("🎯 MODEL EVALUATION FOR TCSVT PAPER")
        print("="*80)
        
        # Load checkpoint
        self.model, self.checkpoint = self._load_checkpoint()
        
        # Video mapping
        self.video_names = {
            1: 'sports',
            2: 'animation', 
            3: 'news',
            4: 'nature',
            5: 'game',
            6: 'movie'
        }
    
    def _load_checkpoint(self) -> Tuple[ContentAwareActor, dict]:
        """Load model from checkpoint"""
        
        print(f"\n📦 Loading Checkpoint:")
        print(f"   Path: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Create model
        model = ContentAwareActor(
            state_dim=(6, 8),
            action_dim=6,
            content_dim=2
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        print(f"   ✅ Model loaded successfully")
        print(f"   📊 Checkpoint info:")
        print(f"      Update: {checkpoint.get('update', 'N/A')}")
        print(f"      Timestep: {checkpoint.get('timestep', 'N/A')}")
        print(f"      Val Reward: {checkpoint.get('reward', 'N/A')}")
        print(f"      Val Rebuffer: {checkpoint.get('rebuffer', 'N/A')}")
        
        return model, checkpoint
    
    def evaluate_episode(
        self, 
        env: ContentAwareEnvFCC, 
        video_id: int = None,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate single episode
        
        Args:
            env: Environment instance
            video_id: Specific video (None for random)
            deterministic: Use greedy policy (True) or sample (False)
            
        Returns:
            Episode results dict
        """
        state = env.reset(video_id=video_id)
        
        episode_reward = 0.0
        episode_rebuffer = 0.0
        episode_bitrates = []
        episode_vmafs = []
        episode_smoothness = []
        chunk_details = []
        
        done = False
        last_bitrate = 0
        
        while not done:
            # Prepare state tensors
            with torch.no_grad():
                net = torch.FloatTensor(state['network']).unsqueeze(0).to(self.device)
                cont = torch.FloatTensor(state['content']).unsqueeze(0).to(self.device)
                vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(self.device)
                
                # Get action
                action_probs, _ = self.model(net, cont, vmaf)
                
                if deterministic:
                    action = action_probs.argmax(dim=1).item()
                else:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().item()
            
            # Step environment
            state, reward, done, info = env.step(action)
            
            # Collect metrics
            episode_reward += reward
            episode_rebuffer += info['rebuffer_time']
            episode_bitrates.append(info['bitrate'])
            episode_vmafs.append(info['vmaf'])
            
            # Smoothness
            if last_bitrate > 0:
                smoothness = abs(info['bitrate'] - last_bitrate)
                episode_smoothness.append(smoothness)
            last_bitrate = info['bitrate']
            
            # Detailed chunk info
            chunk_details.append({
                'chunk': info['chunk_idx'],
                'bitrate': info['bitrate'],
                'vmaf': info['vmaf'],
                'rebuffer': info['rebuffer_time'],
                'buffer': info['buffer'],
                'throughput': info['throughput']
            })
        
        return {
            'reward': episode_reward,
            'rebuffer': episode_rebuffer,
            'mean_bitrate': np.mean(episode_bitrates),
            'std_bitrate': np.std(episode_bitrates),
            'mean_vmaf': np.mean(episode_vmafs),
            'std_vmaf': np.std(episode_vmafs),
            'mean_smoothness': np.mean(episode_smoothness) if episode_smoothness else 0,
            'video_id': env.video_id,
            'video_name': env.get_video_name(),
            'num_chunks': len(chunk_details),
            'chunks': chunk_details
        }
    
    def evaluate_test_set(
        self,
        env: ContentAwareEnvFCC,
        n_episodes_per_video: int = 10
    ) -> Dict:
        """
        Full evaluation on test set
        
        Args:
            env: Test environment
            n_episodes_per_video: Episodes per video
            
        Returns:
            Complete evaluation results
        """
        print("\n" + "="*80)
        print("🧪 FULL TEST SET EVALUATION")
        print("="*80)
        print(f"   Videos: 6 (sports, animation, news, nature, game, movie)")
        print(f"   Episodes per video: {n_episodes_per_video}")
        print(f"   Total episodes: {6 * n_episodes_per_video}")
        print("="*80)
        
        video_ids = list(range(1, 7))  # 1-6
        all_results = []
        per_video_results = {vid: [] for vid in video_ids}
        
        # Evaluate each video
        for video_id in video_ids:
            video_name = self.video_names[video_id]
            print(f"\n📹 Evaluating Video {video_id}: {video_name}")
            print("-"*80)
            
            for ep in range(n_episodes_per_video):
                result = self.evaluate_episode(env, video_id=video_id)
                
                all_results.append(result)
                per_video_results[video_id].append(result)
                
                # Progress
                print(f"   Ep {ep+1:2d}/{n_episodes_per_video}: "
                      f"R={result['reward']:+7.1f}, "
                      f"Rebuf={result['rebuffer']:5.2f}s, "
                      f"BR={result['mean_bitrate']:4.0f}kbps, "
                      f"VMAF={result['mean_vmaf']:5.1f}")
        
        # Compute statistics
        return self._compute_statistics(all_results, per_video_results, video_ids)
    
    def _compute_statistics(
        self,
        all_results: List[Dict],
        per_video_results: Dict[int, List[Dict]],
        video_ids: List[int]
    ) -> Dict:
        """Compute overall and per-video statistics"""
        
        print("\n" + "="*80)
        print("📊 RESULTS SUMMARY")
        print("="*80)
        
        # Overall statistics
        all_rewards = [r['reward'] for r in all_results]
        all_rebuffers = [r['rebuffer'] for r in all_results]
        all_bitrates = [r['mean_bitrate'] for r in all_results]
        all_vmafs = [r['mean_vmaf'] for r in all_results]
        all_smoothness = [r['mean_smoothness'] for r in all_results]
        
        overall_stats = {
            'reward_mean': float(np.mean(all_rewards)),
            'reward_std': float(np.std(all_rewards)),
            'rebuffer_mean': float(np.mean(all_rebuffers)),
            'rebuffer_std': float(np.std(all_rebuffers)),
            'bitrate_mean': float(np.mean(all_bitrates)),
            'bitrate_std': float(np.std(all_bitrates)),
            'vmaf_mean': float(np.mean(all_vmafs)),
            'vmaf_std': float(np.std(all_vmafs)),
            'smoothness_mean': float(np.mean(all_smoothness)),
            'smoothness_std': float(np.std(all_smoothness)),
            'n_episodes': len(all_results)
        }
        
        print(f"\n🌍 Overall Performance ({len(all_results)} episodes):")
        print(f"   Reward:      {overall_stats['reward_mean']:+8.2f} ± {overall_stats['reward_std']:6.2f}")
        print(f"   Rebuffering: {overall_stats['rebuffer_mean']:8.2f} ± {overall_stats['rebuffer_std']:6.2f} s")
        print(f"   Bitrate:     {overall_stats['bitrate_mean']:8.0f} ± {overall_stats['bitrate_std']:6.0f} kbps")
        print(f"   VMAF:        {overall_stats['vmaf_mean']:8.1f} ± {overall_stats['vmaf_std']:6.1f}")
        print(f"   Smoothness:  {overall_stats['smoothness_mean']:8.0f} ± {overall_stats['smoothness_std']:6.0f} kbps")
        
        # Per-video statistics
        print(f"\n📹 Per-Video Breakdown:")
        print("-"*80)
        print(f"{'Video':<12} {'Reward':>12} {'Rebuffer':>12} {'Bitrate':>12} {'VMAF':>10}")
        print("-"*80)
        
        per_video_stats = {}
        
        for video_id in video_ids:
            video_results = per_video_results[video_id]
            video_name = self.video_names[video_id]
            
            rewards = [r['reward'] for r in video_results]
            rebuffers = [r['rebuffer'] for r in video_results]
            bitrates = [r['mean_bitrate'] for r in video_results]
            vmafs = [r['mean_vmaf'] for r in video_results]
            
            print(f"{video_name:<12} "
                  f"{np.mean(rewards):+7.1f}±{np.std(rewards):4.1f}  "
                  f"{np.mean(rebuffers):6.2f}±{np.std(rebuffers):4.2f}s  "
                  f"{np.mean(bitrates):6.0f}±{np.std(bitrates):4.0f}  "
                  f"{np.mean(vmafs):5.1f}±{np.std(vmafs):3.1f}")
            
            per_video_stats[video_name] = {
                'reward_mean': float(np.mean(rewards)),
                'reward_std': float(np.std(rewards)),
                'rebuffer_mean': float(np.mean(rebuffers)),
                'rebuffer_std': float(np.std(rebuffers)),
                'bitrate_mean': float(np.mean(bitrates)),
                'bitrate_std': float(np.std(bitrates)),
                'vmaf_mean': float(np.mean(vmafs)),
                'vmaf_std': float(np.std(vmafs))
            }
        
        print("-"*80)
        
        return {
            'overall': overall_stats,
            'per_video': per_video_stats,
            'all_episodes': all_results,
            'checkpoint_info': {
                'path': self.checkpoint_path,
                'update': self.checkpoint.get('update'),
                'timestep': self.checkpoint.get('timestep')
            },
            'evaluation_time': datetime.now().isoformat()
        }


def main():
    """Main evaluation script"""
    
    # Configuration
    CHECKPOINT_PATHS = [
        'results/advanced_training/best_model.pth',
        'results/advanced_training/checkpoint_250.pth',
        'results/fcc_training_improved/checkpoint_best.pth'
    ]
    
    # Find existing checkpoint
    checkpoint_path = None
    for path in CHECKPOINT_PATHS:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoint found!")
        print("   Searched paths:")
        for path in CHECKPOINT_PATHS:
            print(f"   - {path}")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create evaluator
    evaluator = ModelEvaluator(checkpoint_path, device=device)
    
    # Create test environment
    print("\n🏗️  Creating Test Environment...")
    
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    
    test_env = ContentAwareEnvFCC(
        fcc_trace_loader=fcc_loader,
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        video_dir='data/videos',
        mode='test'
    )
    
    print("   ✅ Test environment ready")
    
    # Run evaluation
    results = evaluator.evaluate_test_set(test_env, n_episodes_per_video=10)
    
    # Save results
    output_dir = 'results/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'best_model_test_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print paper-ready summary
    print("\n" + "="*80)
    print("📝 FOR IEEE TCSVT PAPER")
    print("="*80)
    
    overall = results['overall']
    
    print(f"\nProposed Method Performance (Test Set):")
    print(f"   • QoE Reward:     {overall['reward_mean']:+.2f} ± {overall['reward_std']:.2f}")
    print(f"   • Rebuffering:    {overall['rebuffer_mean']:.2f} ± {overall['rebuffer_std']:.2f} s")
    print(f"   • Average Bitrate: {overall['bitrate_mean']:.0f} ± {overall['bitrate_std']:.0f} kbps")
    print(f"   • VMAF Score:     {overall['vmaf_mean']:.1f} ± {overall['vmaf_std']:.1f}")
    print(f"   • Bitrate Switches: {overall['smoothness_mean']:.0f} ± {overall['smoothness_std']:.0f} kbps")
    print(f"   • Test Episodes:  {overall['n_episodes']}")
    
    print("\n" + "="*80)
    print("🎯 Comparison with Baselines:")
    print("="*80)
    print("""
Method          Reward    Rebuffer   Bitrate    VMAF
--------------------------------------------------------
MPC            +79.23      6.27s     1328 kbps   68.2
Comyco         +92.57      1.02s      601 kbps   62.4
Pensieve      +100.58      2.09s     1169 kbps   71.5
Proposed      +{:.2f}     {:.2f}s    {:.0f} kbps   {:.1f}
--------------------------------------------------------
Improvement   +{:.1f}%    -{:.1f}%   +{:.1f}%    {:.1f}%
    """.format(
        overall['reward_mean'],
        overall['rebuffer_mean'],
        overall['bitrate_mean'],
        overall['vmaf_mean'],
        ((overall['reward_mean'] - 100.58) / 100.58 * 100),
        ((2.09 - overall['rebuffer_mean']) / 2.09 * 100),
        ((overall['bitrate_mean'] - 1169) / 1169 * 100),
        ((overall['vmaf_mean'] - 71.5) / 71.5 * 100)
    ))
    
    print("="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()