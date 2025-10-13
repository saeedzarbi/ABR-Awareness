"""
Curriculum Learning Training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.ppo_trainer import PPOTrainer
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_v2 import ContentAwareEnvV2
import numpy as np


def analyze_trace_difficulty(env):
    """Analyze traces and sort by difficulty"""
    
    print("Analyzing trace difficulty...")
    
    trace_stats = []
    
    for trace_idx in range(len(env.trace_loader.train_traces)):
        trace = env.trace_loader.train_traces[trace_idx]
        
        # Calculate stats
        throughputs = [t['throughput'] for t in trace]
        mean_tp = np.mean(throughputs)
        std_tp = np.std(throughputs)
        min_tp = np.min(throughputs)
        
        # Difficulty score (lower mean, higher variance = harder)
        difficulty = -mean_tp / 1000 + std_tp / 1000 + (1000 / min_tp)
        
        trace_stats.append({
            'idx': trace_idx,
            'difficulty': difficulty,
            'mean': mean_tp,
            'std': std_tp
        })
    
    # Sort by difficulty
    trace_stats.sort(key=lambda x: x['difficulty'])
    
    return trace_stats


def curriculum_training(timesteps_per_phase=200000):
    """Train with curriculum"""
    
    print("=" * 70)
    print("Curriculum Learning Training")
    print("=" * 70)
    
    # Analyze traces
    env = ContentAwareEnvV2(use_real_traces=True)
    trace_stats = analyze_trace_difficulty(env)
    
    num_traces = len(trace_stats)
    
    # Phase 1: Easy (33% easiest)
    easy_indices = [t['idx'] for t in trace_stats[:num_traces//3]]
    print(f"\nPhase 1: {len(easy_indices)} easy traces")
    
    # Phase 2: Medium (middle 33%)
    medium_indices = [t['idx'] for t in trace_stats[num_traces//3:2*num_traces//3]]
    print(f"Phase 2: {len(medium_indices)} medium traces")
    
    # Phase 3: All
    print(f"Phase 3: All {num_traces} traces")
    
    # Create model
    model = create_content_aware_model()
    trainer = PPOTrainer(model)
    
    # Phase 1: Easy
    print("\n" + "=" * 70)
    print("PHASE 1: Easy Traces")
    print("=" * 70)
    
    env_easy = ContentAwareEnvV2(use_real_traces=True)
    env_easy.trace_loader.train_traces = [
        env.trace_loader.train_traces[i] for i in easy_indices
    ]
    
    trainer.train(
        env_easy,
        total_timesteps=timesteps_per_phase,
        run_name="curriculum_phase1"
    )
    
    # Phase 2: Medium
    print("\n" + "=" * 70)
    print("PHASE 2: Medium Traces")
    print("=" * 70)
    
    env_medium = ContentAwareEnvV2(use_real_traces=True)
    env_medium.trace_loader.train_traces = [
        env.trace_loader.train_traces[i] for i in medium_indices
    ]
    
    trainer.train(
        env_medium,
        total_timesteps=timesteps_per_phase,
        run_name="curriculum_phase2"
    )
    
    # Phase 3: All
    print("\n" + "=" * 70)
    print("PHASE 3: All Traces")
    print("=" * 70)
    
    trainer.train(
        env,
        total_timesteps=timesteps_per_phase,
        run_name="curriculum_phase3"
    )
    
    print("\n✓ Curriculum training complete!")


if __name__ == '__main__':
    curriculum_training(timesteps_per_phase=200000)
