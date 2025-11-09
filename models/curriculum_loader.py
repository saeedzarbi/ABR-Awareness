"""
Curriculum Learning for Network Traces
Start with easy traces, gradually increase difficulty
"""

import numpy as np
from typing import List, Dict, Tuple


class CurriculumTraceLoader:
    """
    Manages curriculum learning by progressively loading harder traces
    
    Difficulty factors:
    - Mean throughput (lower = harder)
    - Throughput variance (higher = harder)
    - Minimum throughput (lower = harder)
    """
    
    def __init__(self, fcc_trace_loader, n_samples: int = 100):
        """
        Args:
            fcc_trace_loader: FCCTraceLoader instance
            n_samples: Number of traces to sample for analysis
        """
        self.fcc_loader = fcc_trace_loader
        self.current_difficulty = 0.0
        
        print("\n" + "="*80)
        print("📚 CURRICULUM: Analyzing Network Traces")
        print("="*80)
        
        # Analyze and categorize traces
        self.traces_by_difficulty = self._analyze_traces(n_samples)
        
        # Print curriculum stats
        self._print_curriculum_stats()
        
        print("✅ Curriculum ready!")
        print("="*80)
    
    def _analyze_traces(self, n_samples: int) -> List[Dict]:
        """
        Analyze traces and sort by difficulty
        """
        traces_info = []
        
        print(f"\nAnalyzing {n_samples} traces...")
        
        for i in range(n_samples):
            # Sample trace
            trace = self.fcc_loader.get_trace(mode='train')
            
            # Convert Mbps to kbps
            throughputs = trace[:, 1] * 1000.0
            
            # Compute difficulty metrics
            mean_tp = np.mean(throughputs)
            std_tp = np.std(throughputs)
            min_tp = np.min(throughputs)
            max_tp = np.max(throughputs)
            cv = std_tp / mean_tp if mean_tp > 0 else 0  # Coefficient of variation
            
            # Difficulty score (higher = harder)
            # Factors: low mean, high variance, low minimum
            difficulty = (
                (1.0 / (mean_tp / 1000.0)) * 0.4 +  # Low mean = hard
                cv * 0.3 +                           # High variance = hard
                (1.0 / (min_tp / 100.0)) * 0.3       # Low min = hard
            )
            
            traces_info.append({
                'trace': trace,
                'difficulty': difficulty,
                'mean_tp': mean_tp,
                'std_tp': std_tp,
                'min_tp': min_tp,
                'max_tp': max_tp,
                'cv': cv
            })
            
            if (i + 1) % 25 == 0:
                print(f"  Analyzed {i+1}/{n_samples} traces...")
        
        # Sort by difficulty (easy → hard)
        traces_info.sort(key=lambda x: x['difficulty'])
        
        return traces_info
    
    def _print_curriculum_stats(self):
        """Print curriculum statistics"""
        n_traces = len(self.traces_by_difficulty)
        
        # Easy traces (first 1/3)
        easy_traces = self.traces_by_difficulty[:n_traces//3]
        easy_mean = np.mean([t['mean_tp'] for t in easy_traces])
        easy_std = np.mean([t['std_tp'] for t in easy_traces])
        
        # Medium traces (middle 1/3)
        medium_traces = self.traces_by_difficulty[n_traces//3:2*n_traces//3]
        medium_mean = np.mean([t['mean_tp'] for t in medium_traces])
        medium_std = np.mean([t['std_tp'] for t in medium_traces])
        
        # Hard traces (last 1/3)
        hard_traces = self.traces_by_difficulty[2*n_traces//3:]
        hard_mean = np.mean([t['mean_tp'] for t in hard_traces])
        hard_std = np.mean([t['std_tp'] for t in hard_traces])
        
        print(f"\nCurriculum Statistics:")
        print(f"  Easy traces   (0.0-0.33): Mean TP = {easy_mean:6.1f} kbps, Std = {easy_std:5.1f}")
        print(f"  Medium traces (0.33-0.67): Mean TP = {medium_mean:6.1f} kbps, Std = {medium_std:5.1f}")
        print(f"  Hard traces   (0.67-1.0): Mean TP = {hard_mean:6.1f} kbps, Std = {hard_std:5.1f}")
    
    def get_curriculum_trace(self, difficulty_level: float):
        """
        Get trace based on difficulty level
        
        Args:
            difficulty_level: 0.0 = easiest, 1.0 = hardest
        
        Returns:
            trace: Network trace data
        """
        # Clip to valid range
        difficulty_level = np.clip(difficulty_level, 0.0, 1.0)
        
        # Map to trace index
        idx = int(difficulty_level * (len(self.traces_by_difficulty) - 1))
        idx = np.clip(idx, 0, len(self.traces_by_difficulty) - 1)
        
        return self.traces_by_difficulty[idx]['trace']
    
    def update_difficulty(self, update_num: int, total_updates: int) -> float:
        """
        Compute curriculum difficulty based on training progress
        
        Args:
            update_num: Current update number
            total_updates: Total planned updates
        
        Returns:
            difficulty: Current difficulty level (0-1)
        """
        # Linear curriculum: 0 → 1 over 70% of training
        # Then stay at 1.0 for final 30%
        curriculum_length = int(total_updates * 0.7)
        
        if update_num < curriculum_length:
            difficulty = update_num / curriculum_length
        else:
            difficulty = 1.0
        
        self.current_difficulty = difficulty
        return difficulty
    
    def get_difficulty_stats(self) -> Dict:
        """Get statistics about current difficulty level"""
        idx = int(self.current_difficulty * (len(self.traces_by_difficulty) - 1))
        idx = np.clip(idx, 0, len(self.traces_by_difficulty) - 1)
        
        trace_info = self.traces_by_difficulty[idx]
        
        return {
            'difficulty': self.current_difficulty,
            'mean_throughput': trace_info['mean_tp'],
            'std_throughput': trace_info['std_tp'],
            'min_throughput': trace_info['min_tp'],
            'difficulty_score': trace_info['difficulty']
        }


if __name__ == '__main__':
    print("Testing Curriculum Loader")
    print("="*60)
    
    from models.fcc_trace_loader import FCCTraceLoader
    
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    
    curriculum = CurriculumTraceLoader(fcc_loader, n_samples=50)
    
    # Test different difficulty levels
    print("\n" + "="*60)
    print("Testing Difficulty Progression:")
    print("="*60)
    
    for difficulty in [0.0, 0.25, 0.5, 0.75, 1.0]:
        trace = curriculum.get_curriculum_trace(difficulty)
        throughputs = trace[:, 1] * 1000.0
        
        print(f"\nDifficulty {difficulty:.2f}:")
        print(f"  Mean TP:   {np.mean(throughputs):6.1f} kbps")
        print(f"  Std TP:    {np.std(throughputs):6.1f} kbps")
        print(f"  Min TP:    {np.min(throughputs):6.1f} kbps")
        print(f"  Max TP:    {np.max(throughputs):6.1f} kbps")
    
    print("\n✓ Curriculum tests passed!")