"""
Network Trace Loader
Loads and manages real network traces for training
"""

import numpy as np
import random
from pathlib import Path
import json


class NetworkTrace:
    """Single network trace"""
    
    def __init__(self, trace_id, timestamps, throughputs, metadata=None):
        self.trace_id = trace_id
        self.timestamps = np.array(timestamps, dtype=np.float32)
        self.throughputs = np.array(throughputs, dtype=np.float32)
        self.metadata = metadata or {}
        self.duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 0 else 0
        
        # Precompute statistics
        self.mean = np.mean(self.throughputs)
        self.std = np.std(self.throughputs)
        self.min = np.min(self.throughputs)
        self.max = np.max(self.throughputs)
    
    def get_throughput(self, timestamp):
        """
        Get throughput at specific timestamp (with interpolation)
        """
        if timestamp >= self.timestamps[-1]:
            return self.throughputs[-1]
        
        if timestamp <= self.timestamps[0]:
            return self.throughputs[0]
        
        # Linear interpolation
        idx = np.searchsorted(self.timestamps, timestamp)
        if idx == 0:
            return self.throughputs[0]
        
        # Interpolate between idx-1 and idx
        t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
        tp0, tp1 = self.throughputs[idx-1], self.throughputs[idx]
        
        alpha = (timestamp - t0) / (t1 - t0)
        return tp0 + alpha * (tp1 - tp0)
    
    def __repr__(self):
        return f"NetworkTrace({self.trace_id}, duration={self.duration:.0f}s, mean={self.mean:.0f}kbps)"


class TraceLoader:
    """
    Load and manage network traces
    """
    
    def __init__(self, trace_dir='data/network_traces/cooked_traces', train_ratio=0.7, val_ratio=0.15):
        self.trace_dir = Path(trace_dir)
        self.traces = []
        
        # Load all traces
        self._load_traces()
        
        # Split train/val/test
        self._split_traces(train_ratio, val_ratio)
        
        print(f"✓ Loaded {len(self.traces)} traces")
        print(f"  Train: {len(self.train_traces)}")
        print(f"  Val: {len(self.val_traces)}")
        print(f"  Test: {len(self.test_traces)}")
    
    def _load_traces(self):
        """Load all trace files"""
        
        trace_files = list(self.trace_dir.glob('*'))
        
        for trace_file in trace_files:
            try:
                trace = self._parse_pensieve_trace(trace_file)
                if trace is not None:
                    self.traces.append(trace)
            except Exception as e:
                print(f"Warning: Could not load {trace_file.name}: {e}")
        
        if not self.traces:
            raise ValueError(f"No traces found in {self.trace_dir}")
    
    def _parse_pensieve_trace(self, trace_file):
        """
        Parse Pensieve trace format:
        timestamp(s)  throughput(Mbps)
        
        Convert Mbps to kbps
        """
        timestamps = []
        throughputs = []
        
        with open(trace_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        tp_mbps = float(parts[1])
                        
                        # Convert Mbps to kbps
                        tp_kbps = tp_mbps * 1000.0
                        
                        timestamps.append(t)
                        throughputs.append(tp_kbps)
                        
                    except ValueError:
                        continue
        
        if len(timestamps) < 10:  # Too short
            return None
        
        # Create trace object
        trace = NetworkTrace(
            trace_id=trace_file.stem,
            timestamps=timestamps,
            throughputs=throughputs,
            metadata={'source': 'pensieve', 'unit': 'kbps'}
        )
        
        return trace
    
    def _split_traces(self, train_ratio, val_ratio):
        """Split traces into train/val/test"""
        
        # Shuffle
        random.seed(42)
        random.shuffle(self.traces)
        
        n = len(self.traces)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        self.train_traces = self.traces[:n_train]
        self.val_traces = self.traces[n_train:n_train+n_val]
        self.test_traces = self.traces[n_train+n_val:]
    
    def sample_trace(self, split='train'):
        """Sample a random trace"""
        
        if split == 'train':
            return random.choice(self.train_traces)
        elif split == 'val':
            return random.choice(self.val_traces)
        elif split == 'test':
            return random.choice(self.test_traces)
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def get_trace_by_id(self, trace_id):
        """Get specific trace by ID"""
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None
    
    def get_statistics(self, split='train'):
        """Get statistics for a split"""
        
        if split == 'train':
            traces = self.train_traces
        elif split == 'val':
            traces = self.val_traces
        else:
            traces = self.test_traces
        
        means = [t.mean for t in traces]
        stds = [t.std for t in traces]
        
        return {
            'count': len(traces),
            'mean_throughput': np.mean(means),
            'std_throughput': np.std(means),
            'mean_variability': np.mean(stds)
        }


# ============================================
# Test
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing TraceLoader")
    print("=" * 60)
    
    # Load traces
    loader = TraceLoader(trace_dir='data/network_traces/cooked_traces')
    
    # Sample traces
    print("\nSampling traces:")
    for i in range(5):
        trace = loader.sample_trace('train')
        print(f"  {i+1}. {trace}")
    
    # Statistics
    print("\nStatistics:")
    for split in ['train', 'val', 'test']:
        stats = loader.get_statistics(split)
        print(f"  {split.capitalize()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean throughput: {stats['mean_throughput']:.0f} kbps")
        print(f"    Mean variability: {stats['mean_variability']:.0f} kbps")
    
    # Test interpolation
    print("\nTesting interpolation:")
    trace = loader.sample_trace('train')
    print(f"  Trace: {trace.trace_id}")
    for t in [0, 10, 20, 30]:
        tp = trace.get_throughput(t)
        print(f"    t={t:2d}s: {tp:7.0f} kbps ({tp/1000:.2f} Mbps)")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
