"""
Network Trace Loader
Loads and manages real network traces for training and evaluation.
(✅ Clean version — no circular imports)
"""

import numpy as np
import random
from pathlib import Path
import json


class NetworkTrace:
    """Represents a single network throughput trace."""

    def __init__(self, trace_id, timestamps, throughputs, metadata=None):
        self.trace_id = trace_id
        self.timestamps = np.array(timestamps, dtype=np.float32)
        self.throughputs = np.array(throughputs, dtype=np.float32)
        self.metadata = metadata or {}

        self.duration = (
            self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 0 else 0
        )

        # Precompute simple statistics
        self.mean = np.mean(self.throughputs)
        self.std = np.std(self.throughputs)
        self.min = np.min(self.throughputs)
        self.max = np.max(self.throughputs)

    def get_throughput(self, timestamp):
        """
        Returns the throughput (kbps) at a specific timestamp using linear interpolation.
        """
        if len(self.timestamps) == 0:
            return 0.0

        if timestamp <= self.timestamps[0]:
            return self.throughputs[0]
        if timestamp >= self.timestamps[-1]:
            return self.throughputs[-1]

        idx = np.searchsorted(self.timestamps, timestamp)
        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        tp0, tp1 = self.throughputs[idx - 1], self.throughputs[idx]
        alpha = (timestamp - t0) / (t1 - t0)
        return tp0 + alpha * (tp1 - tp0)

    def __repr__(self):
        return f"NetworkTrace({self.trace_id}, duration={self.duration:.0f}s, mean={self.mean:.0f}kbps)"


class TraceLoader:
    """
    Loads and manages Pensieve-style network traces from disk.
    Handles train/val/test splitting and provides sampling utilities.
    """

    def __init__(self, trace_dir="data/network_traces/cooked_traces", train_ratio=0.7, val_ratio=0.15):
        self.trace_dir = Path(trace_dir)
        self.traces = []

        # Load all traces
        self._load_traces()

        # Split train/val/test
        self._split_traces(train_ratio, val_ratio)

        print(f"✓ Loaded {len(self.traces)} traces")
        print(f"  Train: {len(self.train_traces)}")
        print(f"  Val:   {len(self.val_traces)}")
        print(f"  Test:  {len(self.test_traces)}")

    # ---------------------------------------------------------------------
    # Internal functions
    # ---------------------------------------------------------------------

    def _load_traces(self):
        """Load all trace files from directory."""
        trace_files = list(self.trace_dir.glob("*"))

        for trace_file in trace_files:
            try:
                trace = self._parse_pensieve_trace(trace_file)
                if trace is not None:
                    self.traces.append(trace)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {trace_file.name}: {e}")

        if not self.traces:
            raise ValueError(f"No valid traces found in {self.trace_dir}")

    def _parse_pensieve_trace(self, trace_file):
        """
        Parse a trace in Pensieve format:
        timestamp(s)  throughput(Mbps)
        and convert throughput to kbps.
        """
        timestamps = []
        throughputs = []

        with open(trace_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        tp_mbps = float(parts[1])
                        throughputs.append(tp_mbps * 1000.0)  # Mbps → kbps
                        timestamps.append(t)
                    except ValueError:
                        continue

        if len(timestamps) < 10:
            # Skip too short traces
            return None

        return NetworkTrace(
            trace_id=trace_file.stem,
            timestamps=timestamps,
            throughputs=throughputs,
            metadata={"source": "pensieve", "unit": "kbps"},
        )

    def _split_traces(self, train_ratio, val_ratio):
        """Randomly split traces into train/val/test."""
        random.seed(42)
        random.shuffle(self.traces)

        n = len(self.traces)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self.train_traces = self.traces[:n_train]
        self.val_traces = self.traces[n_train:n_train + n_val]
        self.test_traces = self.traces[n_train + n_val:]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def sample_trace(self, split="train"):
        """Sample a random trace from the specified split."""
        if split == "train":
            return random.choice(self.train_traces)
        elif split == "val":
            return random.choice(self.val_traces)
        elif split == "test":
            return random.choice(self.test_traces)
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_trace_by_id(self, trace_id):
        """Find a specific trace by its ID."""
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_statistics(self, split="train"):
        """Compute mean/std statistics for a given split."""
        if split == "train":
            traces = self.train_traces
        elif split == "val":
            traces = self.val_traces
        else:
            traces = self.test_traces

        means = [t.mean for t in traces]
        stds = [t.std for t in traces]

        return {
            "count": len(traces),
            "mean_throughput": np.mean(means),
            "std_throughput": np.std(means),
            "mean_variability": np.mean(stds),
        }


# ---------------------------------------------------------------------
# Test module standalone
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing TraceLoader")
    print("=" * 60)

    loader = TraceLoader(trace_dir="data/network_traces/cooked_traces")

    print("\nSampling traces:")
    for i in range(3):
        trace = loader.sample_trace("train")
        print(f"  {i+1}. {trace}")

    print("\nStatistics:")
    for split in ["train", "val", "test"]:
        stats = loader.get_statistics(split)
        print(f"  {split.capitalize()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean throughput: {stats['mean_throughput']:.0f} kbps")
        print(f"    Mean variability: {stats['mean_variability']:.0f} kbps")

    print("\n✓ All tests passed!")
    print("=" * 60)
