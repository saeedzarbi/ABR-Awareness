"""
Network Trace Loader
Loads and manages real network traces for training and evaluation.
(Clean version — no circular imports)
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
        self.duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 0 else 0
        self.mean = np.mean(self.throughputs)
        self.std = np.std(self.throughputs)
        self.min = np.min(self.throughputs)
        self.max = np.max(self.throughputs)

    def get_throughput(self, timestamp):
        """Return throughput at a specific timestamp using linear interpolation."""
        if len(self.timestamps) == 0:
            return 0.0
        if timestamp <= self.timestamps[0]:
            return self.throughputs[0]
        if timestamp >= self.timestamps[-1]:
            return self.throughputs[-1]
        idx = np.searchsorted(self.timestamps, timestamp)
        t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
        tp0, tp1 = self.throughputs[idx-1], self.throughputs[idx]
        alpha = (timestamp - t0) / (t1 - t0)
        return tp0 + alpha * (tp1 - tp0)

    def __repr__(self):
        return f"NetworkTrace({self.trace_id}, duration={self.duration:.0f}s, mean={self.mean:.0f}kbps)"


class TraceLoader:
    """Loads and manages Pensieve-style network traces."""

    def __init__(self, trace_dir="data/network_traces/cooked_traces", train_ratio=0.7, val_ratio=0.15):
        self.trace_dir = Path(trace_dir)
        self.traces = []
        self._load_traces()
        self._split_traces(train_ratio, val_ratio)

        print(f"✓ Loaded {len(self.traces)} traces")
        print(f"  Train: {len(self.train_traces)}")
        print(f"  Val:   {len(self.val_traces)}")
        print(f"  Test:  {len(self.test_traces)}")

    def _load_traces(self):
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
        timestamps, throughputs = [], []
        with open(trace_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        tp_mbps = float(parts[1])
                        timestamps.append(t)
                        throughputs.append(tp_mbps * 1000.0)  # Mbps -> kbps
                    except ValueError:
                        continue
        if len(timestamps) < 10:
            return None
        return NetworkTrace(trace_file.stem, timestamps, throughputs, metadata={"source":"pensieve","unit":"kbps"})

    def _split_traces(self, train_ratio, val_ratio):
        random.seed(42)
        random.shuffle(self.traces)
        n = len(self.traces)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        self.train_traces = self.traces[:n_train]
        self.val_traces = self.traces[n_train:n_train+n_val]
        self.test_traces = self.traces[n_train+n_val:]

    def sample_trace(self, split="train"):
        if split=="train":
            return random.choice(self.train_traces)
        elif split=="val":
            return random.choice(self.val_traces)
        elif split=="test":
            return random.choice(self.test_traces)
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_trace_by_id(self, trace_id):
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_statistics(self, split="train"):
        if split=="train":
            traces = self.train_traces
        elif split=="val":
            traces = self.val_traces
        else:
            traces = self.test_traces
        means = [t.mean for t in traces]
        stds = [t.std for t in traces]
        return {
            "count": len(traces),
            "mean_throughput": np.mean(means),
            "std_throughput": np.std(means),
            "mean_variability": np.mean(stds)
        }
"""
Network Trace Loader
Loads and manages real network traces for training and evaluation.
(Clean version — no circular imports)
"""

# import numpy as np
# import random
# from pathlib import Path
# import json


# class NetworkTrace:
#     """Represents a single network throughput trace."""

#     def __init__(self, trace_id, timestamps, throughputs, metadata=None):
#         self.trace_id = trace_id
#         self.timestamps = np.array(timestamps, dtype=np.float32)
#         self.throughputs = np.array(throughputs, dtype=np.float32)
#         self.metadata = metadata or {}
#         self.duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 0 else 0
#         self.mean = np.mean(self.throughputs)
#         self.std = np.std(self.throughputs)
#         self.min = np.min(self.throughputs)
#         self.max = np.max(self.throughputs)

#     def get_throughput(self, timestamp):
#         """Return throughput at a specific timestamp using linear interpolation."""
#         if len(self.timestamps) == 0:
#             return 0.0
#         if timestamp <= self.timestamps[0]:
#             return self.throughputs[0]
#         if timestamp >= self.timestamps[-1]:
#             return self.throughputs[-1]
#         idx = np.searchsorted(self.timestamps, timestamp)
#         t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
#         tp0, tp1 = self.throughputs[idx-1], self.throughputs[idx]
#         alpha = (timestamp - t0) / (t1 - t0)
#         return tp0 + alpha * (tp1 - tp0)

#     def __repr__(self):
#         return f"NetworkTrace({self.trace_id}, duration={self.duration:.0f}s, mean={self.mean:.0f}kbps)"


# class TraceLoader:
#     """Loads and manages Pensieve-style network traces."""

#     def __init__(self, trace_dir="data/network_traces/cooked_traces", train_ratio=0.7, val_ratio=0.15):
#         self.trace_dir = Path(trace_dir)
#         self.traces = []
#         self._load_traces()
#         self._split_traces(train_ratio, val_ratio)

#         print(f"✓ Loaded {len(self.traces)} traces")
#         print(f"  Train: {len(self.train_traces)}")
#         print(f"  Val:   {len(self.val_traces)}")
#         print(f"  Test:  {len(self.test_traces)}")

#     def _load_traces(self):
#         trace_files = list(self.trace_dir.glob("*"))
#         for trace_file in trace_files:
#             try:
#                 trace = self._parse_pensieve_trace(trace_file)
#                 if trace is not None:
#                     self.traces.append(trace)
#             except Exception as e:
#                 print(f"⚠️ Warning: Could not load {trace_file.name}: {e}")
#         if not self.traces:
#             raise ValueError(f"No valid traces found in {self.trace_dir}")

#     def _parse_pensieve_trace(self, trace_file):
#         timestamps, throughputs = [], []
#         with open(trace_file, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 2:
#                     try:
#                         t = float(parts[0])
#                         tp_mbps = float(parts[1])
#                         timestamps.append(t)
#                         throughputs.append(tp_mbps * 1000.0)  # Mbps -> kbps
#                     except ValueError:
#                         continue
#         if len(timestamps) < 10:
#             return None
#         return NetworkTrace(trace_file.stem, timestamps, throughputs, metadata={"source":"pensieve","unit":"kbps"})

#     def _split_traces(self, train_ratio, val_ratio):
#         random.seed(42)
#         random.shuffle(self.traces)
#         n = len(self.traces)
#         n_train = int(n * train_ratio)
#         n_val = int(n * val_ratio)
#         self.train_traces = self.traces[:n_train]
#         self.val_traces = self.traces[n_train:n_train+n_val]
#         self.test_traces = self.traces[n_train+n_val:]

#     def sample_trace(self, split="train"):
#         if split=="train":
#             return random.choice(self.train_traces)
#         elif split=="val":
#             return random.choice(self.val_traces)
#         elif split=="test":
#             return random.choice(self.test_traces)
#         else:
#             raise ValueError(f"Unknown split: {split}")

#     def get_trace_files(self, split="train"):
#         """Returns the list of traces for the given split."""
#         if split == "train":
#             return self.train_traces
#         elif split == "val":
#             return self.val_traces
#         elif split == "test":
#             return self.test_traces
#         else:
#             raise ValueError(f"Unknown split: {split}")

#     def get_trace_by_id(self, trace_id):
#         for trace in self.traces:
#             if trace.trace_id == trace_id:
#                 return trace
#         return None

#     def get_statistics(self, split="train"):
#         if split=="train":
#             traces = self.train_traces
#         elif split=="val":
#             traces = self.val_traces
#         else:
#             traces = self.test_traces
#         means = [t.mean for t in traces]
#         stds = [t.std for t in traces]
#         return {
#             "count": len(traces),
#             "mean_throughput": np.mean(means),
#             "std_throughput": np.std(means),
#             "mean_variability": np.mean(stds)
#         }
