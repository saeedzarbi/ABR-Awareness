# models/fcc_trace_loader.py

import os
import numpy as np
from typing import List, Dict

class FCCTraceLoader:
    """Trace loader specifically for FCC traces"""
    
    def __init__(self, fcc_trace_dir: str, 
                 train_file: str, val_file: str, test_file: str):
        """
        Args:
            fcc_trace_dir: Directory containing FCC trace files
            train_file: Path to train split file
            val_file: Path to validation split file  
            test_file: Path to test split file
        """
        self.fcc_trace_dir = fcc_trace_dir
        
        # Load splits
        self.train_traces = self._load_trace_list(train_file)
        self.val_traces = self._load_trace_list(val_file)
        self.test_traces = self._load_trace_list(test_file)
        
        print(f"📊 FCC Traces Loaded:")
        print(f"   Train: {len(self.train_traces)}")
        print(f"   Val: {len(self.val_traces)}")
        print(f"   Test: {len(self.test_traces)}")
        
        # Load all traces into memory
        self.traces = {}
        self._load_all_traces()
    
    def _load_trace_list(self, file_path: str) -> List[str]:
        """Load trace names from file"""
        with open(file_path, 'r') as f:
            traces = [line.strip() for line in f if line.strip()]
        return traces
    
    def _load_all_traces(self):
        """Load all trace data into memory"""
        all_trace_names = self.train_traces + self.val_traces + self.test_traces
        
        for trace_name in all_trace_names:
            trace_path = os.path.join(self.fcc_trace_dir, trace_name)
            
            if not os.path.exists(trace_path):
                print(f"⚠️  Warning: Trace not found: {trace_path}")
                continue
            
            # Load trace data
            trace_data = []
            with open(trace_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        time = float(parts[0])  # Time (ms)
                        throughput = float(parts[1])  # Throughput (Mbps)
                        trace_data.append([time, throughput])
            
            self.traces[trace_name] = np.array(trace_data)
            
        print(f"✅ Loaded {len(self.traces)} FCC traces into memory")
    
    def get_trace(self, mode: str = 'train') -> np.ndarray:
        """
        Get a random trace from specified mode
        
        Args:
            mode: 'train', 'val', or 'test'
        
        Returns:
            trace_data: Array of [time, throughput] pairs
        """
        if mode == 'train':
            trace_list = self.train_traces
        elif mode == 'val':
            trace_list = self.val_traces
        elif mode == 'test':
            trace_list = self.test_traces
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if not trace_list:
            raise ValueError(f"No traces available for mode: {mode}")
        
        # Random trace
        trace_name = np.random.choice(trace_list)
        
        if trace_name not in self.traces:
            raise ValueError(f"Trace not loaded: {trace_name}")
        
        return self.traces[trace_name].copy()
    
    def get_trace_by_name(self, trace_name: str) -> np.ndarray:
        """Get specific trace by name"""
        if trace_name not in self.traces:
            raise ValueError(f"Trace not found: {trace_name}")
        return self.traces[trace_name].copy()