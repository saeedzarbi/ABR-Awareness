# models/content_aware_env_fcc.py

import numpy as np
from models.content_aware_env_v2 import ContentAwareEnvV2

class ContentAwareEnvFCC(ContentAwareEnvV2):
    """
    Modified environment for FCC traces
    Inherits from ContentAwareEnvV2 but uses FCCTraceLoader
    """
    
    def __init__(self, 
                 fcc_trace_loader,
                 features_file: str,
                 vmaf_file: str,
                 video_dir: str,
                 mode: str = 'train',
                 **kwargs):
        """
        Args:
            fcc_trace_loader: FCCTraceLoader instance
            features_file: Path to video features JSON
            vmaf_file: Path to VMAF scores JSON
            video_dir: Path to video directory
            mode: 'train', 'val', or 'test'
        """
        # Store FCC loader
        self.fcc_trace_loader = fcc_trace_loader
        self.mode = mode
        
        # Call parent constructor WITHOUT trace_loader
        # We'll handle trace loading ourselves
        super().__init__(
            trace_dir='data/network_traces/cooked_traces',  # dummy, won't be used
            features_file=features_file,
            vmaf_file=vmaf_file,
            use_real_traces=True,  # Important!
            **kwargs
        )
        
        # Override trace_loader
        self.use_real_traces = True
        
    def reset(self, video_id=1, split=None):
        """Reset environment with new FCC trace"""
        # Use self.mode instead of split
        if split is None:
            split = self.mode
        
        # Load new FCC trace
        trace_data = self.fcc_trace_loader.get_trace(mode=split)
        
        # Create a simple trace object that has get_throughput method
        from models.trace_loader import NetworkTrace
        self.current_trace = NetworkTrace(
            trace_id=f"fcc_{split}",
            timestamps=trace_data[:, 0],
            throughputs=trace_data[:, 1],
            metadata={'source': 'fcc'}
        )
        self.trace_time = 0.0
        
        # Call parent reset (but skip its trace loading)
        self.video_id = video_id
        self.chunk_idx = 0
        self.buffer = 0.0
        
        # Network state history
        self.past_throughput = []
        self.past_download_time = []
        self.past_bitrates = []
        self.past_errors = []
        
        return self.get_state()


if __name__ == '__main__':
    print("ContentAwareEnvFCC module loaded successfully!")