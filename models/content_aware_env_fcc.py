# models/content_aware_env_fcc.py

import numpy as np
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.fcc_trace_loader import FCCTraceLoader

class ContentAwareEnvFCC(ContentAwareEnvV2):
    """
    Modified environment for FCC traces
    Inherits from ContentAwareEnvV2 but uses FCCTraceLoader
    """
    
    def __init__(self, 
                 fcc_trace_loader: FCCTraceLoader,
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
        
        # Call parent constructor
        # Note: We pass a dummy trace_loader to parent
        # We'll override the trace loading method
        super().__init__(
            trace_loader=None,  # We'll handle this ourselves
            features_file=features_file,
            vmaf_file=vmaf_file,
            video_dir=video_dir,
            mode=mode,
            **kwargs
        )
    
    def _load_trace(self):
        """Override parent's trace loading to use FCC traces"""
        self.trace = self.fcc_trace_loader.get_trace(mode=self.mode)
        self.trace_idx = 0
        
    def reset(self):
        """Reset environment with new FCC trace"""
        # Load new trace
        self._load_trace()
        
        # Call parent reset
        return super().reset()