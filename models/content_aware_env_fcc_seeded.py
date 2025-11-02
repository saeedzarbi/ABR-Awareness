
# import numpy as np
# from models.content_aware_env_v2 import ContentAwareEnvV2

# class ContentAwareEnvFCC(ContentAwareEnvV2):
#     """
#     Modified environment for FCC traces
#     Inherits from ContentAwareEnvV2 but uses FCCTraceLoader
#     """

#     def __init__(self, 
#                  fcc_trace_loader,
#                  features_file: str,
#                  vmaf_file: str,
#                  video_dir: str,
#                  mode: str = 'train',
#                  **kwargs):
#         self.fcc_trace_loader = fcc_trace_loader
#         self.mode = mode

#         super().__init__(
#             trace_dir='data/network_traces/cooked_traces',  # dummy, won't be used
#             features_file=features_file,
#             vmaf_file=vmaf_file,
#             use_real_traces=True,
#             **kwargs
#         )
#         self.use_real_traces = True

#     def seed(self, seed: int):
#         """Set random seed for reproducibility"""
#         import random
#         random.seed(seed)
#         np.random.seed(seed)
#         if hasattr(self.fcc_trace_loader, "seed"):
#             self.fcc_trace_loader.seed(seed)

#     def reset(self, video_id=1, split=None):
#         if split is None:
#             split = self.mode

#         trace_data = self.fcc_trace_loader.get_trace(mode=split)

#         from models.trace_loader import NetworkTrace
#         self.current_trace = NetworkTrace(
#             trace_id=f"fcc_{split}",
#             timestamps=trace_data[:, 0],
#             throughputs=trace_data[:, 1],
#             metadata={'source': 'fcc'}
#         )
#         self.trace_time = 0.0

#         self.video_id = video_id
#         self.chunk_idx = 0
#         self.buffer = 0.0
#         self.past_throughput = []
#         self.past_download_time = []
#         self.past_bitrates = []
#         self.past_errors = []

#         return self.get_state()


# if __name__ == '__main__':
#     print("ContentAwareEnvFCC with seed support loaded successfully!")
import numpy as np
from models.content_aware_env_v2 import ContentAwareEnvV2  # ✅ نسخه‌ی جدید fixed

class ContentAwareEnvFCC(ContentAwareEnvV2):
    """
    Environment برای FCC traces (نسخه هماهنگ با پاداش composite)
    از ContentAwareEnvV2 ارث‌بری می‌کند.
    """

    def __init__(
        self,
        fcc_trace_loader,
        features_file: str,
        vmaf_file: str,
        video_dir: str,
        mode: str = 'train',
        **kwargs
    ):
        """
        پارامترها:
            fcc_trace_loader : نمونه از FCCTraceLoader
            features_file : مسیر فایل ویژگی‌های محتوایی (SI/TI)
            vmaf_file : مسیر فایل جدول VMAF
            video_dir : مسیر ویدئوها (در این نسخه استفاده نمی‌شود)
            mode : train / val / test
        """
        self.fcc_trace_loader = fcc_trace_loader
        self.mode = mode

        # 🔹 اضافه شد: فعال‌سازی پاداش composite برای سازگاری کامل
        super().__init__(
            trace_dir='data/network_traces/cooked_traces',  # dummy
            features_file=features_file,
            vmaf_file=vmaf_file,
            use_real_traces=True,
            use_composite_reward=True,  # ✅ پاداش composite فعال شد
            **kwargs
        )

        self.use_real_traces = True

    def seed(self, seed: int):
        """Set random seed for reproducibility"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self.fcc_trace_loader, "seed"):
            self.fcc_trace_loader.seed(seed)

    def reset(self, video_id=1, split=None):
        """
        Reset محیط با trace واقعی از FCC dataset
        """
        if split is None:
            split = self.mode

        trace_data = self.fcc_trace_loader.get_trace(mode=split)

        from models.trace_loader import NetworkTrace
        self.current_trace = NetworkTrace(
            trace_id=f"fcc_{split}",
            timestamps=trace_data[:, 0],
            throughputs=trace_data[:, 1],
            metadata={'source': 'fcc'}
        )

        self.trace_time = 0.0
        self.video_id = video_id
        self.chunk_idx = 0
        self.buffer = 0.0
        self.past_throughput = []
        self.past_download_time = []
        self.past_bitrates = []
        self.past_errors = []

        return self.get_state()


if __name__ == '__main__':
    print("✓ ContentAwareEnvFCC (Composite Reward Enabled) ready to use.")
