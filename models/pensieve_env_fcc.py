# models/pensieve_env_fcc.py

import numpy as np
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.pensieve_reward import PensieveReward

class PensieveEnvFCC(ContentAwareEnvFCC):
    """
    این کلاس از محیط FCC شما ارث می‌برد اما تابع پاداش را
    به پاداش مبتنی بر بیت‌ریت (اصلی Pensieve) تغییر می‌دهد.
    """
    
    def __init__(self, fcc_trace_loader, features_file, vmaf_file, video_dir, mode='train', **kwargs):
        
        print("🏗️  Initializing PensieveEnvFCC (Bitrate-based Reward)...")
        
        # 1. سازنده والد (ContentAwareEnvFCC) را فراخوانی می‌کنیم.
        super().__init__(
            fcc_trace_loader=fcc_trace_loader,
            features_file=features_file,
            vmaf_file=vmaf_file,
            video_dir=video_dir,
            mode=mode,
            **kwargs
        )
        
        # 2. !! بازنویسی (Override) تابع پاداش !!
        self.reward_function = PensieveReward(
            rebuffer_penalty=4.3, 
            smoothness_penalty=1.0,
            bitrate_levels=self.bitrate_levels
        )
        print("✅ PensieveEnvFCC: Reward function overridden to BITRATE-based.")

    def compute_reward(self, action, rebuffer_time):
        """
        !! این متد، متد 'compute_reward' در content_aware_env_v2.py را بازنویسی می‌کند !!
        
        متد 'step' والد (که ما به آن دست نزدیم) این متد را فراخوانی خواهد کرد.
        """
        
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        # =======================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # !!           تغییر اصلی و رفع خطا اینجاست           !!
        #
        # بر اساس خطای شما، تابع پاداش انتظار 'action' را دارد نه 'bitrate'.
        # پس ما 'action' را مستقیماً پاس می‌دهیم.
        #
        # توجه: این کد فرض می‌کند که فایل pensieve_reward.py شما
        # متد compute_reward(self, action, rebuffer_time, last_bitrate) را دارد.
        
        reward = self.reward_function.compute_reward(
            action,          # <--- پاس دادن اندیس (0-5)
            rebuffer_time,
            last_bitrate
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # =======================================================
        
        return float(reward)