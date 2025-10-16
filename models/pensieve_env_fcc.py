# models/pensieve_env_fcc.py

import numpy as np
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.pensieve_reward import PensieveReward

class PensieveEnvFCC(ContentAwareEnvFCC):
    """
    این کلاس از محیط FCC شما (ContentAwareEnvFCC) ارث می‌برد
    اما تابع پاداش را به پاداش مبتنی بر بیت‌ریت (اصلی Pensieve) تغییر می‌دهد.
    
    ما متد 'step' را بازنویسی نمی‌کنیم، بلکه متد 'compute_reward' را
    که 'step' والد (content_aware_env_v2.py) فراخوانی می‌کند، بازنویسی می‌کنیم.
    """
    
    def __init__(self, fcc_trace_loader, features_file, vmaf_file, video_dir, mode='train', **kwargs):
        
        print("🏗️  Initializing PensieveEnvFCC (Bitrate-based Reward)...")
        
        # 1. سازنده والد (ContentAwareEnvFCC) را فراخوانی می‌کنیم.
        # این کار تمام trace ها و فایل‌ها را لود می‌کند و self.reward_function
        # را بر اساس کد والد (V2) روی حالت VMAF-based تنظیم می‌کند.
        super().__init__(
            fcc_trace_loader=fcc_trace_loader,
            features_file=features_file,
            vmaf_file=vmaf_file,
            video_dir=video_dir,
            mode=mode,
            **kwargs
        )
        
        # 2. !! بازنویسی (Override) تابع پاداش !!
        # حالا ما self.reward_function را با نمونه صحیح (bitrate-based)
        # جایگزین می‌کنیم تا پاداش اصلی Pensieve محاسبه شود.
        self.reward_function = PensieveReward(
            rebuffer_penalty=4.3, 
            smoothness_penalty=1.0,
            bitrate_levels=self.bitrate_levels # self.bitrate_levels از والد می‌آید
        )
        print("✅ PensieveEnvFCC: Reward function overridden to BITRATE-based.")

    def compute_reward(self, action, rebuffer_time):
        """
        !! این متد، متد 'compute_reward' در content_aware_env_v2.py را بازنویسی می‌کند !!
        
        متد 'step' والد (که ما به آن دست نزدیم) این متد را فراخوانی خواهد کرد.
        ما در اینجا پاداش را مبتنی بر بیت‌ریت (و نه VMAF) محاسبه می‌کنیم.
        """
        
        # دریافت بیت‌ریت‌های فعلی و قبلی (بر اساس کیلوبیت بر ثانیه)
        # این متغیرها در والد (self) موجود هستند
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        # محاسبه پاداش Pensieve با استفاده از compute_reward (مبتنی بر بیت‌ریت)
        # self.reward_function ما اکنون به نمونه صحیح (bitrate-based) اشاره دارد
        reward = self.reward_function.compute_reward(
            bitrate=current_bitrate,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate
        )
        
        return float(reward)