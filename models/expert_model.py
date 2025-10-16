# models/expert_model.py

import numpy as np

class ExpertAgent:
    """
    یک "متخصص" ساده برای تولید داده‌های آموزشی برای Comyco.
    این متخصص سعی می‌کند بر اساس بافر و سرعت شبکه، تصمیم بهینه بگیرد.
    """
    def __init__(self, bitrate_levels=None):
        print("🧠 Initializing Simple Expert Agent for Imitation Learning")
        self.bitrate_levels = bitrate_levels if bitrate_levels is not None else [300, 750, 1850, 2850, 4300, 6000]

    def select_action(self, state):
        """
        یک منطق ساده برای انتخاب اکشن بهینه.
        - اگر بافر کم است -> کیفیت را کم کن.
        - اگر بافر زیاد است و سرعت شبکه خوب است -> کیفیت را زیاد کن.
        """
        buffer_size = state['network'][2, -1] * 60.0  # بازگرداندن به ثانیه
        
        # دریافت آخرین سرعت شبکه
        past_throughputs = [t * 6000.0 for t in state['network'][0, :] if t > 0]
        last_throughput = past_throughputs[-1] if past_throughputs else 1000.0

        # منطق تصمیم‌گیری متخصص
        if buffer_size < 5.0:
            # بافر بسیار کم است، ریسک نکن
            return 0  # پایین‌ترین کیفیت
        elif buffer_size < 15.0:
            # بافر کم است، محافظه‌کار باش
            # کیفیتی را انتخاب کن که سرعت شبکه از پس آن بربیاید
            for i in reversed(range(len(self.bitrate_levels))):
                if self.bitrate_levels[i] < last_throughput * 0.8:
                    return i
            return 0
        else:
            # بافر زیاد است، بهترین کیفیت ممکن را انتخاب کن
            for i in reversed(range(len(self.bitrate_levels))):
                if self.bitrate_levels[i] < last_throughput * 1.2:
                    return i
            return 0