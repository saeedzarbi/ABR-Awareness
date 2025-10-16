# models/mpc_model.py

import numpy as np

class MPC:
    """
    Model Predictive Control (MPC) Agent.
    این یک الگوریتم است، نه یک شبکه عصبی.
    """
    
    def __init__(self, future_chunks=5, bitrate_levels=None, smoothness_penalty=1.0, rebuffer_penalty=4.3):
        print(f"🧠 Initializing MPC Agent (lookahead = {future_chunks} chunks)")
        self.future_chunks = future_chunks
        self.bitrate_levels = bitrate_levels if bitrate_levels is not None else [300, 750, 1850, 2850, 4300, 6000]
        self.smoothness_penalty = smoothness_penalty
        self.rebuffer_penalty = rebuffer_penalty
        self.M_IN_K = 1000.0

    def predict_throughput(self, past_throughputs):
        """
        پیش‌بینی سرعت شبکه برای آینده.
        ساده‌ترین روش: استفاده از میانگین هارمونیک (Harmonic Mean).
        """
        if not past_throughputs:
            return 1000.0  # یک مقدار پیش‌فرض اولیه
        
        # حذف مقادیر صفر
        past_throughputs = [t for t in past_throughputs if t > 0]
        if not past_throughputs:
            return 1000.0

        # محاسبه میانگین هارمونیک
        harmonic_mean = len(past_throughputs) / np.sum(1.0 / np.array(past_throughputs))
        return harmonic_mean

    def select_action(self, state):
        """
        انتخاب بهترین بیت‌ریت با نگاه به آینده.
        
        Args:
            state (dict): دیکشنری وضعیت که از محیط می‌آید.
        """
        past_throughputs = [t * 6000.0 for t in state['network'][0, :] if t > 0] # بازگرداندن به kbps
        current_buffer = state['network'][2, -1] * 60.0  # بازگرداندن به ثانیه
        
        last_bitrate = 0
        past_bitrates_normalized = [b for b in state['network'][3, :] if b > 0]
        if past_bitrates_normalized:
            last_bitrate = past_bitrates_normalized[-1] * 6000.0

        # پیش‌بینی سرعت شبکه برای N چانک آینده
        predicted_throughput = self.predict_throughput(past_throughputs)
        
        # جستجوی کامل (Brute-force search) برای یافتن بهترین مسیر
        best_action_sequence = None
        max_reward = float('-inf')

        # تمام ترکیب‌های ممکن از بیت‌ریت‌ها را امتحان می‌کنیم
        from itertools import product
        all_action_combos = product(range(len(self.bitrate_levels)), repeat=self.future_chunks)

        for action_sequence in all_action_combos:
            temp_buffer = current_buffer
            temp_last_bitrate = last_bitrate
            current_reward = 0
            is_feasible = True

            for action in action_sequence:
                bitrate_kbps = self.bitrate_levels[action]
                chunk_size_kbit = bitrate_kbps * 4.0  # فرض می‌کنیم هر چانک ۴ ثانیه است

                # زمان دانلود تخمینی
                download_time = chunk_size_kbit / predicted_throughput if predicted_throughput > 0 else 4.0
                
                # محاسبه rebuffer
                rebuffer = max(0, download_time - temp_buffer)
                
                # محاسبه پاداش برای این قدم
                quality_reward = bitrate_kbps / self.M_IN_K
                smoothness_penalty_val = self.smoothness_penalty * abs(bitrate_kbps - temp_last_bitrate) / self.M_IN_K
                
                reward = quality_reward - self.rebuffer_penalty * rebuffer - smoothness_penalty_val
                current_reward += reward

                # به‌روزرسانی بافر و بیت‌ریت برای قدم بعدی شبیه‌سازی
                temp_buffer = max(0, temp_buffer - download_time) + 4.0
                temp_last_bitrate = bitrate_kbps

            # اگر این مسیر بهترین مسیر تا الان بود، آن را ذخیره کن
            if current_reward > max_reward:
                max_reward = current_reward
                best_action_sequence = action_sequence
        
        # فقط اولین تصمیم از بهترین مسیر را برمی‌گردانیم
        return best_action_sequence[0] if best_action_sequence else 0