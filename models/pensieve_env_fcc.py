# models/pensieve_env_fcc.py

import numpy as np
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.pensieve_reward import PensieveReward

class PensieveEnvFCC(ContentAwareEnvFCC):
    """
    Overrides ContentAwareEnvFCC to use the Pensieve BITRATE-based reward.
    It still returns the SAME state dictionary, to remain compatible
    with the PPO trainer.
    """
    
    def __init__(self, fcc_trace_loader, features_file, vmaf_file, video_dir, mode='train', **kwargs):
        
        print("🏗️  Initializing PensieveEnvFCC (Bitrate-based Reward)...")
        
        # Call parent constructor
        super().__init__(
            fcc_trace_loader=fcc_trace_loader,
            features_file=features_file,
            vmaf_file=vmaf_file,
            video_dir=video_dir,
            mode=mode,
            **kwargs
        )
        
        # !! OVERRIDE the reward function !!
        # ما شیء تابع پاداش را که در والد (V2) ساخته شده است، بازنویسی می‌کنیم
        # تا از تابع پاداش اصلی Pensieve (مبتنی بر بیت‌ریت) استفاده کند
        self.reward_function = PensieveReward(
            rebuffer_penalty=4.3, 
            smoothness_penalty=1.0
        )
        print("✅ PensieveEnvFCC: Reward function overridden to BITRATE-based.")
        
    def step(self, action):
        """
        This method is an EXACT COPY of the step() method from
        ContentAwareEnvV2, with ONLY the reward calculation line changed.
        """
        
        # =================================================================
        # ===> شروع: کپی کامل از content_aware_env_v2.py - step() <===
        # =================================================================
        
        # 1. Get chunk size based on action
        chunk_size_kbps = self.video_data['bitrates_kbps'][action]
        chunk_duration_s = self.video_data['chunk_duration_s']
        chunk_size_bytes = chunk_size_kbps * chunk_duration_s * 1000 / 8
        
        # 2. Simulate download time
        # (This is a simplified simulation; a real player is more complex)
        download_time_s = 0.0
        
        if self.use_real_traces:
            download_time_s = self.current_trace.get_download_time(
                chunk_size_bytes, 
                self.trace_time
            )
            self.trace_time += download_time_s
        else:
            # Synthetic traces
            throughput_kbps = self.current_trace.get_throughput(self.trace_time)
            if throughput_kbps <= 0:
                download_time_s = 5.0  # Failsafe
            else:
                download_time_s = (chunk_size_bytes * 8 / 1000) / throughput_kbps
            
            self.trace_time += download_time_s
        
        # 3. Calculate rebuffering time
        rebuffer_time_s = 0.0
        if self.buffer < download_time_s:
            rebuffer_time_s = download_time_s - self.buffer
            self.buffer = 0.0
        else:
            self.buffer -= download_time_s
            
        # 4. Update buffer
        self.buffer += chunk_duration_s
        
        # 5. Get VMAF score for the chunk (needed for info dict)
        vmaf_score = self.get_vmaf(self.video_id, self.chunk_idx, action)

        # 6. Get last bitrate (for smoothness penalty)
        last_bitrate_kbps = self.past_bitrates[-1] if self.chunk_idx > 0 else 0

        # ----------------------------------------------------
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # !!           تغییر اصلی اینجاست           !!
        #
        # کد اصلی در content_aware_env_v2.py این بود:
        # reward = self.reward_function.compute_reward_vmaf(
        #     vmaf_score=vmaf_score,
        #     rebuffer_time=rebuffer_time_s,
        #     last_bitrate=last_bitrate_kbps,
        #     current_bitrate=chunk_size_kbps
        # )
        
        # ما آن را با پاداش مبتنی بر بیت‌ریت Pensieve جایگزین می‌کنیم:
        reward = self.reward_function.compute_reward(
            bitrate=chunk_size_kbps,
            rebuffer_time=rebuffer_time_s,
            last_bitrate=last_bitrate_kbps
        )
        
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # ----------------------------------------------------

        # 7. Update history
        self.past_bitrates.append(chunk_size_kbps)
        self.past_download_time.append(download_time_s)
        
        try:
            throughput_estimate = (chunk_size_bytes * 8 / 1000) / download_time_s
            self.past_throughput.append(throughput_estimate)
        except ZeroDivisionError:
            self.past_throughput.append(0)
            
        # 8. Check if done
        self.chunk_idx += 1
        done = (self.chunk_idx == self.video_data['total_chunks'])
        
        if done:
            self.trace_time = 0.0 # Reset trace time for next episode
            
        # 9. Prepare info dictionary
        info = {
            'bitrate': chunk_size_kbps,
            'rebuffer_time': rebuffer_time_s,
            'download_time': download_time_s,
            'vmaf': vmaf_score,
            'last_bitrate': last_bitrate_kbps,
            'buffer': self.buffer,
            'throughput': self.past_throughput[-1]
        }
        
        # 10. Get next state
        next_state = self.get_state()
        
        # ===============================================================
        # ===> پایان: کپی کامل از content_aware_env_v2.py - step() <===
        # ===============================================================

        return next_state, reward, done, info