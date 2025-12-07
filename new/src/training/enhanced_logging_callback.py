"""
Enhanced Logging Callback for ABR Training
==========================================
این callback اطلاعات دقیق از training جمع می‌کنه برای تحلیل بعدی
"""

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import json

class EnhancedLoggingCallback(BaseCallback):
    """
    Callback برای لاگ کردن اطلاعات دقیق در هنگام training
    """
    
    def __init__(self, log_dir: str, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        
        # فایل‌های لاگ
        self.action_log_path = self.log_dir / "actions_detailed.csv"
        self.episode_log_path = self.log_dir / "episodes_detailed.csv"
        self.reward_log_path = self.log_dir / "rewards_breakdown.csv"
        
        # ایجاد فایل‌ها
        self._init_log_files()
        
        # Cache برای episode
        self.episode_cache = {
            'actions': [],
            'rewards': [],
            'buffers': [],
            'rebuffers': [],
            'vmaf': []
        }
        
    def _init_log_files(self):
        """ایجاد فایل‌های لاگ با header"""
        
        # Action log
        with open(self.action_log_path, 'w') as f:
            f.write("step,mean_action,action_distribution,action_variance\n")
        
        # Episode log
        with open(self.episode_log_path, 'w') as f:
            f.write("step,episode,avg_reward,avg_buffer,rebuffer_rate,avg_vmaf,action_entropy\n")
        
        # Reward breakdown
        with open(self.reward_log_path, 'w') as f:
            f.write("step,avg_total_reward,avg_vmaf_component,avg_rebuffer_penalty,avg_smooth_penalty\n")
    
    def _on_step(self) -> bool:
        """هر step صدا زده می‌شه"""
        
        # Log actions periodically
        if self.n_calls % self.log_freq == 0:
            self._log_actions()
            
        # Collect episode data if available
        if 'infos' in self.locals:
            self._collect_episode_data()
            
        return True
    
    def _log_actions(self):
        """لاگ کردن action distribution"""
        actions = self.locals.get('actions', [])
        
        if len(actions) > 0:
            mean_action = np.mean(actions)
            action_dist = np.bincount(actions, minlength=6) / len(actions)
            action_var = np.var(actions)
            
            with open(self.action_log_path, 'a') as f:
                f.write(f"{self.num_timesteps},{mean_action:.2f},"
                       f"\"{list(action_dist)}\",{action_var:.2f}\n")
            
            # TensorBoard
            self.logger.record("actions/mean", mean_action)
            self.logger.record("actions/variance", action_var)
            for i, prob in enumerate(action_dist):
                self.logger.record(f"actions/bitrate_{i}", prob)
    
    def _collect_episode_data(self):
        """جمع‌آوری داده‌های episode"""
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                # Episode تمام شده
                ep_info = info['episode']
                
                # محاسبه آمار
                avg_reward = ep_info['r']
                ep_length = ep_info['l']
                
                # اگر اطلاعات اضافی داریم
                if 'avg_quality' in info:
                    avg_vmaf = info['avg_quality']
                    total_rebuffer = info.get('total_rebuffer', 0)
                    rebuffer_rate = (total_rebuffer / (ep_length * 4.0)) * 100
                    
                    # لاگ کردن
                    with open(self.episode_log_path, 'a') as f:
                        f.write(f"{self.num_timesteps},{self.n_calls},"
                               f"{avg_reward:.2f},0.0,{rebuffer_rate:.2f},"
                               f"{avg_vmaf:.2f},0.0\n")
                    
                    # TensorBoard
                    self.logger.record("episode/avg_vmaf", avg_vmaf)
                    self.logger.record("episode/rebuffer_rate", rebuffer_rate)
                    self.logger.record("episode/avg_reward", avg_reward)


class EvaluationLogger:
    """
    Logger برای evaluation - داده‌های دقیق‌تر
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_logs = []
        self.episode_logs = []
    
    def log_chunk(self, env, action, info):
        """لاگ کردن هر chunk در evaluation"""
        chunk_data = {
            'episode': len(self.episode_logs),
            'chunk': env.chunk_idx - 1,
            'video': env.current_video_name,
            'action': action,
            'bitrate': env.BITRATE_LEVELS[action],
            'buffer': info['buffer_level'],
            'throughput': info['throughput'],
            'rebuffer': info.get('rebuffer', 0),
            'vmaf': env.last_vmaf,
            'reward': info.get('reward', 0)
        }
        self.chunk_logs.append(chunk_data)
    
    def log_episode(self, env, total_reward, switches):
        """لاگ کردن episode کامل"""
        episode_data = {
            'episode': len(self.episode_logs),
            'video': env.current_video_name,
            'trace_idx': env.current_trace_idx,
            'total_reward': total_reward,
            'avg_vmaf': env.total_quality / env.chunk_idx,
            'total_rebuffer': env.total_rebuffer,
            'rebuffer_rate': (env.total_rebuffer / (env.chunk_idx * 4)) * 100,
            'total_smooth': env.total_smooth,
            'switches': switches,
            'chunks': env.chunk_idx
        }
        self.episode_logs.append(episode_data)
    
    def save_logs(self, prefix='eval'):
        """ذخیره تمام لاگ‌ها"""
        
        # Chunk-level logs
        if self.chunk_logs:
            df_chunks = pd.DataFrame(self.chunk_logs)
            df_chunks.to_csv(self.log_dir / f'{prefix}_chunks.csv', index=False)
            print(f"✅ Saved chunk logs: {len(self.chunk_logs)} chunks")
        
        # Episode-level logs
        if self.episode_logs:
            df_episodes = pd.DataFrame(self.episode_logs)
            df_episodes.to_csv(self.log_dir / f'{prefix}_episodes.csv', index=False)
            print(f"✅ Saved episode logs: {len(self.episode_logs)} episodes")
            
            # خلاصه آمار
            print("\n📊 Episode Statistics:")
            print(df_episodes.describe())
    
    def analyze_logs(self):
        """تحلیل سریع لاگ‌ها"""
        if not self.episode_logs:
            return
        
        df = pd.DataFrame(self.episode_logs)
        
        print("\n" + "="*60)
        print("📈 تحلیل عملکرد:")
        print("="*60)
        
        # Per-video stats
        if 'video' in df.columns:
            print("\n🎬 عملکرد به تفکیک ویدیو:")
            video_stats = df.groupby('video').agg({
                'avg_vmaf': 'mean',
                'rebuffer_rate': 'mean',
                'total_reward': 'mean',
                'switches': 'mean'
            }).round(2)
            print(video_stats)
        
        # Action distribution from chunks
        if self.chunk_logs:
            df_chunks = pd.DataFrame(self.chunk_logs)
            print("\n🎬 Action Distribution:")
            action_counts = df_chunks['action'].value_counts().sort_index()
            for action, count in action_counts.items():
                pct = (count / len(df_chunks)) * 100
                print(f"   Bitrate {action}: {pct:5.1f}%")
        
        # Rebuffer analysis
        print(f"\n⏸️ Rebuffer Analysis:")
        print(f"   Mean: {df['rebuffer_rate'].mean():.2f}%")
        print(f"   Std:  {df['rebuffer_rate'].std():.2f}%")
        print(f"   Max:  {df['rebuffer_rate'].max():.2f}%")
        
        # Episodes with high rebuffer
        high_rebuf = df[df['rebuffer_rate'] > 5.0]
        if len(high_rebuf) > 0:
            print(f"\n⚠️ Episodes with rebuffer > 5%: {len(high_rebuf)}")
            print(high_rebuf[['episode', 'video', 'rebuffer_rate', 'avg_vmaf']])


# ===================================================================
# نحوه استفاده در کد
# ===================================================================

"""
# در train_ppo_multi_dynamic.py:

from enhanced_logging_callback import EnhancedLoggingCallback

# اضافه کردن به callbacks:
callbacks = CallbackList([
    CheckpointCallback(...),
    EvalCallback(...),
    ActionLogCallback(...),
    EnhancedLoggingCallback(log_dir=log_dir, log_freq=5000)  # ← جدید
])


# در final_multi.py (برای evaluation):

from enhanced_logging_callback import EvaluationLogger

# قبل از evaluation loop:
eval_logger = EvaluationLogger(log_dir=PATHS['logs'] / 'evaluation')

# در evaluation loop:
for ep in range(episodes_per_video):
    obs, info = env.reset()
    done = False
    total_reward = 0
    switches = 0
    last_br = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        # لاگ chunk
        eval_logger.log_chunk(env, action, info)
        
        total_reward += reward
        if action != last_br:
            switches += 1
        last_br = action
    
    # لاگ episode
    eval_logger.log_episode(env, total_reward, switches)

# در پایان:
eval_logger.save_logs(prefix=f'{method}_eval')
eval_logger.analyze_logs()
"""
