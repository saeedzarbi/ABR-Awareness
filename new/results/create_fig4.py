import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Add new/ to path (configs and src live there)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from configs.paths import get_paths

PATHS = get_paths()

print("🧠 Loading Proposed Model for Policy Visualization...")
model_path = PATHS['models'] / 'ppo_proposed_v2_fresh' / 'best_model' / 'best_model.zip'
if not model_path.exists():
    model_path = PATHS['models'] / 'ppo_proposed_v2_fresh' / 'final_model.zip'

model = PPO.load(str(model_path))

# تعریف محدوده‌ها برای شبکه و بافر
throughputs = np.linspace(500, 10000, 50)  # از 500 kbps تا 10 Mbps
buffers = np.linspace(0, 30, 50)           # از 0 تا 30 ثانیه

policy_map = np.zeros((len(buffers), len(throughputs)))

# مقادیر نرمال‌سازی بر اساس محیط شما
MIN_TP = 10.0
MAX_TP = 20000.0
MAX_BUF = 30.0

# مقادیر ثابت برای سایر ویژگی‌ها
typical_vmaf = np.array([0.35, 0.58, 0.74, 0.84, 0.91, 0.97])
typical_sizes = np.array([300, 750, 1200, 1850, 2850, 6000]) * 1000 * 4.0 / 30000000.0

for i, buf in enumerate(buffers):
    for j, tp in enumerate(throughputs):
        # ساخت Observation فرضی برای مدل
        log_tp = np.log(tp / MIN_TP) / np.log(MAX_TP / MIN_TP)
        tp_history = [log_tp] * 12
        buf_obs = buf / MAX_BUF
        buf_trend = 0.0  # فرض می‌کنیم بافر ثابت است
        last_br_obs = 2.0 / 5.0  # فرض می‌کنیم کیفیت قبلی متوسط بوده
        content_obs = [0.5, 0.5] # SI و TI متوسط
        
        obs = np.concatenate([
            tp_history, [buf_obs], [buf_trend], [last_br_obs],
            content_obs, typical_vmaf, typical_sizes
        ]).astype(np.float32)
        
        action, _ = model.predict(obs, deterministic=True)
        policy_map[i, j] = action

# رسم Heatmap
plt.figure(figsize=(9, 7))
sns.set_theme(style="white")

# چون اکشن‌ها 0 تا 5 هستند، نقشه رنگی گسسته می‌سازیم
cmap = sns.color_palette("RdYlGn", 6)
ax = sns.heatmap(policy_map, cmap=cmap, cbar_kws={'ticks': [0, 1, 2, 3, 4, 5]}, 
                 xticklabels=np.round(throughputs/1000, 1), 
                 yticklabels=np.round(buffers, 1))

# تنظیم برچسب‌های محورها برای زیبایی
ax.set_xticks(np.arange(0, 50, 5))
ax.set_xticklabels(np.round(throughputs[::5]/1000, 1))
ax.set_yticks(np.arange(0, 50, 5))
ax.set_yticklabels(np.round(buffers[::5], 1))
ax.invert_yaxis()

plt.title('Learned Bitrate Selection Policy ($B_{ref} = 15s$)', fontsize=14, pad=15)
plt.xlabel('Estimated Throughput (Mbps)', fontsize=12)
plt.ylabel('Current Buffer Level (seconds)', fontsize=12)

# تنظیم نام کیفیت‌ها در نوار رنگی
cbar = ax.collections[0].colorbar
cbar.set_ticklabels(['300 kbps', '750 kbps', '1200 kbps', '1850 kbps', '2850 kbps', '6000 kbps'])
cbar.set_label('Selected Bitrate', rotation=270, labelpad=20)

plt.tight_layout()
script_dir = Path(__file__).resolve().parent
plt.savefig(script_dir / 'fig_8_policy_heatmap.png', dpi=300)
print("✅ Saved: fig_8_policy_heatmap.png")