import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# تنظیم استایل
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# تولید داده‌های فرضی برای شبکه استاندارد (Broadband/4G)
# نوسانات ملایم بین 1.5 تا 5 مگابیت
time_std = np.arange(0, 100, 1)
np.random.seed(42)
base_std = 3.0 + np.sin(time_std / 5.0) * 1.5
noise_std = np.random.normal(0, 0.5, len(time_std))
trace_std = np.clip(base_std + noise_std, 0.5, 6.0)

# تولید داده‌های فرضی برای شبکه 5G mmWave
# سرعت بسیار بالا (10 تا 20 مگابیت) اما با افت‌های ناگهانی به نزدیک صفر (Blockage)
time_5g = np.arange(0, 100, 1)
base_5g = 15.0 + np.sin(time_5g / 3.0) * 4.0
noise_5g = np.random.normal(0, 2.0, len(time_5g))
trace_5g = np.clip(base_5g + noise_5g, 5.0, 25.0)

# اعمال Blockage های خشن 5G (موانع فیزیکی)
trace_5g[20:25] = np.random.uniform(0.1, 0.5, 5) # افت شدید اول
trace_5g[55:65] = np.random.uniform(0.1, 0.5, 10) # افت شدید و طولانی دوم
trace_5g[85:88] = np.random.uniform(0.1, 0.5, 3) # افت شدید سوم

# رسم نمودارها
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# پلات شبکه استاندارد
ax1.plot(time_std, trace_std, color='#3498db', linewidth=2, label='Standard Trace (FCC/HSDPA)')
ax1.fill_between(time_std, trace_std, color='#3498db', alpha=0.2)
ax1.set_title('Comparison of Network Dynamics: Standard vs. 5G mmWave', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Throughput (Mbps)', fontweight='bold')
ax1.set_ylim(0, 8)
ax1.legend(loc='upper right')

# پلات شبکه 5G
ax2.plot(time_5g, trace_5g, color='#e74c3c', linewidth=2, label='5G mmWave Trace (with Blockages)')
ax2.fill_between(time_5g, trace_5g, color='#e74c3c', alpha=0.2)
ax2.set_xlabel('Time (seconds)', fontweight='bold')
ax2.set_ylabel('Throughput (Mbps)', fontweight='bold')
ax2.set_ylim(0, 25)

# هایلایت کردن نقاط افت در 5G
ax2.axvspan(20, 25, color='black', alpha=0.1, label='Signal Blockage / Starvation')
ax2.axvspan(55, 65, color='black', alpha=0.1)
ax2.axvspan(85, 88, color='black', alpha=0.1)
ax2.legend(loc='upper right')

plt.tight_layout()
script_dir = Path(__file__).resolve().parent
plt.savefig(script_dir / 'fig_9_network_traces.png', dpi=300)
print("✅ Saved: fig_9_network_traces.png")