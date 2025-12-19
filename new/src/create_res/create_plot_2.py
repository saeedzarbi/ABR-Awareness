import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# تنظیمات گرافیکی استاندارد
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})
sns.set_style("whitegrid")

def generate_extra_plots():
    print("🚀 Generating Additional Analysis Plots...")

    # ==========================================
    # 1. Training Convergence Plot
    # ==========================================
    print("\n📈 Generating Training Convergence Plot...")
    try:
        df_train = pd.read_csv('episodes_detailed.csv')
        
        # محاسبه میانگین متحرک (Moving Average) برای صاف کردن نمودار
        window_size = 100
        df_train['Reward_MA'] = df_train['avg_reward'].rolling(window=window_size).mean()
        df_train['VMAF_MA'] = df_train['avg_vmaf'].rolling(window=window_size).mean()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # رسم پاداش (Reward)
        color = 'tab:blue'
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Average Reward', color=color)
        ax1.plot(df_train['step'], df_train['Reward_MA'], color=color, linewidth=2, label='Avg Reward (MA)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # رسم VMAF روی محور دوم
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Average VMAF', color=color)
        ax2.plot(df_train['step'], df_train['VMAF_MA'], color=color, linewidth=2, linestyle='--', label='Avg VMAF (MA)')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Training Convergence: Reward & VMAF Improvement over Time', fontweight='bold')
        fig.tight_layout()
        plt.savefig('fig_training_convergence.png', dpi=300)
        print("✅ Saved: fig_training_convergence.png")
        
    except FileNotFoundError:
        print("⚠️ File episodes_detailed.csv not found.")

    # ==========================================
    # 2. Box Plot of QoE (Variance Analysis)
    # ==========================================
    print("\n📦 Generating QoE Box Plot...")
    try:
        df_stats = pd.read_csv('detailed_stats_multi_video_22.csv')
        
        # فیلتر متدها
        methods = ['BBA', 'Genie', 'RobustMPC', 'Pensieve', 'Proposed']
        df_plot = df_stats[df_stats['Method'].isin(methods)].copy()
        
        # مرتب‌سازی
        df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=methods, ordered=True)
        
        plt.figure(figsize=(12, 6))
        
        # پالت رنگ
        colors = sns.color_palette("muted", len(methods))
        palette = {m: c for m, c in zip(methods, colors)}
        palette['Proposed'] = '#d62728'
        
        sns.boxplot(x='Method', y='QoE', data=df_plot, palette=palette, showfliers=False) # showfliers=False برای حذف نقاط پرت شدید
        sns.stripplot(x='Method', y='QoE', data=df_plot, color='black', alpha=0.3, jitter=0.2, size=3) # نمایش نقاط واقعی
        
        plt.title('QoE Distribution and Variance Comparison', fontweight='bold')
        plt.ylabel('QoE Score')
        plt.xlabel('Method')
        
        plt.savefig('fig_qoe_boxplot.png', dpi=300)
        print("✅ Saved: fig_qoe_boxplot.png")
        
    except FileNotFoundError:
        print("⚠️ File detailed_stats_multi_video_22.csv not found.")

    # ==========================================
    # 3. Buffer Level CDF (Safety Analysis)
    # ==========================================
    print("\n🛡️ Generating Buffer Level CDF...")
    try:
        # بارگذاری همه فایل‌های چانک مربوط به مدل پیشنهادی
        videos = ['bigbuckbunny', 'crowd_run', 'sintel', 'tearsofsteel_short']
        buffer_data = []
        
        for vid in videos:
            try:
                df = pd.read_csv(f'Proposed_{vid}_chunks.csv')
                buffer_data.extend(df['buffer'].tolist())
            except:
                pass
        
        if buffer_data:
            buffer_data = np.sort(buffer_data)
            yvals = np.arange(len(buffer_data)) / float(len(buffer_data) - 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(buffer_data, yvals, color='#2ecc71', linewidth=3, label='Proposed Method')
            
            # خطوط راهنما
            plt.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Danger Zone (<5s)')
            plt.axvline(x=15, color='gray', linestyle=':', alpha=0.5, label='Target Buffer (15s)')
            
            plt.title('CDF of Buffer Levels (Safety Analysis)', fontweight='bold')
            plt.xlabel('Buffer Level (seconds)')
            plt.ylabel('Cumulative Probability')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.savefig('fig_buffer_cdf.png', dpi=300)
            print("✅ Saved: fig_buffer_cdf.png")
        else:
            print("⚠️ No chunk data found for buffer CDF.")
            
    except Exception as e:
        print(f"⚠️ Error in Buffer CDF: {e}")

if __name__ == "__main__":
    generate_extra_plots()