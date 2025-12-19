import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# تنظیمات گرافیکی
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
sns.set_style("whitegrid")

def plot_action_analysis():
    print("🔬 در حال تحلیل رفتار دقیق ایجنت (Action Analysis)...")
    
    # لیست فایل‌های لاگ چانک (که در مرحله ارزیابی تولید شده‌اند)
    # مطمئن شوید این فایل‌ها در پوشه هستند
    files = {
        'CrowdRun (Hard)': 'Proposed_crowd_run_chunks.csv',
        'Sintel (Easy)': 'Proposed_sintel_chunks.csv',
        'BigBuckBunny': 'Proposed_bigbuckbunny_chunks.csv'
    }
    
    # 1. نمودار توزیع بیت‌ریت (Bitrate Distribution)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, fname) in enumerate(files.items()):
        if not os.path.exists(fname):
            print(f"⚠️ فایل {fname} یافت نشد. رد کردن...")
            continue
            
        df = pd.read_csv(fname)
        
        # محاسبه درصد انتخاب هر بیت‌ریت
        bitrate_counts = df['bitrate'].value_counts(normalize=True).sort_index() * 100
        
        sns.barplot(x=bitrate_counts.index, y=bitrate_counts.values, ax=axes[i], palette="viridis")
        axes[i].set_title(f'Bitrate Choices: {name}', fontweight='bold')
        axes[i].set_xlabel('Bitrate (kbps)')
        axes[i].set_ylabel('Selection Frequency (%)')
        axes[i].set_ylim(0, 100)
        
        # اضافه کردن مقادیر روی ستون‌ها
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_action_distribution.png', dpi=300)
    print("✅ نمودار توزیع اکشن ذخیره شد: fig_action_distribution.png")

    # 2. تحلیل زمانی (Behavior over Time) - برای CrowdRun
    if os.path.exists('Proposed_crowd_run_chunks.csv'):
        df = pd.read_csv('Proposed_crowd_run_chunks.csv')
        # فیلتر کردن برای یک اپیزود نمونه (مثلا اپیزود 0)
        sample_episode = df[df['episode'] == 0]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # محور سمت چپ: سرعت شبکه و بیت‌ریت انتخابی
        ax1.set_xlabel('Chunk Index')
        ax1.set_ylabel('Bitrate / Throughput (kbps)', color='tab:blue')
        ax1.plot(sample_episode['chunk'], sample_episode['throughput'], color='gray', alpha=0.5, linestyle='--', label='Network Throughput')
        ax1.plot(sample_episode['chunk'], sample_episode['bitrate'], color='tab:blue', linewidth=2, marker='o', label='Chosen Bitrate')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        
        # محور سمت راست: سطح بافر
        ax2 = ax1.twinx()
        ax2.set_ylabel('Buffer Level (sec)', color='tab:green')
        ax2.fill_between(sample_episode['chunk'], sample_episode['buffer'], color='tab:green', alpha=0.2, label='Buffer Level')
        ax2.plot(sample_episode['chunk'], sample_episode['buffer'], color='tab:green', linestyle='-', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='tab:green')
        
        # خط بافر امن
        ax2.axhline(y=15, color='red', linestyle=':', alpha=0.5, label='Target Buffer')
        
        plt.title('Agent Behavior in Hard Scenario (CrowdRun)', fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig_agent_behavior_time.png', dpi=300)
        print("✅ نمودار رفتار زمانی ذخیره شد: fig_agent_behavior_time.png")

if __name__ == "__main__":
    plot_action_analysis()