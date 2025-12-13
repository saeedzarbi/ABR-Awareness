import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# تنظیم استایل نمودارها برای کیفیت مقاله
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

def plot_comparative_analysis(csv_file='/root/new/ABR-Awareness/new/results/detailed_stats_multi_video_22.csv'):
    # ۱. بارگذاری داده‌ها
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return

    # لیست روش‌هایی که می‌خواهیم مقایسه کنیم
    target_methods = ['Proposed', 'Pensieve', 'RobustMPC', 'BBA', 'Genie']
    df = df[df['Method'].isin(target_methods)]

    # ترتیب نمایش در نمودار
    order = ['Proposed', 'Pensieve', 'RobustMPC', 'BBA', 'Genie']
    
    # رنگ‌بندی (Proposed را متمایز می‌کنیم)
    palette = {
        'Proposed': '#d62728',  # قرمز برای روش پیشنهادی
        'Pensieve': '#2ca02c',  # سبز
        'RobustMPC': '#ff7f0e', # نارنجی
        'BBA': '#1f77b4',       # آبی
        'Genie': '#9467bd'      # بنفش
    }

    # ==========================================
    # نمودار ۱: مقایسه میانگین شاخص‌ها (Bar Plots)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # الف) QoE
    sns.barplot(x='Method', y='QoE', data=df, ax=axes[0, 0], order=order, palette=palette, capsize=0.1, errorbar='sd')
    axes[0, 0].set_title('Average QoE (Higher is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('Mean QoE')
    axes[0, 0].set_xlabel('')

    # ب) VMAF (کیفیت تصویر)
    sns.barplot(x='Method', y='VMAF', data=df, ax=axes[0, 1], order=order, palette=palette, capsize=0.1, errorbar='sd')
    axes[0, 1].set_title('Video Quality (VMAF) (Higher is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('VMAF Score')
    axes[0, 1].set_xlabel('')

    # ج) Rebuffering (بافرینگ)
    sns.barplot(x='Method', y='Rebuffer', data=df, ax=axes[1, 0], order=order, palette=palette, capsize=0.1, errorbar='sd')
    axes[1, 0].set_title('Rebuffering Ratio (Lower is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('Rebuffer (%)')
    axes[1, 0].set_xlabel('')

    # د) Switching (تغییر کیفیت)
    sns.barplot(x='Method', y='Switch', data=df, ax=axes[1, 1], order=order, palette=palette, capsize=0.1, errorbar='sd')
    axes[1, 1].set_title('Switch Rate (Lower is Better)', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Switches')
    axes[1, 1].set_xlabel('')

    plt.tight_layout()
    plt.savefig('comparative_metrics_bar.png', dpi=300, bbox_inches='tight')
    print("✅ نمودار میله‌ای ذخیره شد: comparative_metrics_bar.png")
    plt.show()

    # ==========================================
    # نمودار ۲: تابع توزیع تجمعی QoE (CDF Plot)
    # ==========================================
    plt.figure(figsize=(10, 7))
    
    linestyles = {'Proposed': '-', 'Pensieve': '--', 'RobustMPC': '-.', 'BBA': ':', 'Genie': (0, (3, 1, 1, 1))}
    linewidths = {'Proposed': 3, 'Pensieve': 2, 'RobustMPC': 2, 'BBA': 2, 'Genie': 2}

    for method in order:
        if method not in df['Method'].unique(): continue
        
        # مرتب‌سازی داده‌ها برای رسم CDF
        method_data = df[df['Method'] == method]['QoE'].sort_values()
        y_vals = np.arange(len(method_data)) / float(len(method_data) - 1)
        
        plt.plot(method_data, y_vals, label=method, 
                 color=palette[method], 
                 linestyle=linestyles.get(method, '-'),
                 linewidth=linewidths.get(method, 2))

    plt.title('CDF of Quality of Experience (QoE)', fontweight='bold')
    plt.xlabel('QoE Score')
    plt.ylabel('Cumulative Probability')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    
    plt.savefig('comparative_qoe_cdf.png', dpi=300, bbox_inches='tight')
    print("✅ نمودار CDF ذخیره شد: comparative_qoe_cdf.png")
    plt.show()

# اجرای تابع
plot_comparative_analysis()