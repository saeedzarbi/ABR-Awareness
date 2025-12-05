import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# تنظیمات استایل IEEE
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.5), # اندازه استاندارد تک‌ستونی IEEE
    'lines.linewidth': 1.5,
    'savefig.dpi': 300 # کیفیت بالا برای چاپ
})

def plot_cdf_qoe():
    try:
        # بررسی وجود فایل
        if not os.path.exists('new/results/detailed_stats.csv'):
            print("Error: 'new/results/detailed_stats.csv' not found.")
            return

        df = pd.read_csv('new/results/detailed_stats.csv')
        
        # تعریف رنگ‌ها و استایل‌ها
        methods = {
            'Proposed': {'label': 'Proposed (Lyapunov)', 'color': '#d62728', 'style': '-'}, # قرمز
            'Pensieve': {'label': 'Pensieve', 'color': '#1f77b4', 'style': '--'}, # آبی
            'RobustMPC': {'label': 'RobustMPC', 'color': '#2ca02c', 'style': '-.'}, # سبز
            'BBA': {'label': 'BBA', 'color': '#7f7f7f', 'style': ':'} # خاکستری
        }
        
        plt.figure()
        for m_key, m_props in methods.items():
            subset = df[df['Method'] == m_key]['QoE']
            if not subset.empty:
                sorted_data = np.sort(subset)
                yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
                plt.plot(sorted_data, yvals, label=m_props['label'], color=m_props['color'], linestyle=m_props['style'])

        plt.xlabel('Quality of Experience (QoE)')
        plt.ylabel('CDF')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout(pad=0.2)
        plt.savefig('cdf_comparison.png')
        print("✅ نمودار CDF (cdf_comparison.png) با موفقیت تولید شد.")
    except Exception as e:
        print(f"❌ خطا در تولید CDF: {e}")

def plot_bar_metrics():
    try:
        # بررسی وجود فایل
        if not os.path.exists('new/results/tcsvt_generalization_results.csv'):
            print("Error: 'new/results/tcsvt_generalization_results.csv' not found.")
            return

        df = pd.read_csv('new/results/tcsvt_generalization_results.csv')
        
        # فیلتر کردن روش‌های مورد نظر
        target_methods = ['Proposed (Lyapunov)', 'Pensieve*', 'RobustMPC', 'BBA']
        df = df[df['Method'].isin(target_methods)].copy()
        
        # تمیزکاری نام‌ها
        df['Method'] = df['Method'].replace({'Pensieve*': 'Pensieve', 'Proposed (Lyapunov)': 'Proposed'})
        
        # ترتیب‌دهی برای نمایش بهتر
        order = ['Pensieve', 'BBA', 'RobustMPC', 'Proposed']
        df = df.set_index('Method').reindex(order).reset_index()

        fig, ax1 = plt.subplots()
        x = np.arange(len(df['Method']))
        width = 0.35

        # محور چپ: VMAF
        rects1 = ax1.bar(x - width/2, df['Avg VMAF'], width, label='Avg VMAF', color='#4c72b0', alpha=0.9)
        ax1.set_ylabel('Avg VMAF', color='black')
        ax1.set_ylim(0, 100)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Method'], rotation=15) # کمی چرخش برای خوانایی بهتر
        
        # محور راست: Rebuffering
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, df['Rebuffering Ratio (%)'], width, label='Rebuf %', color='#c44e52', hatch='///', alpha=0.7)
        ax2.set_ylabel('Rebuffer Ratio (%)', color='black')
        ax2.set_ylim(0, 25) # کمی بالاتر از ماکزیمم (16 درصد)

        # لجند ترکیبی
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=8)

        plt.tight_layout(pad=0.2)
        plt.savefig('metrics_comparison.png')
        print("✅ نمودار متریک‌ها (metrics_comparison.png) با موفقیت تولید شد.")
    except Exception as e:
        print(f"❌ خطا در تولید نمودار میله‌ای: {e}")

if __name__ == "__main__":
    if os.path.exists('new/results'):
        print("شروع تولید نمودارها...")
        plot_cdf_qoe()
        plot_bar_metrics()
    else:
        print("❌ پوشه 'new/results' پیدا نشد. لطفاً فایل‌های نتایج را آپلود کنید.")