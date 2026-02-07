import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# CONFIGURATION & FILE PATHS
# ==========================================
DETAILED_RESULTS_PATH = 'detailed_stats_multi_video_final.csv'
ABLATION_RESULTS_PATH = 'ablation_results.csv'
TIME_SERIES_LOG_PATH = 'logs/evaluation_v22/Proposed_crowd_run_chunks.csv'

sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

COLORS = sns.color_palette("deep")

def load_data():
    """Load all necessary datasets gracefully."""
    data = {}
    try:
        data['main'] = pd.read_csv(DETAILED_RESULTS_PATH)
        print(f"✅ Loaded Main Results: {len(data['main'])} rows")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DETAILED_RESULTS_PATH}")
    
    try:
        data['ablation'] = pd.read_csv(ABLATION_RESULTS_PATH)
        print(f"✅ Loaded Ablation Results")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {ABLATION_RESULTS_PATH}")

    try:
        data['time_series'] = pd.read_csv(TIME_SERIES_LOG_PATH)
        print(f"✅ Loaded Time Series Log")
    except FileNotFoundError:
        print(f"❌ Warning: Could not find {TIME_SERIES_LOG_PATH} (Fig 2 will be skipped)")
        
    return data

# ==========================================
# FIGURE 1: CDF of QoE
# ==========================================
def plot_cdf_qoe(df):
    if df is None: return
    plt.figure(figsize=(6, 4))
    
    methods = df['Method'].unique()
    methods = sorted(methods, key=lambda x: 1 if 'Proposed' in x else 0)

    for method in methods:
        qoe_scores = df[df['Method'] == method]['QoE'].dropna().sort_values()
        y_vals = np.arange(1, len(qoe_scores) + 1) / len(qoe_scores)
        
        if 'Proposed' in method:
            plt.plot(qoe_scores, y_vals, label=method, linewidth=2.5, color='red', linestyle='-')
        else:
            plt.plot(qoe_scores, y_vals, label=method, linewidth=1.5, linestyle='--')

    plt.xlabel('Average QoE')
    plt.ylabel('CDF')
    plt.title('CDF of Average QoE (Performance Consistency)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_cdf_qoe.png')
    print("Output: fig_cdf_qoe.png")

# ==========================================
# FIGURE 2: Time Series Behavior
# ==========================================
def plot_time_series(df):
    if df is None: return
    # فقط اپیزود اول را برمی‌داریم
    df_ep = df[df['episode'] == 0].copy()
    
    # تبدیل ایندکس چانک به زمان (هر چانک ۴ ثانیه)
    df_ep['time'] = df_ep['chunk'] * 4.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    # نمودار بالا: بیت‌ریت و پهنای باند
    # فرض بر این است که بیت‌ریت در فایل لاگ به kbps است -> تبدیل به Mbps
    ax1.plot(df_ep['time'], df_ep['bitrate'] / 1000.0, color='#d62728', linewidth=2, label='Selected Bitrate')
    ax1.plot(df_ep['time'], df_ep['throughput'] / 1000.0, color='gray', linestyle='--', alpha=0.6, label='Network Throughput')
    ax1.set_ylabel('Rate (Mbps)')
    ax1.set_title('Agent Response to Network Fluctuations')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # نمودار پایین: بافر
    ax2.fill_between(df_ep['time'], df_ep['buffer'], color='#2ca02c', alpha=0.2)
    ax2.plot(df_ep['time'], df_ep['buffer'], color='#2ca02c', linewidth=2, label='Buffer Level')
    ax2.axhline(y=4.0, color='black', linestyle=':', label='Rebuf Threshold')
    ax2.set_ylabel('Buffer (sec)')
    ax2.set_xlabel('Time (seconds)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_time_series.png')
    print("Output: fig_time_series.png")

# ==========================================
# FIGURE 3: Stability Boxplot
# ==========================================
def plot_stability(df):
    if df is None: return
    plt.figure(figsize=(6, 4))
    
    # حذف داده‌های پرت (Outliers) برای تمیزی نمودار
    sns.boxplot(x='Method', y='Switch', data=df, showfliers=False, palette="Set2")
    
    plt.ylabel('Bitrate Switches per Session')
    plt.xlabel('')
    plt.title('Stability Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_stability_boxplot.png')
    print("Output: fig_stability_boxplot.png")

# ==========================================
# FIGURE 4: Ablation Study
# ==========================================
def plot_ablation(df):
    if df is None: return
    
    # نگاشت نام‌های فنی به نام‌های مقاله
    name_map = {
        'Baseline (V1)': 'Base PPO',
        'Conservative (V2)': '+ Stability',
        'No Buffer-Aware (V3)': '+ Future Info',
        'Full (V4)': 'Proposed (All)'
    }
    # اگر ستون variant دارید از آن استفاده کنید، وگرنه نام ستون را چک کنید
    col_name = 'variant' if 'variant' in df.columns else 'Configuration'
    if col_name in df.columns:
        df['clean_name'] = df[col_name].map(name_map).fillna(df[col_name])
    else:
        df['clean_name'] = df.iloc[:, 0] # فرض بر اینکه ستون اول نام است

    # مرتب‌سازی
    df = df.sort_values('quality_mean') # یا Avg. VMAF

    fig, ax1 = plt.subplots(figsize=(7, 5))

    x = np.arange(len(df))
    width = 0.35

    # میله‌های VMAF (محور چپ)
    ax1.bar(x - width/2, df['quality_mean'], width, label='VMAF', color='#1f77b4', alpha=0.8)
    ax1.set_ylabel('Average VMAF', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(bottom=40, top=100) 

    # میله‌های Rebuffering (محور راست)
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, df['rebuffer_mean'], width, label='Rebuffering', color='#d62728', alpha=0.8)
    ax2.set_ylabel('Rebuffering Ratio (%)', color='#d62728', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#d62728')

    ax1.set_xticks(x)
    ax1.set_xticklabels(df['clean_name'], rotation=15)
    ax1.set_title('Ablation Study: Feature Impact')

    # Legend ترکیبی
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('fig_ablation.png')
    print("Output: fig_ablation.png")

# ==========================================
# FIGURE 5: Trade-off Scatter Plot
# ==========================================
def plot_tradeoff(df):
    if df is None: return
    plt.figure(figsize=(6, 5))
    
    # محاسبه میانگین برای هر متد
    summary = df.groupby('Method').agg({
        'Rebuffer': 'mean',
        'VMAF': 'mean'
    }).reset_index()

    ax = plt.gca()
    sns.scatterplot(x='Rebuffer', y='VMAF', hue='Method', style='Method', s=200, data=summary, palette="bright", ax=ax)
    
    # تاکید روی روش Proposed در گوشهٔ بالا-چپ: حلقهٔ بیرونی و برچسب واضح‌تر
    proposed = summary[summary['Method'].str.contains('Proposed', case=False, na=False)]
    if not proposed.empty:
        xp, yp = proposed['Rebuffer'].values[0], proposed['VMAF'].values[0]
        ax.scatter([xp], [yp], s=400, facecolors='none', edgecolors='red', linewidths=2.5, zorder=5)
    
    # برچسب زدن روی نقاط
    for i, row in summary.iterrows():
        plt.text(row['Rebuffer']+0.5, row['VMAF']+0.5, row['Method'], fontsize=9, fontweight='bold')

    plt.xlabel('Rebuffering Impact (Lower is Better)')
    plt.ylabel('Visual Quality (VMAF) (Higher is Better)')
    plt.title('QoE Trade-off Analysis')

    # تنظیم محورها تا روش Proposed در گوشهٔ بالا-چپ (بهترین: Rebuffer کم، VMAF بالا) متمایز دیده شود
    x_min, x_max = summary['Rebuffer'].min(), summary['Rebuffer'].max()
    y_min, y_max = summary['VMAF'].min(), summary['VMAF'].max()
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 5.0)
    # فضای اضافه در سمت راست و پایین تا گوشهٔ بالا-چپ (نقطهٔ Proposed) برجسته شود
    plt.xlim(max(0, x_min - 0.05 * x_range), x_max + 0.25 * x_range)
    plt.ylim(y_min - 0.08 * y_range, min(100, y_max + 0.15 * y_range))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('fig_tradeoff.png')
    print("Output: fig_tradeoff.png")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Starting Plot Generation ---")
    data = load_data()
    
    if 'main' in data:
        plot_cdf_qoe(data['main'])
        plot_stability(data['main'])
        plot_tradeoff(data['main'])
    
    if 'ablation' in data:
        plot_ablation(data['ablation'])
        
    if 'time_series' in data:
        plot_time_series(data['time_series'])
        
    print("--- All plots generated successfully! ---")