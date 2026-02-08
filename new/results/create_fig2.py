import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# PATH TO YOUR CHUNK LOG
# ==========================================
# مسیری که قبلا گفتید فایل در آن است
CHUNKS_LOG_PATH = 'new/results/logs/evaluation_v22/Proposed_crowd_run_chunks.csv'

# تنظیمات گرافیکی
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 300})

def plot_bitrate_distribution():
    # بررسی وجود فایل
    if not os.path.exists(CHUNKS_LOG_PATH):
        # تلاش برای پیدا کردن فایل در مسیر جاری اگر مسیر بالا غلط بود
        if os.path.exists('Proposed_crowd_run_chunks.csv'):
            path = 'Proposed_crowd_run_chunks.csv'
        else:
            print(f"❌ File not found: {CHUNKS_LOG_PATH}")
            return
    else:
        path = CHUNKS_LOG_PATH

    try:
        df = pd.read_csv(path)
        
        # پیدا کردن ستون بیت‌ریت (با حروف بزرگ یا کوچک)
        col_name = None
        for col in ['bitrate', 'Bitrate', 'quality', 'Quality']:
            if col in df.columns:
                col_name = col
                break
        
        if col_name:
            plt.figure(figsize=(7, 5))
            
            # رسم نمودار میله‌ای
            # بیت‌ریت‌ها را به کیلوبیت بر ثانیه نمایش می‌دهیم
            ax = sns.countplot(x=col_name, data=df, palette="Blues_d")
            
            plt.xlabel('Selected Bitrate (kbps)', fontsize=12)
            plt.ylabel('Frequency (Count)', fontsize=12)
            plt.title('Agent Action Distribution Strategy', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            
            # اضافه کردن درصد روی هر میله
            total = len(df)
            for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2 - 0.1
                y = p.get_height() + 10 # کمی بالاتر از میله
                ax.annotate(percentage, (x, y), size=10, weight='bold', color='black')

            plt.savefig('fig_bitrate_dist.png')
            print("✅ fig_bitrate_dist.png created successfully!")
        else:
            print("❌ Column 'bitrate' not found. Available columns:", df.columns)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    plot_bitrate_distribution()