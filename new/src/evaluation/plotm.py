import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# مسیر فایل نتایج صحیح
RESULTS_FILE = Path("/home/saeedzarbi95/test/ABR-Awareness/new/results/tcsvt_generalization_results.csv")
OUTPUT_DIR = Path("/home/saeedzarbi95/test/ABR-Awareness/new/results/paper_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not RESULTS_FILE.exists():
        print(f"❌ File not found: {RESULTS_FILE}")
        return

    df = pd.read_csv(RESULTS_FILE)
    print("📊 Data loaded:")
    print(df)

    # تنظیم استایل نمودارها برای چاپ در مقاله
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    # رنگ‌ها: پیشنهادی (آبی پررنگ)، بیس‌لاین (خاکستری/نارنجی)
    colors = ["#2ecc71", "#95a5a6"] # سبز برای مدل ما، خاکستری برای پنسیو

    # 1. نمودار مقایسه VMAF (مهمترین نمودار)
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x='Method', y='Avg VMAF', data=df, palette=colors)
    
    # افزودن مقدار عددی روی ستون‌ها
    for i, v in enumerate(df['Avg VMAF']):
        ax.text(i, v + 1, f"{v:.1f}", ha='center', fontweight='bold')
        
    plt.title("Visual Quality Comparison (VMAF)", fontweight='bold')
    plt.ylabel("Average VMAF Score (0-100)")
    plt.xlabel("")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig1_VMAF_Comparison.pdf")
    plt.savefig(OUTPUT_DIR / "Fig1_VMAF_Comparison.png", dpi=300)
    print("✓ Figure 1 saved.")

    # 2. نمودار بهره‌وری (Quality Improvement)
    # محاسبه درصد بهبود
    baseline_vmaf = df[df['Method'] == 'Pensieve*']['Avg VMAF'].values[0]
    proposed_vmaf = df[df['Method'] == 'Proposed (Lyapunov)']['Avg VMAF'].values[0]
    improvement = ((proposed_vmaf - baseline_vmaf) / baseline_vmaf) * 100
    
    print(f"\n🚀 IMPROVEMENT: +{improvement:.1f}% over Pensieve")

    # 3. نمودار QoE استاندارد
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x='Method', y='Standard QoE', data=df, palette=colors)
    plt.title("Quality of Experience (QoE)", fontweight='bold')
    plt.ylabel("Total Reward")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig2_QoE_Comparison.pdf")
    plt.savefig(OUTPUT_DIR / "Fig2_QoE_Comparison.png", dpi=300)
    print("✓ Figure 2 saved.")

    # 4. نمودار رفتار سوئیچینگ
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x='Method', y='Switch Freq', data=df, palette=colors)
    plt.title("Bitrate Switching Frequency", fontweight='bold')
    plt.ylabel("Number of Switches")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig3_Switch_Comparison.pdf")
    print("✓ Figure 3 saved.")

if __name__ == "__main__":
    main()