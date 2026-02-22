import pandas as pd

# خواندن فایل
df = pd.read_csv('detailed_stats_multi_video_final.csv')

# فیلتر کردن فقط برای Fugu
fugu_df = df[df['Method'] == 'Fugu (Sim)']

print("--- Fugu Stats ---")
print(f"Mean VMAF: {fugu_df['VMAF'].mean():.2f}")
print(f"Mean Rebuffer: {fugu_df['Rebuffer'].mean():.2f}")
print(f"Mean Switch: {fugu_df['Switch'].mean():.2f}")