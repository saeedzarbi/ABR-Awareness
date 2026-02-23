import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 0. Load Data
# ==========================================
# Make sure these two files are in the same folder as this script
file_std = 'detailed_stats_new_final.csv'
file_5g = 'detailed_stats_new_final5g.csv'

if not os.path.exists(file_std) or not os.path.exists(file_5g):
    print("❌ Error: CSV files not found. Please ensure they are in the current directory.")
    exit()

df_std = pd.read_csv(file_std)
df_5g = pd.read_csv(file_5g)

# Style configuration for IEEE format
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 15})

print("📊 Generating Figures...")

# ==========================================
# 1. Ablation Study Plot
# ==========================================
summary_std = df_std.groupby('Method')['Rebuffer'].mean().reset_index()
ablation_methods = ['Ablation_Base', 'Ablation_Future', 'Ablation_Lyap', 'Proposed']
df_abl = summary_std[summary_std['Method'].isin(ablation_methods)].copy()

rename_abl = {
    'Ablation_Base': 'Base PPO',
    'Ablation_Future': 'PPO + Future',
    'Ablation_Lyap': 'PPO + Lyap',
    'Proposed': 'Proposed (Ours)'
}
df_abl['Method'] = df_abl['Method'].map(rename_abl)
df_abl['Method'] = pd.Categorical(df_abl['Method'], categories=['Base PPO', 'PPO + Future', 'PPO + Lyap', 'Proposed (Ours)'], ordered=True)
df_abl = df_abl.sort_values('Method')

plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Method', y='Rebuffer', data=df_abl, palette=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
plt.title('Ablation Study: Impact of Components on Rebuffering')
plt.ylabel('Mean Rebuffering Rate (%)')
plt.xlabel('Algorithm Variants')
plt.ylim(0, max(df_abl['Rebuffer']) * 1.2)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontweight='bold', fontsize=11, xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('fig_1_ablation.png', dpi=300)
plt.close()
print(" ✅ Saved: fig_1_ablation.png")

# ==========================================
# Data Prep for 5G vs Standard (Zero-Shot)
# ==========================================
main_methods = ['Fugu', 'BBA', 'RobustMPC', 'Pensieve', 'Proposed']
rename_main = {'Proposed': 'Proposed (Ours)'}

df_std_main = df_std[df_std['Method'].isin(main_methods)].copy()
df_std_main['Method'] = df_std_main['Method'].replace(rename_main)
df_std_main['Network'] = 'Standard (Broadband/4G)'

df_5g_main = df_5g[df_5g['Method'].isin(main_methods)].copy()
df_5g_main['Method'] = df_5g_main['Method'].replace(rename_main)
df_5g_main['Network'] = '5G mmWave'

df_combined = pd.concat([df_std_main, df_5g_main])
method_order = ['Fugu', 'BBA', 'RobustMPC', 'Pensieve', 'Proposed (Ours)']
df_combined['Method'] = pd.Categorical(df_combined['Method'], categories=method_order, ordered=True)

# ==========================================
# 2. Rebuffering: Standard vs 5G
# ==========================================
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Method', y='Rebuffer', hue='Network', data=df_combined, 
                 palette={'Standard (Broadband/4G)': '#3498db', '5G mmWave': '#e74c3c'}, errorbar=None)
plt.title('Zero-Shot Generalization: Rebuffering Collapse in 5G Networks')
plt.ylabel('Mean Rebuffering (s / %)')
plt.xlabel('Algorithm')
plt.ylim(0, 150) 

for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontweight='bold', fontsize=10, xytext=(0, 3), textcoords='offset points')

plt.legend(title='Network Type', loc='upper left')
plt.tight_layout()
plt.savefig('fig_2_rebuffering_5g_vs_std.png', dpi=300)
plt.close()
print(" ✅ Saved: fig_2_rebuffering_5g_vs_std.png")

# ==========================================
# 3. QoE: Standard vs 5G
# ==========================================
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Method', y='QoE', hue='Network', data=df_combined, 
                 palette={'Standard (Broadband/4G)': '#2ecc71', '5G mmWave': '#9b59b6'}, errorbar=None)
plt.title('Zero-Shot Generalization: Overall QoE Maintenance in 5G')
plt.ylabel('Mean QoE Score')
plt.xlabel('Algorithm')
plt.ylim(0, 4200)

for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontweight='bold', fontsize=10, xytext=(0, 3), textcoords='offset points')

plt.legend(title='Network Type', loc='lower right')
plt.tight_layout()
plt.savefig('fig_3_qoe_5g_vs_std.png', dpi=300)
plt.close()
print(" ✅ Saved: fig_3_qoe_5g_vs_std.png")

# ==========================================
# 4. CDF of QoE (Standard Dataset)
# ==========================================
plt.figure(figsize=(8, 6))
colors = ['#95a5a6', '#f39c12', '#e74c3c', '#3498db', '#2ecc71']
for idx, method in enumerate(method_order):
    data = df_std_main[df_std_main['Method'] == method]['QoE']
    sns.ecdfplot(data, label=method, color=colors[idx], linewidth=2.5)

plt.title('Cumulative Distribution Function (CDF) of QoE')
plt.xlabel('QoE Score')
plt.ylabel('CDF')
plt.legend(title='Method', loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fig_4_qoe_cdf.png', dpi=300)
plt.close()
print(" ✅ Saved: fig_4_qoe_cdf.png")

print("🎉 All 4 journal-quality figures have been successfully generated!")