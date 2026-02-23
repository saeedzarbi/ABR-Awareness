import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (use script directory so it works from any cwd)
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'detailed_stats_new_final5g.csv'))
main_methods = ['Fugu', 'BBA', 'RobustMPC', 'Pensieve', 'Proposed']
df_main = df[df['Method'].isin(main_methods)].copy()
df_main['Method'] = df_main['Method'].replace({'Proposed': 'Proposed (Ours)'})

order = ['Fugu', 'BBA', 'RobustMPC', 'Pensieve', 'Proposed (Ours)']
df_main['Method'] = pd.Categorical(df_main['Method'], categories=order, ordered=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ==========================================
# 1. Stability Boxplot (Number of Switches)
# ==========================================
plt.figure(figsize=(8, 5))
sns.boxplot(x='Method', y='Switch', data=df_main, 
            palette=['#95a5a6', '#f39c12', '#e74c3c', '#3498db', '#2ecc71'],
            showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
plt.title('Streaming Stability: Number of Bitrate Switches per Episode')
plt.ylabel('Number of Switches')
plt.xlabel('Algorithm')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig_5_stability_boxplot.png'), dpi=300)
plt.close()
print("✅ Saved: fig_5_stability_boxplot.png")

# ==========================================
# 2. Trade-off Scatter Plot (QoE vs Rebuffer)
# ==========================================
plt.figure(figsize=(8, 6))
summary = df_main.groupby('Method').agg({'Rebuffer': 'mean', 'QoE': 'mean'}).reset_index()

sns.scatterplot(x='Rebuffer', y='QoE', hue='Method', s=300, data=summary, 
                palette=['#95a5a6', '#f39c12', '#e74c3c', '#3498db', '#2ecc71'], marker='X')

plt.title('Performance Trade-off: Mean QoE vs. Rebuffering Rate')
plt.ylabel('Mean QoE Score')
plt.xlabel('Mean Rebuffering Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Method')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig_6_tradeoff.png'), dpi=300)
plt.close()
print("✅ Saved: fig_6_tradeoff.png")