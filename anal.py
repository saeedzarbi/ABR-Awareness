import pandas as pd
from scipy.stats import ttest_rel
from tabulate import tabulate

# فایل‌های نتایج اپیزود به صورت reward per episode
paths = {
    "Full Model": "results/ablation_logs/full.log",
    "No Content": "results/ablation_logs/no_content.log",
    "No VMAF": "results/ablation_logs/no_vmaf.log",
    "Pensieve-Like": "results/ablation_logs/pensieve_like.log"
}

# بارگذاری داده‌ها
rewards = {name: pd.read_csv(path)["reward"] for name, path in paths.items()}

# آزمون‌های آماری زوجی نسبت به مدل کامل
comparisons = [("Full Model", "No Content"),
               ("Full Model", "No VMAF"),
               ("Full Model", "Pensieve-Like")]

results = []
for a, b in comparisons:
    t, p = ttest_rel(rewards[a], rewards[b])
    results.append({
        "Comparison": f"{a} vs {b}",
        "t-statistic": f"{t:.2f}",
        "p-value": f"{p:.4f}",
        "Significant (p<0.05)": "Yes" if p < 0.05 else "No"
    })

# چاپ نتایج به‌صورت جدول
print("\n📊 Paired t-test Results:")
print(tabulate(results, headers="keys", tablefmt="github"))

# استخراج خلاصه reward میانگین برای جدول مقاله
summary_table = pd.DataFrame({k: [v.mean()] for k, v in rewards.items()})
print("\n📄 Mean Reward Table:")
print(tabulate(summary_table.T.reset_index().rename(columns={0: "Mean Reward", "index": "Model"}),
               headers="keys", tablefmt="github"))
