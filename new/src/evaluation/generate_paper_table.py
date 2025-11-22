import pandas as pd
from scipy import stats
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.paths import get_paths
PATHS = get_paths()

def main():
    csv_path = PATHS['results'] / 'detailed_stats.csv'
    if not csv_path.exists():
        print("❌ Please run final_comparison.py first.")
        return

    df = pd.read_csv(csv_path)
    methods = df['Method'].unique()
    proposed_method = 'Proposed (Lyapunov)' # نام دقیق مدل خودتان در CSV
    
    if proposed_method not in methods:
        # پیدا کردن نام نزدیک
        proposed_method = [m for m in methods if 'Proposed' in m][0]

    print(f"📊 Calculating T-Tests against {proposed_method}...\n")
    
    print("-" * 80)
    print(f"{'Method':<15} | {'QoE (Mean ± Std)':<20} | {'p-value':<10} | {'Result'}")
    print("-" * 80)
    
    proposed_scores = df[df['Method'] == proposed_method]['QoE']
    
    for m in methods:
        if m == proposed_method:
            continue
            
        other_scores = df[df['Method'] == m]['QoE']
        
        # T-Test Independent
        t_stat, p_val = stats.ttest_ind(proposed_scores, other_scores, equal_var=False)
        
        mean = other_scores.mean()
        std = other_scores.std()
        
        sig = "Not Sig"
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        
        print(f"{m:<15} | {mean:.2f} ± {std:.2f}   | {p_val:.1e}  | {sig}")

    print("-" * 80)
    print("*: p<0.05, **: p<0.01, ***: p<0.001 (Significant difference)")

if __name__ == "__main__":
    main()