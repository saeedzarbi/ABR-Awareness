import pandas as pd
import numpy as np
from pathlib import Path

# فایل CSV در همان پوشهٔ اسکریپت
csv_path = Path(__file__).parent / 'detailed_stats_multi_video_final.csv' 

try:
    df = pd.read_csv(csv_path)
    
    # اگر اسم ستون روش‌ها 'Method'، 'method' یا 'Agent' است، آن را پیدا می‌کنیم
    if 'Method' in df.columns:
        col_method = 'Method'
    elif 'method' in df.columns:
        col_method = 'method'
    elif 'Agent' in df.columns:
        col_method = 'Agent'
    else:
        raise ValueError(f"ستون روش یافت نشد. ستون‌های موجود: {list(df.columns)}")
    col_qoe = 'QoE' if 'QoE' in df.columns else 'qoe'
    
    methods = df[col_method].unique()
    
    print("=== Tail-Risk Analysis (QoE in worst-case scenarios) ===")
    
    for m in methods:
        qoe_data = df[df[col_method] == m][col_qoe].values
        
        if len(qoe_data) == 0:
            continue
            
        mean_qoe = np.mean(qoe_data)
        p5 = np.percentile(qoe_data, 5)  # 5 درصد بدترین تجربه‌ها
        
        # محاسبه CVaR 10% (میانگین 10 درصد پایین‌ترین QoEها)
        threshold = np.percentile(qoe_data, 10)
        cvar_10 = qoe_data[qoe_data <= threshold].mean()
        
        print(f"[{m}]")
        print(f"  -> Average QoE:      {mean_qoe:.2f}")
        print(f"  -> 5th Percentile:   {p5:.2f}  (Worst 5% guarantee)")
        print(f"  -> CVaR (10%):       {cvar_10:.2f}  (Avg of worst 10%)")
        print("-" * 50)
        
except Exception as e:
    print(f"خطا در خواندن فایل یا ستون‌ها: {e}")