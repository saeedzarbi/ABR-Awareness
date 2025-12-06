import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import urllib.request
import urllib.parse

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from src.environment.abr_multi_env import ABREnv
from configs.paths import get_paths

PATHS = get_paths()

# Slack webhook URL (from environment variable)
SLACK_WEBHOOK = os.getenv('SLACK_WEBHOOK_URL', '')

def send_slack_message(status, step, message):
    """Send message to Slack"""
    if not SLACK_WEBHOOK:
        return
    
    color = "good"
    emoji = "✅"
    if status == "error":
        color = "danger"
        emoji = "❌"
    elif status == "info":
        color = "#36a64f"
        emoji = "ℹ️"
    elif status == "warning":
        color = "warning"
        emoji = "⚠️"
    
    payload = {
        "attachments": [
            {
                "color": color,
                "title": f"{emoji} {step}",
                "text": message,
                "footer": "Penalty Tuning Pipeline",
                "ts": int(pd.Timestamp.now().timestamp())
            }
        ]
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            SLACK_WEBHOOK,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"⚠️ Failed to send Slack notification: {e}")

# --- تنظیمات جستجو ---
CANDIDATE_PENALTIES = [35.0, 45.0, 55.0, 65.0, 75.0]  # اعدادی که می‌خواهیم تست کنیم
TRAIN_STEPS_PER_TRIAL = 300_000           # آموزش کوتاه برای تست (حدود ۱۵ دقیقه)
TEST_VIDEO = 'crowd_run'                  # تست روی سخت‌ترین ویدیو

def make_env_with_penalty(penalty_value, rank=0):
    """ساخت محیط با جریمه خاص"""
    def _init():
        env = ABREnv(
            video_names=['bigbuckbunny', 'crowd_run', 'tearsofsteel_short'],
            trace_dir=str(PATHS['train_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=48
        )
        # اعمال جریمه مورد نظر
        env.REBUF_PENALTY_BASE = penalty_value
        return Monitor(env)
    return _init

def run_grid_search():
    print(f"\n🔍 Starting Grid Search for Best Penalty...")
    print(f"Candidates: {CANDIDATE_PENALTIES}")
    print("="*60)
    
    send_slack_message("info", "Grid Search Started", 
                      f"Starting penalty tuning with candidates: {CANDIDATE_PENALTIES}")
    
    results = []

    for idx, penalty in enumerate(CANDIDATE_PENALTIES, 1):
        print(f"\n🧪 Testing Penalty: {penalty} ({idx}/{len(CANDIDATE_PENALTIES)})")
        
        send_slack_message("info", f"Testing Penalty {penalty}", 
                          f"Starting trial {idx}/{len(CANDIDATE_PENALTIES)}: Testing penalty={penalty}")
        
        # 1. آموزش مدل (Training)
        # از DummyVecEnv استفاده می‌کنیم تا سبک‌تر باشد
        train_env = DummyVecEnv([make_env_with_penalty(penalty, i) for i in range(4)])
        
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            verbose=0  # Silent training
        )
        
        print(f"   Training for {TRAIN_STEPS_PER_TRIAL} steps...", end='', flush=True)
        model.learn(total_timesteps=TRAIN_STEPS_PER_TRIAL)
        print(" Done.")
        
        send_slack_message("info", f"Training Completed", 
                          f"Penalty {penalty}: Training completed, starting evaluation...")
        
        # 2. ارزیابی (Evaluation)
        print(f"   Evaluating on {TEST_VIDEO}...", end='', flush=True)
        eval_env = ABREnv(
            video_names=[TEST_VIDEO], # فقط روی ویدیوی سخت تست کن
            trace_dir=str(PATHS['test_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features'])
        )
        eval_env.REBUF_PENALTY_BASE = penalty # مهم: اعمال همان جریمه در تست
        
        metrics = {'vmaf': [], 'rebuf': [], 'qoe': []}
        
        for _ in range(15): # 15 اپیزود تست
            obs, info = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, info = eval_env.step(action)
            
            metrics['vmaf'].append(info['avg_quality'])
            metrics['rebuf'].append((info['total_rebuffer'] / (eval_env.chunk_idx * 4.0)) * 100)
            
            # محاسبه QoE استاندارد برای مقایسه عادلانه (نه با جریمه متغیر)
            # اینجا از جریمه استاندارد 50 استفاده می‌کنیم تا معیار سنجش یکی باشد
            std_qoe = info['total_quality'] - (50.0 * info['total_rebuffer']) - (0.1 * info['total_smoothness'])
            metrics['qoe'].append(std_qoe)

        avg_vmaf = np.mean(metrics['vmaf'])
        avg_rebuf = np.mean(metrics['rebuf'])
        avg_qoe = np.mean(metrics['qoe'])
        
        print(f"\n   -> Result: VMAF={avg_vmaf:.2f}, Rebuf={avg_rebuf:.2f}%, QoE={avg_qoe:.2f}")
        
        # Send result to Slack
        result_msg = f"Penalty {penalty} Results:\n• VMAF: {avg_vmaf:.2f}\n• Rebuffer: {avg_rebuf:.2f}%\n• QoE: {avg_qoe:.2f}"
        send_slack_message("success", f"Penalty {penalty} Completed", result_msg)
        
        results.append({
            'Penalty': penalty,
            'VMAF': avg_vmaf,
            'Rebuffer (%)': avg_rebuf,
            'QoE': avg_qoe
        })
        
        train_env.close()

    # 3. نمایش و ذخیره نتایج
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*60)
    print(df)
    
    # محاسبه آمار کلی
    best_vmaf_idx = df['VMAF'].idxmax()
    best_qoe_idx = df['QoE'].idxmax()
    lowest_rebuf_idx = df['Rebuffer (%)'].idxmin()
    
    # پیدا کردن بهترین مقدار
    # شرط: بیشترین QoE به شرطی که ریبافر زیر 3% باشد
    valid_configs = df[df['Rebuffer (%)'] < 3.0]
    if not valid_configs.empty:
        best_cfg = valid_configs.loc[valid_configs['QoE'].idxmax()]
        best_msg = f"Best Penalty: {best_cfg['Penalty']}\n• QoE: {best_cfg['QoE']:.1f}\n• VMAF: {best_cfg['VMAF']:.2f}\n• Rebuffer: {best_cfg['Rebuffer (%)']:.2f}%"
        print(f"\n✅ Best Penalty Found: {best_cfg['Penalty']} (QoE={best_cfg['QoE']:.1f})")
        send_slack_message("success", "Best Penalty Found", best_msg)
    else:
        warning_msg = "No config satisfied Rebuffer < 3%. Manual selection required."
        print(f"\n⚠ {warning_msg}")
        send_slack_message("warning", "No Valid Config", warning_msg)
    
    # ذخیره نمودار
    _, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Penalty Value')
    ax1.set_ylabel('VMAF Score', color=color)
    ax1.plot(df['Penalty'], df['VMAF'], color=color, marker='o', label='VMAF')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Rebuffer (%)', color=color)
    ax2.plot(df['Penalty'], df['Rebuffer (%)'], color=color, marker='x', linestyle='--', label='Rebuffer')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Trade-off Analysis: VMAF vs Rebuffer")
    plt.tight_layout()
    plot_path = PATHS['results'] / 'penalty_tuning.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"\n✓ Plot saved to: {plot_path}")
    
    # ساخت پیام جزئیات کامل برای Slack
    details_msg = "📊 *Detailed Results Summary*\n\n"
    
    # جدول کامل نتایج
    details_msg += "*All Configurations:*\n"
    for _, row in df.iterrows():
        details_msg += f"• Penalty {row['Penalty']:.1f}: VMAF={row['VMAF']:.2f}, Rebuf={row['Rebuffer (%)']:.2f}%, QoE={row['QoE']:.1f}\n"
    
    # آمار کلی
    details_msg += f"\n*Statistics:*\n"
    details_msg += f"• Best VMAF: {df.loc[best_vmaf_idx, 'VMAF']:.2f} (Penalty: {df.loc[best_vmaf_idx, 'Penalty']:.1f})\n"
    details_msg += f"• Best QoE: {df.loc[best_qoe_idx, 'QoE']:.1f} (Penalty: {df.loc[best_qoe_idx, 'Penalty']:.1f})\n"
    details_msg += f"• Lowest Rebuffer: {df.loc[lowest_rebuf_idx, 'Rebuffer (%)']:.2f}% (Penalty: {df.loc[lowest_rebuf_idx, 'Penalty']:.1f})\n"
    
    # بهترین انتخاب
    if not valid_configs.empty:
        details_msg += f"\n*✅ Recommended Penalty:* {best_cfg['Penalty']:.1f}\n"
        details_msg += f"  - QoE: {best_cfg['QoE']:.1f}\n"
        details_msg += f"  - VMAF: {best_cfg['VMAF']:.2f}\n"
        details_msg += f"  - Rebuffer: {best_cfg['Rebuffer (%)']:.2f}%\n"
    
    # اطلاعات اضافی
    details_msg += f"\n*Test Configuration:*\n"
    details_msg += f"• Test Video: {TEST_VIDEO}\n"
    details_msg += f"• Training Steps: {TRAIN_STEPS_PER_TRIAL:,}\n"
    details_msg += f"• Total Trials: {len(CANDIDATE_PENALTIES)}\n"
    details_msg += f"\n📈 Plot saved to: `{plot_path}`"
    
    # ارسال پیام جزئیات
    send_slack_message("success", "Grid Search Completed - Full Details", details_msg)

if __name__ == '__main__':
    run_grid_search()