"""
تست بهترین مدل از آموزش 'low_lr' (نرخ یادگیری 1e-4)
با استفاده از Wrapper پیشرفته (BufferAwarePolicy + SmoothPolicy)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) # فرض می‌کنیم اسکریپت در scripts/evaluation/ است

import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# ایمپورت مدل اصلی، محیط و لودر
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

# ✅ 1. ایمپورت Wrapper های جدید از فایلی که ارائه دادید
try:
    from models.policy_wrapper import BufferAwarePolicy, SmoothPolicy
except ImportError:
    print("خطا: فایل 'models/policy_wrapper.py' که حاوی BufferAwarePolicy است، پیدا نشد.")
    print("لطفاً مطمئن شوید فایل Wrapper را در پوشه 'models/' ذخیره کرده‌اید.")
    sys.exit(1)


print("=" * 80)
print("🧪 Testing Best Model from 'low_lr' Training (1e-4)")
print("   with Advanced BufferAware + Smooth Wrapper")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════
# بارگذاری مدل
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContentAwareActor(state_dim = (6, 8), action_dim = 6, content_dim = 2).to(DEVICE)

# ✅ 2. بارگذاری بهترین مدل از آموزش 'low_lr'
CHECKPOINT_PATH = 'results/fcc_training/checkpoint_best.pth'

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    update_num = checkpoint.get('update', 'N/A')
    val_reward = checkpoint.get('val_reward', 'N/A')
    print(f"✅ Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"   Trained for {update_num} updates.")
    print(f"   Best Validation Reward during training: {val_reward:+.2f}")
    print()
except Exception as e:
    print(f"❌ Error loading checkpoint! {e}")
    print("   آیا اسکریپت 'train_fcc_from_scratch_g.py' با کانفیگ low_lr تمام شده است؟")
    sys.exit(1)

model.eval() # تنظیم مدل روی حالت ارزیابی


# ═══════════════════════════════════════════════════════════
# ✅ 3. ساخت Policy Wrapper پیشرفته
# ═══════════════════════════════════════════════════════════
# ساخت پالیسی پایه با قوانین بافر
buffer_policy = BufferAwarePolicy(model)
# ساخت پالیسی نهایی با اعمال قوانین Smoothing روی پالیسی بافر
policy = SmoothPolicy(buffer_policy, max_jump=2) # max_jump=2 از گزارش اولیه شما

print("✅ Advanced Policy Wrapper (BufferAware + Smooth) enabled.")
print(f"   Max jump size: {policy.max_jump}")
print()


# ═══════════════════════════════════════════════════════════
# بارگذاری محیط تست
# ═══════════════════════════════════════════════════════════
print("📦 Loading Test Environment (VMAF-based reward)...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'
)
num_test_episodes = len(loader.test_traces)
print(f"   ✅ Environment loaded. Testing on {num_test_episodes} traces.")
print()

# ═══════════════════════════════════════════════════════════
# ✅ 4. حلقه ارزیابی اصلاح‌شده
# ═══════════════════════════════════════════════════════════

results = [] # ذخیره نتایج کامل هر اپیزود

print(f"🧪 Running evaluation on {num_test_episodes} test episodes...")
print("-" * 80)

for ep in tqdm(range(num_test_episodes), desc="Evaluating Model"):
    policy.reset() # ریست کردن وضعیت Wrapper (مانند last_action)
    state = env.reset(video_id=np.random.randint(1, 7))

    ep_reward = 0
    ep_rebuffer_time = 0
    ep_bitrates = []
    ep_vmafs = []

    done = False
    recent_rebuffer = 0.0 # متغیر برای نگهداری توقف مرحله قبل

    while not done:
        # انتخاب اکشن با استفاده از Wrapper پیشرفته
        action = policy.select_action(
            state,
            env.buffer, # ارسال بافر فعلی
            recent_rebuffer # ارسال زمان توقف مرحله قبلی
        )

        state, reward, done, info = env.step(action)

        # به‌روزرسانی متغیرها برای ارسال به Wrapper در گام بعدی
        recent_rebuffer = info['rebuffer_time']

        # جمع‌آوری آمار اپیزود
        ep_reward += reward
        ep_rebuffer_time += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
        ep_vmafs.append(info.get('vmaf', 0.0))

    # ذخیره نتایج این اپیزود
    results.append({
        'reward': ep_reward,
        'rebuffer_time': ep_rebuffer_time,
        'avg_bitrate': np.mean(ep_bitrates) if ep_bitrates else 0,
        'avg_vmaf': np.mean(ep_vmafs) if ep_vmafs else 0
    })

print()
print("=" * 80)
print("📊 FINAL TEST RESULTS (Low LR Model + Advanced Wrapper)")
print("=" * 80)
print()

# محاسبه آمار نهایی
mean_reward = np.mean([r['reward'] for r in results])
std_reward = np.std([r['reward'] for r in results])
min_reward = np.min([r['reward'] for r in results])
max_reward = np.max([r['reward'] for r in results])

mean_rebuffer = np.mean([r['rebuffer_time'] for r in results])
total_rebuffer = np.sum([r['rebuffer_time'] for r in results])

mean_bitrate = np.mean([r['avg_bitrate'] for r in results])
mean_vmaf = np.mean([r['avg_vmaf'] for r in results])

print(f"Reward (VMAF-based):")
print(f"  Mean:        {mean_reward:+8.2f}")
print(f"  Std:         {std_reward:8.2f}")
print(f"  Min:         {min_reward:+8.2f}")
print(f"  Max:         {max_reward:+8.2f}")
print()

print(f"Rebuffering:")
print(f"  Mean:        {mean_rebuffer:8.2f}s")
print(f"  Total:       {total_rebuffer:8.2f}s")
print()

print(f"Bitrate:")
print(f"  Mean:        {mean_bitrate:8.0f} kbps")
print()

print(f"VMAF (Reference):")
print(f"  Mean:        {mean_vmaf:8.1f}")
print()


print("=" * 80)
print("vs BBA Baseline:")
bba_reward = 102.16 # از گزارش شما
print(f"  Baseline (BBA): {bba_reward:+8.2f}")
print(f"  Our Model:      {mean_reward:+8.2f}")
improvement = ((mean_reward - bba_reward) / abs(bba_reward)) * 100
print(f"  Improvement:    {improvement:+.1f}%")
if improvement > 0:
    print("  Status:         ✅ Better!")
else:
    print("  Status:         ⚠️ Worse or Equal")
print("=" * 80)

# ذخیره نتایج در فایل JSON
output_results = {
    "model": CHECKPOINT_PATH,
    "wrapper": "BufferAwarePolicy + SmoothPolicy(max_jump=2)",
    "timestamp": datetime.now().isoformat(),
    "n_episodes": num_test_episodes,
    "metrics": {
        "reward": {"mean": mean_reward, "std": std_reward, "min": min_reward, "max": max_reward},
        "rebuffering": {"mean": mean_rebuffer, "total": total_rebuffer},
        "bitrate": {"mean": mean_bitrate},
        "vmaf": {"mean": mean_vmaf}
    },
    "comparison": {
        "baseline_bba": bba_reward,
        "improvement_percent": improvement
    }
}

output_filename = f"results/evaluation_low_lr_advanced_wrapper.json"
try:
    with open(output_filename, 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"\n📝 Detailed results saved to: {output_filename}")
except Exception as e:
    print(f"\n❌ Error saving results JSON: {e}")