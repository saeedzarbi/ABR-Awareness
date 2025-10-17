# scripts/evaluation/evaluate_fcc_model.py

import sys
from pathlib import Path
# اضافه کردن root پروژه به path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm # اضافه کردن tqdm برای نمایش پیشرفت

# ایمپورت مدل و محیط اصلی شما
from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

print("=" * 70)
print("🧪 Evaluating YOUR Content-Aware Model on TEST Set")
print("=" * 70)

# --- تنظیمات ---
CHECKPOINT_PATH = 'results/fcc_training/checkpoint_400.pth' # مسیر چک‌پوینت مورد نظر
# (می‌توانید checkpoint_100.pth یا هر کدام که بهتر بود را انتخاب کنید)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_TEST_EPISODES = 50 # تعداد اپیزودهای تست برای میانگین‌گیری

# ۱. لود کردن مدل آموزش‌دیده
print(f"\n📦 Loading model from: {CHECKPOINT_PATH}")
model = ContentAwareActor(state_dim=(6, 8), action_dim=6, content_dim=2).to(DEVICE)

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    update_num = checkpoint.get('update', 'N/A') # گرفتن شماره آپدیت از چک‌پوینت
    print(f"   ✅ Loaded checkpoint from update {update_num}")
except Exception as e:
    print(f"   ❌ Error loading checkpoint! {e}")
    sys.exit(1)

model.eval() # تنظیم مدل روی حالت ارزیابی (غیرفعال کردن dropout و ...)

# ۲. لود کردن محیط تست (محیط اصلی با پاداش VMAF)
print("\n📦 Loading test environment (VMAF-based reward)...")
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
    mode='test'  # استفاده از داده‌های تست
)
print("   ✅ Environment loaded")

# ۳. اجرای ارزیابی
num_episodes_to_run = min(NUM_TEST_EPISODES, len(loader.test_traces)) # اجرای حداکثر به تعداد فایل‌های تست
print(f"\n🧪 Running evaluation on {num_episodes_to_run} test episodes...")
print("-" * 70)

episode_rewards = []
episode_rebuffers = []
episode_bitrates = []
episode_vmafs = []

# استفاده از tqdm برای نمایش نوار پیشرفت
for ep in tqdm(range(num_episodes_to_run), desc="Evaluating Your Model"):
    state = env.reset(video_id=np.random.randint(1, 7)) # انتخاب تصادفی ویدئو برای هر اپیزود
    episode_reward = 0
    episode_rebuffer = 0
    bitrates = []
    vmafs = []
    done = False

    while not done:
        # آماده‌سازی state برای ورودی مدل
        network_state = torch.FloatTensor(state['network']).unsqueeze(0).to(DEVICE)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0).to(DEVICE)
        vmaf_features = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(DEVICE)

        # انتخاب بهترین اکشن (بدون نمونه‌گیری تصادفی)
        with torch.no_grad():
            action_probs, _ = model(network_state, content_features, vmaf_features)
            action = action_probs.argmax(dim=1).item() # انتخاب قطعی بهترین اکشن

        # اجرای اکشن در محیط
        state, reward, done, info = env.step(action)

        # جمع‌آوری آمار اپیزود
        episode_reward += reward
        episode_rebuffer += info['rebuffer_time']
        bitrates.append(info['bitrate'])
        vmafs.append(info.get('vmaf', 0.0)) # استفاده از .get() برای اطمینان

    # ذخیره نتایج اپیزود
    episode_rewards.append(episode_reward)
    episode_rebuffers.append(episode_rebuffer)
    episode_bitrates.append(np.mean(bitrates) if bitrates else 0)
    episode_vmafs.append(np.mean(vmafs) if vmafs else 0)


# ۴. نمایش نتایج نهایی
print("\n" + "=" * 70)
print("📊 YOUR CONTENT-AWARE MODEL TEST SET RESULTS")
print(f"(Using Checkpoint: {CHECKPOINT_PATH})")
print("=" * 70)
if episode_rewards: # اطمینان از اینکه حداقل یک اپیزود اجرا شده
    print(f"Reward (VMAF-based):")
    print(f"  Mean:  {np.mean(episode_rewards):+7.2f}")
    print(f"  Std:   {np.std(episode_rewards):7.2f}")
    print(f"  Min:   {np.min(episode_rewards):+7.2f}")
    print(f"  Max:   {np.max(episode_rewards):+7.2f}")
    print()
    print(f"Rebuffering:")
    print(f"  Mean:  {np.mean(episode_rebuffers):7.2f}s")
    print(f"  Total (Sum): {np.sum(episode_rebuffers):7.2f}s in {num_episodes_to_run} episodes")
    print()
    print(f"Bitrate:")
    print(f"  Mean:  {np.mean(episode_bitrates):7.0f} kbps")
    print(f"  Std:   {np.std(episode_bitrates):7.0f} kbps")
    print()
    print(f"VMAF:")
    print(f"  Mean:  {np.mean(episode_vmafs):7.1f}")
    print("=" * 70)

    # مقایسه با بیس‌لاین BBA (مقدار ثابت از گزارش شما)
    bba_reward = 102.16
    print("Baseline Comparison (BBA):")
    print(f"  BBA Reward:      {bba_reward:+7.2f}")
    print(f"  Your Model Mean: {np.mean(episode_rewards):+7.2f}")
    if bba_reward != 0:
      improvement = ((np.mean(episode_rewards) - bba_reward) / abs(bba_reward)) * 100
      print(f"  Improvement vs BBA: {improvement:+.1f}%")
    print("=" * 70)
else:
    print("❌ No episodes were run. Check configuration or data.")