"""
تست بهترین مدل از آموزش low_lr (نرخ یادگیری 1e-4)
همراه با Safety Wrapper
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) # فرض می‌کنیم اسکریپت در scripts/evaluation/ است

import torch
import numpy as np
from tqdm import tqdm

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.policy_wrapper import SafetyWrapper # <-- ایمپورت کردن Safety Wrapper

print("=" * 80)
print("🧪 Testing Best Model from 'low_lr' Training (1e-4)")
print("=" * 80)
print()

# ═══════════════════════════════════════════════════════════
# بارگذاری مدل
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContentAwareActor(state_dim = (6, 8), action_dim = 6, content_dim = 2).to(DEVICE)

# ✅✅✅ تغییر اصلی اینجاست ✅✅✅
# ما بهترین مدل را از پوشه آموزش low_lr می‌خوانیم
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
    print(f"   Did the training script 'train_fcc_from_scratch_g.py' finish?")
    sys.exit(1)

model.eval()


# ═══════════════════════════════════════════════════════════
# Safety Wrapper
# ═══════════════════════════════════════════════════════════
# استفاده از همان Wrapper که در گزارش اولیه بهترین نتایج را داد
wrapper = SafetyWrapper(
    low_buffer_threshold=5.0,  # ثانیه
    low_buffer_action_idx=0,   # 300 kbps
    mid_buffer_threshold=10.0, # ثانیه
    mid_buffer_action_idx=2    # 1850 kbps
)
print("✅ Safety Wrapper enabled.")
print(f"   Low Buffer Threshold: {wrapper.low_buffer_threshold}s -> Action {wrapper.low_buffer_action_idx}")
print(f"   Mid Buffer Threshold: {wrapper.mid_buffer_threshold}s -> Action {wrapper.mid_buffer_action_idx}")
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

# از محیط اصلی آگاه از محتوا برای ارزیابی استفاده می‌کنیم
env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test'
)
print(f"   ✅ Environment loaded. Testing on {len(loader.test_traces)} traces.")
print()

# ═══════════════════════════════════════════════════════════
# حلقه ارزیابی
# ═══════════════════════════════════════════════════════════

rewards = []
rebuffers = []
bitrates_list = []
safety_counts = []
vmafs_list = [] # اضافه شد برای لاگ VMAF

num_test_episodes = len(loader.test_traces)

print(f"🧪 Running evaluation on {num_test_episodes} test episodes...")
print("-" * 80)

for ep in tqdm(range(num_test_episodes), desc="Evaluating Model"):
    state = env.reset(video_id=np.random.randint(1, 7)) # تست روی ویدئوهای مختلف
    ep_reward = 0
    ep_rebuffer = 0
    ep_bitrates = []
    ep_vmafs = []
    ep_safety = 0
    done = False

    while not done:
        network_state = torch.FloatTensor(state['network']).unsqueeze(0).to(DEVICE)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0).to(DEVICE)
        vmaf_features = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action_probs, _ = model(network_state, content_features, vmaf_features)
            model_action = action_probs.argmax(dim=1).item()

        # --- اعمال Safety Wrapper ---
        # (کد Wrapper از فایل eval_final.py شما گرفته شده است)
        safe_action = wrapper.get_safe_action(
            model_action=model_action,
            buffer_level=env.buffer # ارسال بافر فعلی به Wrapper
        )

        if safe_action != model_action:
            ep_safety += 1
        # ---------------------------

        state, reward, done, info = env.step(safe_action)

        ep_reward += reward
        ep_rebuffer += info['rebuffer_time']
        ep_bitrates.append(info['bitrate'])
        ep_vmafs.append(info.get('vmaf', 0.0)) # استفاده از .get() برای جلوگیری از خطا

    rewards.append(ep_reward)
    rebuffers.append(ep_rebuffer)
    bitrates_list.append(np.mean(ep_bitrates) if ep_bitrates else 0)
    vmafs_list.append(np.mean(ep_vmafs) if ep_vmafs else 0)
    safety_counts.append(ep_safety)

print()
print("=" * 80)
print("📊 FINAL TEST RESULTS (Low LR Model + Safety Wrapper)")
print("=" * 80)
print()

print(f"Reward (VMAF-based):")
print(f"  Mean:        {np.mean(rewards):+8.2f}")
print(f"  Std:         {np.std(rewards):8.2f}")
print(f"  Min:         {np.min(rewards):+8.2f}")
print(f"  Max:         {np.max(rewards):+8.2f}")
print()

print(f"Rebuffering:")
print(f"  Mean:        {np.mean(rebuffers):8.2f}s")
print(f"  Total:       {np.sum(rebuffers):8.2f}s")
print()

print(f"Bitrate:")
print(f"  Mean:        {np.mean(bitrates_list):8.0f} kbps")
print()

print(f"VMAF (Reference):")
print(f"  Mean:        {np.mean(vmafs_list):8.1f}")
print()

print(f"Safety Wrapper Interventions:")
print(f"  Mean:        {np.mean(safety_counts):8.2f} per episode")
print(f"  Total:       {np.sum(safety_counts):8.0f} times")
print()

print("=" * 80)
print("vs BBA Baseline:")
bba_reward = 102.16 # از گزارش شما
print(f"  Baseline (BBA): {bba_reward:+8.2f}")
print(f"  Our Model:      {np.mean(rewards):+8.2f}")
improvement = ((np.mean(rewards) - bba_reward) / abs(bba_reward)) * 100
print(f"  Improvement:    {improvement:+.1f}%")
if improvement > 0:
    print("  Status:         ✅ Better!")
else:
    print("  Status:         ⚠️ Worse or Equal")
print("=" * 80)