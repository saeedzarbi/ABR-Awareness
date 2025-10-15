import os
import random

# مسیر مطلق
fcc_dir = "data/fcc_traces"

# لیست فایل‌ها
files = []
for f in os.listdir(fcc_dir):
    path = os.path.join(fcc_dir, f)
    # فقط فایل‌ها، نه پوشه‌ها
    if os.path.isfile(path):
        files.append(f)

print(f"✅ پیدا شد: {len(files)} فایل")

# Shuffle
random.seed(42)
random.shuffle(files)

# Split: 70% train, 15% val, 15% test
n = len(files)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train = sorted(files[:train_end])
val = sorted(files[train_end:val_end])
test = sorted(files[val_end:])

print(f"📊 Train: {len(train)} ({len(train)/n*100:.1f}%)")
print(f"📊 Val: {len(val)} ({len(val)/n*100:.1f}%)")
print(f"📊 Test: {len(test)} ({len(test)/n*100:.1f}%)")

# ساخت پوشه خروجی
output_dir = "data/network_traces/fcc/splits"
os.makedirs(output_dir, exist_ok=True)

# ذخیره
with open(f"{output_dir}/fcc_train.txt", "w") as f:
    f.write("\n".join(train))

with open(f"{output_dir}/fcc_val.txt", "w") as f:
    f.write("\n".join(val))

with open(f"{output_dir}/fcc_test.txt", "w") as f:
    f.write("\n".join(test))

print(f"\n✅ ذخیره شد در: {output_dir}/")
print("   - fcc_train.txt")
print("   - fcc_val.txt")
print("   - fcc_test.txt")
