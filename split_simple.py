import os
import random

# تعداد فایل‌های FCC
files = os.listdir("data/fcc_traces")
# فقط فایل‌ها (نه پوشه cooked)
files = [f for f in files if not os.path.isdir(f"data/fcc_traces/{f}")]

print(f"تعداد کل فایل‌ها: {len(files)}")

# مرتب و shuffle
random.seed(42)
random.shuffle(files)

# تقسیم: 70% train, 15% val, 15% test
total = len(files)
train_files = files[:int(total * 0.7)]
val_files = files[int(total * 0.7):int(total * 0.85)]
test_files = files[int(total * 0.85):]

print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")
print(f"Test: {len(test_files)}")

# ساخت پوشه
os.makedirs("data/network_traces/fcc/splits", exist_ok=True)

# ذخیره
with open("data/network_traces/fcc/splits/fcc_train.txt", "w") as f:
    for name in sorted(train_files):
        f.write(name + "\n")

with open("data/network_traces/fcc/splits/fcc_val.txt", "w") as f:
    for name in sorted(val_files):
        f.write(name + "\n")

with open("data/network_traces/fcc/splits/fcc_test.txt", "w") as f:
    for name in sorted(test_files):
        f.write(name + "\n")

print("\n✅ تمام! فایل‌ها ذخیره شدن در: data/network_traces/fcc/splits/")
