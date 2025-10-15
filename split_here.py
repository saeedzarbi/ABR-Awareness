import os
import random

# پوشه فعلی
fcc_dir = "."

# لیست فایل‌ها (غیر از پوشه cooked)
files = []
for f in os.listdir(fcc_dir):
    if os.path.isfile(f) and f != "split_here.py" and f != "split_fcc.py":
        files.append(f)

print(f"✅ پیدا شد: {len(files)} فایل")

# Shuffle
random.seed(42)
random.shuffle(files)

# Split
n = len(files)
train = sorted(files[:int(n*0.7)])
val = sorted(files[int(n*0.7):int(n*0.85)])
test = sorted(files[int(n*0.85):])

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")

# ذخیره در پوشه والد
os.makedirs("../../network_traces/fcc/splits", exist_ok=True)

with open("../../network_traces/fcc/splits/fcc_train.txt", "w") as f:
    f.write("\n".join(train))

with open("../../network_traces/fcc/splits/fcc_val.txt", "w") as f:
    f.write("\n".join(val))

with open("../../network_traces/fcc/splits/fcc_test.txt", "w") as f:
    f.write("\n".join(test))

print("\n✅ Done!")
