import os
import json
import numpy as np

# ایجاد پوشه برای تریس‌های 5G
output_dir = "5g_test_traces_json"
os.makedirs(output_dir, exist_ok=True)

NUM_TRACES = 20
TRACE_LENGTH = 300  # هر تریس 300 ثانیه

print("Generating 5G mmWave JSON traces...")

for i in range(NUM_TRACES):
    time = 0.0
    state = 0 
    throughput_list_kbps = []
    
    while time < TRACE_LENGTH:
        if state == 0:
            # حالت 0: دید مستقیم (LoS) -> سرعت 5G بین 80 تا 150 مگابیت (80,000 تا 150,000 کیلوبیت)
            bw_kbps = np.random.uniform(80000.0, 150000.0)
            duration = np.random.exponential(8.0) 
            next_state = 1
        else:
            # حالت 1: مسدود شدن (NLoS) -> افت ناگهانی بین 2 تا 10 مگابیت (2,000 تا 10,000 کیلوبیت)
            bw_kbps = np.random.uniform(2000.0, 10000.0)
            duration = np.random.exponential(2.0) 
            next_state = 0
        
        # اضافه کردن پهنای باند به لیست به ازای هر ثانیه
        for _ in range(int(max(1, duration))):
            if time >= TRACE_LENGTH:
                break
            throughput_list_kbps.append(bw_kbps)
            time += 1.0
            
        state = next_state

    # ذخیره در قالب فایل JSON دقیقاً مشابه فرمت شما
    file_data = {
        "throughput_kbps": throughput_list_kbps
    }
    
    with open(os.path.join(output_dir, f"trace_5g_{i}.json"), "w") as f:
        json.dump(file_data, f, indent=2)

print(f"✅ {NUM_TRACES} JSON traces successfully generated in '{output_dir}' folder!")