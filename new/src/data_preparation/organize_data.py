import os
import json
import shutil
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
# مسیرهایی که فایل‌های خام شما در آنجا هستند
# (مسیرها را بر اساس ساختار فایل‌های آپلود شده شما تنظیم کردم)
SOURCE_FCC_DIR = Path("/home/saeedzarbi95/test/ABR-Awareness/data/fcc_traces")  # جایی که فایل‌های mixed هستند
SOURCE_NORWAY_DIR = Path("/home/saeedzarbi95/test/ABR-Awareness/data/network_traces/cooked_test_traces")

# مسیرهای خروجی استاندارد شده
DEST_TRAIN = Path("data/standardized/train_traces")
DEST_TEST = Path("data/standardized/test_traces")
# =================================================

def ensure_dirs():
    """Create destination directories if not exist."""
    if DEST_TRAIN.exists(): shutil.rmtree(DEST_TRAIN)
    if DEST_TEST.exists(): shutil.rmtree(DEST_TEST)
    
    DEST_TRAIN.mkdir(parents=True, exist_ok=True)
    DEST_TEST.mkdir(parents=True, exist_ok=True)
    print("✓ Clean directories created.")

def convert_and_save(file_path, dest_folder):
    """Reads a raw trace file and saves it as JSON in the destination."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        throughput_history = []
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            # استراتژی استخراج پهنای باند:
            # معمولاً ستون آخر پهنای باند است.
            # اگر عدد کوچک بود (مثلاً < 100) احتمالاً مگابیت است، به کیلوبیت تبدیل می‌کنیم.
            try:
                val = float(parts[-1])
                if val < 0.001: continue # حذف مقادیر صفر یا منفی
                
                # heuristic conversion: if val < 200, assume Mbps -> convert to Kbps
                # FCC traces are usually in seconds and throughput (sometimes Mbps, sometimes Kbps)
                # Norway traces are typically in Kbps
                
                final_val = val * 1000 if val < 100 else val
                throughput_history.append(final_val)
            except ValueError:
                continue
                
        if len(throughput_history) > 5:
            output_file = dest_folder / (file_path.stem + ".json")
            data = {"throughput_kbps": throughput_history}
            
            with open(output_file, 'w') as f:
                json.dump(data, f)
            return True
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return False
    return False

def main():
    print("📦 Organizing Data for IEEE TCSVT Submission (Smart Split)...")
    ensure_dirs()
    
    # --- 1. پردازش فایل‌های FCC (هوشمند) ---
    # این فایل‌ها شامل هم تست و هم آموزش هستند
    if SOURCE_FCC_DIR.exists():
        print(f"\nProcessing FCC Traces from {SOURCE_FCC_DIR}...")
        files = list(SOURCE_FCC_DIR.glob("*"))
        train_count = 0
        test_count = 0
        
        for file_path in files:
            if not file_path.is_file(): continue
            if file_path.suffix == '.json': continue # Skip processed files
            
            filename = file_path.name
            
            # منطق جداسازی بر اساس نام فایل
            if filename.startswith("test_"):
                # این یک تریس تست است
                if convert_and_save(file_path, DEST_TEST):
                    test_count += 1
            elif filename.startswith("trace_"):
                # این یک تریس آموزش است
                if convert_and_save(file_path, DEST_TRAIN):
                    train_count += 1
            else:
                # سایر فایل‌ها (اگر مطمئن نیستیم، فعلا در تست می‌گذاریم یا نادیده می‌گیریم)
                # برای امنیت بیشتر نادیده می‌گیریم تا داده پرت وارد نشود
                pass
                
        print(f"   -> {train_count} traces moved to TRAINING set.")
        print(f"   -> {test_count} traces moved to TEST set.")
    else:
        print(f"⚠ Source directory {SOURCE_FCC_DIR} not found. Please check path.")

    # --- 2. پردازش فایل‌های نروژ (همگی تست) ---
    if SOURCE_NORWAY_DIR.exists():
        print(f"\nProcessing Norway Traces from {SOURCE_NORWAY_DIR}...")
        norway_count = 0
        for file_path in SOURCE_NORWAY_DIR.glob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                if convert_and_save(file_path, DEST_TEST):
                    norway_count += 1
        print(f"   -> {norway_count} Norway traces added to TEST set.")
    
    # --- گزارش نهایی ---
    print("\n" + "="*50)
    print(f"✅ Final Dataset Status:")
    print(f"   Training Set: {len(list(DEST_TRAIN.glob('*.json')))} files")
    print(f"   Test Set:     {len(list(DEST_TEST.glob('*.json')))} files")
    print("="*50)
    print("Next: Run training scripts. They will now automatically pick up the correct splits.")

if __name__ == "__main__":
    main()