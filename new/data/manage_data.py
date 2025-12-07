#!/usr/bin/env python3
"""
Network Trace Converter and Splitter
=====================================
این اسکریپت:
1. FCC traces (TSV) رو به JSON تبدیل می‌کنه
2. Norway traces (JSON) رو کپی می‌کنه  
3. همه رو shuffle می‌کنه
4. به صورت 80/20 تقسیم می‌کنه (train/test)
5. در دایرکتوری‌های مناسب ذخیره می‌کنه
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import shutil

# ═══════════════════════════════════════════════════════════════════════
# تنظیمات
# ═══════════════════════════════════════════════════════════════════════

# مسیر فایل‌های اصلی
FCC_RAW_DIR = Path('new/data/network_traces')          # فایل‌های FCC (TSV)
NORWAY_RAW_DIR = Path('new/data/norway')    # فایل‌های Norway (JSON)

# مسیر خروجی
OUTPUT_DIR = Path('new/data/standardized')
TRAIN_DIR = OUTPUT_DIR / 'train_traces'
TEST_DIR = OUTPUT_DIR / 'test_traces'

# نسبت تقسیم
TRAIN_RATIO = 0.8  # 80% برای training
TEST_RATIO = 0.2   # 20% برای test

# Random seed برای reproducibility
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════
# توابع کمکی
# ═══════════════════════════════════════════════════════════════════════

def convert_fcc_to_json(tsv_file: Path) -> Dict:
    """
    تبدیل FCC trace (TSV) به JSON format
    
    Input (TSV):
        0.106    1.132
        0.216    1.200
        ...
    
    Output (JSON):
        {"throughput_kbps": [1132, 1200, ...]}
    """
    throughputs = []
    
    try:
        with open(tsv_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    # ستون دوم = throughput در Mbps
                    throughput_mbps = float(parts[1])
                    # تبدیل به kbps
                    throughput_kbps = throughput_mbps * 1000.0
                    throughputs.append(throughput_kbps)
        
        return {"throughput_kbps": throughputs}
    
    except Exception as e:
        print(f"⚠️ خطا در خواندن {tsv_file.name}: {e}")
        return None


def load_norway_trace(json_file: Path) -> Dict:
    """
    بارگذاری Norway trace (قبلاً JSON است)
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # بررسی فرمت
        if 'throughput_kbps' not in data:
            print(f"⚠️ فرمت نامعتبر: {json_file.name}")
            return None
        
        return data
    
    except Exception as e:
        print(f"⚠️ خطا در خواندن {json_file.name}: {e}")
        return None


def get_trace_stats(data: Dict) -> Dict:
    """
    محاسبه آمار یک trace
    """
    throughputs = data['throughput_kbps']
    
    return {
        'count': len(throughputs),
        'min': min(throughputs),
        'max': max(throughputs),
        'mean': sum(throughputs) / len(throughputs),
        'median': sorted(throughputs)[len(throughputs) // 2]
    }


def save_json_trace(data: Dict, output_file: Path):
    """
    ذخیره trace به فرمت JSON
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ خطا در ذخیره {output_file.name}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# تابع اصلی
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("🚀 Network Trace Converter & Splitter")
    print("="*80)
    
    # تنظیم random seed
    random.seed(RANDOM_SEED)
    
    # ایجاد دایرکتوری‌های خروجی
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    all_traces = []
    
    # ─────────────────────────────────────────────────────────────────
    # مرحله 1: پردازش FCC traces
    # ─────────────────────────────────────────────────────────────────
    
    print("\n📂 مرحله 1: پردازش FCC Traces")
    print("-"*80)
    print(FCC_RAW_DIR)
    fcc_files = []
    if FCC_RAW_DIR.exists():
        # پیدا کردن تمام فایل‌ها (بدون فیلتر پسوند)
        fcc_files = [f for f in FCC_RAW_DIR.iterdir() if f.is_file()]
        print(f"   ✅ تعداد فایل‌های FCC پیدا شده: {len(fcc_files)}")
    else:
        print(f"   ⚠️ دایرکتوری FCC پیدا نشد: {FCC_RAW_DIR}")
        print(f"   💡 لطفاً فایل‌های FCC را در این مسیر قرار دهید")
    
    fcc_converted = 0
    for fcc_file in fcc_files:
        # تبدیل TSV به JSON
        data = convert_fcc_to_json(fcc_file)
        
        if data and len(data['throughput_kbps']) > 50:  # حداقل 50 sample
            all_traces.append({
                'name': f"fcc_{fcc_file.stem}",
                'data': data,
                'source': 'FCC'
            })
            fcc_converted += 1
        else:
            print(f"   ⚠️ رد شد (کم‌تر از 50 sample): {fcc_file.name}")
    
    print(f"   ✅ تبدیل موفق: {fcc_converted}/{len(fcc_files)}")
    
    # ─────────────────────────────────────────────────────────────────
    # مرحله 2: پردازش Norway traces
    # ─────────────────────────────────────────────────────────────────
    
    print("\n📂 مرحله 2: پردازش Norway Traces")
    print("-"*80)
    
    norway_files = []
    if NORWAY_RAW_DIR.exists():
        norway_files = list(NORWAY_RAW_DIR.glob('*.json'))
        print(f"   ✅ تعداد فایل‌های Norway پیدا شده: {len(norway_files)}")
    else:
        print(f"   ⚠️ دایرکتوری Norway پیدا نشد: {NORWAY_RAW_DIR}")
        print(f"   💡 اگر فایل‌های Norway دارید، در این مسیر قرار دهید")
    
    norway_loaded = 0
    for norway_file in norway_files:
        data = load_norway_trace(norway_file)
        
        if data and len(data['throughput_kbps']) > 50:
            all_traces.append({
                'name': f"norway_{norway_file.stem}",
                'data': data,
                'source': 'Norway'
            })
            norway_loaded += 1
        else:
            print(f"   ⚠️ رد شد: {norway_file.name}")
    
    print(f"   ✅ بارگذاری موفق: {norway_loaded}/{len(norway_files)}")
    
    # ─────────────────────────────────────────────────────────────────
    # مرحله 3: Shuffle & Split
    # ─────────────────────────────────────────────────────────────────
    
    print(f"\n🔀 مرحله 3: Shuffle & Split")
    print("-"*80)
    
    total_traces = len(all_traces)
    print(f"   📊 جمع کل traces: {total_traces}")
    
    if total_traces == 0:
        print("\n❌ هیچ trace معتبری پیدا نشد!")
        print("💡 لطفاً مسیرهای زیر را چک کنید:")
        print(f"   • FCC: {FCC_RAW_DIR.absolute()}")
        print(f"   • Norway: {NORWAY_RAW_DIR.absolute()}")
        return
    
    # Shuffle
    random.shuffle(all_traces)
    print(f"   ✅ Shuffled")
    
    # Split
    train_count = int(total_traces * TRAIN_RATIO)
    test_count = total_traces - train_count
    
    train_traces = all_traces[:train_count]
    test_traces = all_traces[train_count:]
    
    print(f"   ✅ Split: {train_count} train / {test_count} test")
    
    # ─────────────────────────────────────────────────────────────────
    # مرحله 4: ذخیره فایل‌ها
    # ─────────────────────────────────────────────────────────────────
    
    print(f"\n💾 مرحله 4: ذخیره فایل‌ها")
    print("-"*80)
    
    # ذخیره train traces
    print(f"   💾 ذخیره {len(train_traces)} train traces...")
    for i, trace in enumerate(train_traces, 1):
        output_file = TRAIN_DIR / f"{trace['name']}.json"
        save_json_trace(trace['data'], output_file)
        if i % 10 == 0:
            print(f"      Progress: {i}/{len(train_traces)}", end='\r')
    print(f"      ✅ {len(train_traces)} فایل ذخیره شد")
    
    # ذخیره test traces
    print(f"   💾 ذخیره {len(test_traces)} test traces...")
    for i, trace in enumerate(test_traces, 1):
        output_file = TEST_DIR / f"{trace['name']}.json"
        save_json_trace(trace['data'], output_file)
        if i % 10 == 0:
            print(f"      Progress: {i}/{len(test_traces)}", end='\r')
    print(f"      ✅ {len(test_traces)} فایل ذخیره شد")
    
    # ─────────────────────────────────────────────────────────────────
    # مرحله 5: نمایش آمار
    # ─────────────────────────────────────────────────────────────────
    
    print(f"\n📊 آمار نهایی:")
    print("="*80)
    
    # آمار کلی
    print(f"\n📁 تعداد فایل‌ها:")
    print(f"   • FCC traces:    {fcc_converted}")
    print(f"   • Norway traces: {norway_loaded}")
    print(f"   • جمع کل:       {total_traces}")
    
    print(f"\n📂 تقسیم‌بندی:")
    print(f"   • Training:      {train_count} ({TRAIN_RATIO*100:.0f}%)")
    print(f"   • Test:          {test_count} ({TEST_RATIO*100:.0f}%)")
    
    # آمار throughput
    print(f"\n📈 آمار Throughput (همه traces):")
    all_throughputs = []
    for trace in all_traces:
        all_throughputs.extend(trace['data']['throughput_kbps'])
    
    print(f"   • Min:    {min(all_throughputs):>7.1f} kbps")
    print(f"   • Max:    {max(all_throughputs):>7.1f} kbps")
    print(f"   • Mean:   {sum(all_throughputs)/len(all_throughputs):>7.1f} kbps")
    print(f"   • Median: {sorted(all_throughputs)[len(all_throughputs)//2]:>7.1f} kbps")
    
    # نمونه traces
    print(f"\n🔍 نمونه traces ذخیره شده:")
    print(f"\n   Train traces:")
    for trace in train_traces[:3]:
        stats = get_trace_stats(trace['data'])
        print(f"      • {trace['name'][:40]:40s}: {stats['count']:>4} samples, "
              f"Mean={stats['mean']:>6.1f} kbps")
    if len(train_traces) > 3:
        print(f"      ... و {len(train_traces)-3} trace دیگر")
    
    print(f"\n   Test traces:")
    for trace in test_traces[:3]:
        stats = get_trace_stats(trace['data'])
        print(f"      • {trace['name'][:40]:40s}: {stats['count']:>4} samples, "
              f"Mean={stats['mean']:>6.1f} kbps")
    if len(test_traces) > 3:
        print(f"      ... و {len(test_traces)-3} trace دیگر")
    
    # مسیرهای خروجی
    print(f"\n📍 مسیرهای خروجی:")
    print(f"   • Train: {TRAIN_DIR.absolute()}")
    print(f"   • Test:  {TEST_DIR.absolute()}")
    
    print("\n" + "="*80)
    print("✅ تبدیل و تقسیم‌بندی با موفقیت انجام شد!")
    print("="*80)
    
    print("\n💡 مراحل بعدی:")
    print("   1. فایل‌های JSON را چک کنید")
    print("   2. Training را شروع کنید:")
    print("      python experiments/training/train_ppo_multi_dynamic.py")
    print("   3. Evaluation:")
    print("      python experiments/evaluation/final_multi.py --episodes 20")


if __name__ == '__main__':
    main()