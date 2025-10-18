#!/bin/bash

# این اسکریپت تمام مدل‌ها را آموزش داده و سپس همه را روی
# مجموعه اعتبارسنجی (Validation Set) ارزیابی می‌کند.
# اگر هر دستوری خطا دهد، اسکریپت متوقف می‌شود.
set -e

# --- پیکربندی ---
# تمام خروجی‌ها در این فایل ذخیره می‌شوند
OUTPUT_LOG_FILE="all_validation_results_log.txt"

# --- شروع ---
# پاک کردن لاگ قبلی
rm -f "$OUTPUT_LOG_FILE"

echo "--- 🚀 شروع اجرای کامل آزمایش‌ها ---" | tee -a "$OUTPUT_LOG_FILE"
echo "--- زمان شروع: $(date) ---" | tee -a "$OUTPUT_LOG_FILE"
echo "--- لاگ کامل در $OUTPUT_LOG_FILE ذخیره خواهد شد ---" | tee -a "$OUTPUT_LOG_FILE"

# ============================================================
# بخش ۱: آموزش مدل‌ها
# ============================================================

# ۱. آموزش مدل اصلی شما (Content-Aware)
echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
echo "1. در حال آموزش مدل اصلی شما (Content-Aware)..." | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
# این اسکریپت مدل را در 'results/fcc_training_continued/' ذخیره می‌کند
# python3 scripts/training/train_fcc_from_scratch.py >> "$OUTPUT_LOG_FILE" 2>&1

# ۲. آموزش مدل Pensieve
echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
echo "2. در حال آموزش مدل Pensieve..." | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
# این اسکریپت مدل را در 'results/pensieve_fcc_training/' ذخیره می‌کند
python3 scripts/training/train_pensieve.py >> "$OUTPUT_LOG_FILE" 2>&1

# ۳. آموزش مدل Comyco (دو مرحله‌ای)
echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
echo "3. در حال آموزش مدل Comyco (مرحله ۱: تولید داده متخصص)..." | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
# این اسکریپت فایل 'data/expert_data.json' را می‌سازد
# (فرض می‌کنیم اسکریپت generate_expert_data.py در scripts/training/ قرار دارد)
python3 scripts/training/generate_expert_data.py >> "$OUTPUT_LOG_FILE" 2>&1

echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
echo "4. در حال آموزش مدل Comyco (مرحله ۲: آموزش تقلیدی)..." | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
# این اسکریپت مدل را در 'results/comyco_model.pth' ذخیره می‌کند
python3 scripts/training/train_comyco.py >> "$OUTPUT_LOG_FILE" 2>&1

echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "✅✅✅ آموزش تمام مدل‌ها کامل شد. ✅✅✅" | tee -a "$OUTPUT_LOG_FILE"


# ============================================================
# بخش ۲: ارزیابی مدل‌ها (روی مجموعه Validation)
# ============================================================

echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
echo "5. شروع ارزیابی مدل‌ها روی مجموعه اعتبارسنجی (Validation Set)" | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"

# ۵.۱ ارزیابی مدل شما (همان اسکریپت test_con.py)
echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "--- 5.1 ارزیابی مدل شما (Content-Aware) ---" | tee -a "$OUTPUT_LOG_FILE"
# این اسکریپت از 'mode=val' استفاده می‌کند
python3 test_con.py >> "$OUTPUT_LOG_FILE" 2>&1

# ۵.۲ ارزیابی مدل MPC
echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "--- 5.2 ارزیابی MPC ---" | tee -a "$OUTPUT_LOG_FILE"
# (فرض می‌کنیم اسکریپت evaluate_mpc_val.py که قبلاً دادم، موجود است)
python3 scripts/evaluation/evaluate_mpc_val.py >> "$OUTPUT_LOG_FILE" 2>&1

# ۵.۳ ارزیابی مدل Pensieve
echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "--- 5.3 ارزیابی Pensieve ---" | tee -a "$OUTPUT_LOG_FILE"
# (فرض می‌کنیم اسکریپت evaluate_pensieve_val.py که قبلاً دادم، موجود است)
python3 scripts/evaluation/evaluate_pensieve_val.py >> "$OUTPUT_LOG_FILE" 2>&1

# ۵.۴ ارزیابی مدل Comyco
echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "--- 5.4 ارزیابی Comyco ---" | tee -a "$OUTPUT_LOG_FILE"
# (فرض می‌کنیم اسکریپت evaluate_comyco_val.py که قبلاً دادم، موجود است)
python3 scripts/evaluation/evaluate_comyco_val.py >> "$OUTPUT_LOG_FILE" 2>&1


# ============================================================
# پایان
# ============================================================

echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "============================================================" | tee -a "$OUTPUT_LOG_FILE"
echo "🎉 اجرای تمام اسکریپت‌ها با موفقیت انجام شد." | tee -a "$OUTPUT_LOG_FILE"
echo "--- زمان پایان: $(date) ---" | tee -a "$OUTPUT_LOG_FILE"
echo "--- نتایج کامل در $OUTPUT_LOG_FILE موجود است. ---" | tee -a "$OUTPUT_LOG_FILE"