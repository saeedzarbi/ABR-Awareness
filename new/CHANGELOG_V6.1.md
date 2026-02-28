# V6.1 Fixes (Post-Diagnosis)

## مشکل
مدل Proposed V6 در ارزیابی رتبه ۶ از ۹ داشت و در بیش از ۵۰٪ چانک‌ها ریبافر می‌کرد. علت اصلی: جریمه ریبافر در CMDP Lagrangian ضعیف شده بود و `lambda_rebuf` به سمت مقادیر پایین می‌رفت، در نتیجه مدل تقریباً همیشه ۱۲۰۰ kbps انتخاب می‌کرد.

## تغییرات اعمال‌شده

### 1. `src/training/constrained_abr_v6.py`
- **rebuf_target:** `0.07` → `0.05` (محدودیت ریبافر سخت‌گیرتر)
- **lambda_rebuf_range:** `(2.0, 8.0)` → `(4.3, 12.0)`
  - حداقل ۴.۳ = وزن ارزیابی، تا جریمه ریبافر هرگز کمتر از ارزیابی نباشد
  - حداکثر ۱۲ تا در صورت ریبافر زیاد، dual بتواند رشد کند

### 2. `src/training/constrained_abr.py`
- **به‌روزرسانی dual:** وقتی ریبافر از هدف بیشتر است (`rebuf_gap > 0`)، `lambda_rebuf` با ضریب **۲** افزایش می‌یابد (واکنش قوی‌تر به نقض محدودیت). وقتی کمتر از هدف است، با ضریب **۱** کاهش می‌یابد تا `lambda` به‌سرعت پایین نیاید.

### 3. `src/environment/abr_multi_env_v6.py`
- **REBUF_PENALTY_BASE:** `5.0` → `6.0` (هم‌راستا با ارزیابی و جلوگیری از سیاست بیش از حد تهاجمی)

### 4. `src/evaluation/evaluate_all_models_v6.py`
- **CATASTROPHIC_RATIO:** `2.5` → `2.0` (در حالت light guard زودتر مداخله شود وقتی `dl_time > 2 * buffer`)

### 5. `src/training/train_all_models_v6.py`
- به‌روزرسانی docstring با اشاره به تنظیمات V6.1

## قدم بعدی
مدل **Proposed** را دوباره train کنید (فقط proposed؛ بقیه مدل‌ها بدون Lagrangian هستند):

```bash
cd new/src/training
python train_all_models_v6.py --models proposed
```

سپس ارزیابی را با هر سه حالت raw / light / safe اجرا کنید.
