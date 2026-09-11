# اصلاحات خواسته‌شده — وضعیت پیاده‌سازی

**منبع:** نوت‌های PDF (`__25_`, `__11_`) + چک‌لیست `res.md`  
**نسخهٔ مرجع:** `new/src/paper/overleaf_upload/main.tex`  
**تاریخ:** 2026-08-29 (به‌روز)

---

## ۱. Abstract — نوت‌های PDF

| # | خواسته | وضعیت |
|---|--------|--------|
| 1 | حذف «blind spot» | ✅ `per-chunk limitation` |
| 2 | trade-off saturation | ⏸ اختیاری — skip |
| 3 | converts → tradeoff | ✅ `exploits saturation to bank buffer headroom` |
| 4 | First/Second/Third | ⏸ skip |
| 5–6 | certificate probabilistic | ✅ |
| 7–8 | savings / ??? | ⏸ skip |
| 9 | ABR | ✅ |

---

## ۲. Introduction — نوت‌های PDF

| # | خواسته | وضعیت |
|---|--------|--------|
| 1 | whereas stall/buffer | ✅ |
| 2 | session → per-chunk transition | ✅ |
| 3 | marginal VMAF at top rungs | ✅ |
| 4 | خلاصه related work | ⏸ skip |
| 5 | supports negligible bitrate reductions | ✅ |
| 6 | shield too rigid | ✅ |
| 7 | VMAF-knee conservation wording | ⏸ skip (banking ثابت ماند) |

---

## ۲b. Related Work — نوت PDF

| # | خواسته | وضعیت |
|---|--------|--------|
| 1 | Clients execute local adaptation policies... | ✅ §2.1 |

---

## ۲c. System model — نوت PDF (`__11_`)

| # | خواسته | وضعیت |
|---|--------|--------|
| 1 | CPS label | ✅ N/A |
| 2 | PPO یک‌دفعه آمد | ✅ defer به `Section~\ref{sec:system:cmdp}` |
| 3 | §3.6 CPS در system model | ✅ قبلاً به Method منتقل شده |
| 4 | Quality saturates at upper rungs | ✅ §3.2 |

---

## ۳. res.md — چک‌لیست

| # | خواسته | وضعیت |
|---|--------|--------|
| 1–4 | co-design demote, ladder split, exchangeability, ε TOST | ✅ |
| C | coverage-under-drift | ✅ **جدید** — `coverage_under_drift.py` + Fig. + §5 |
| A | Lumos5G primary | DEFER |
| B | حذف §5.4 | ⏸ skip |
| D | عنوان Certified | ⏸ اختیاری |

---

## ۴. آزمایش coverage-under-drift

**اسکریپت:** `new/src/evaluation/coverage_under_drift.py`

```bash
cd new
python src/evaluation/coverage_under_drift.py --episodes 204
```

**خروجی:** `new/results/v18_certified/coverage_drift/`  
**شکل مقاله:** `overleaf_upload/figures/fig_cps_coverage_drift.pdf`

**نتایج (greedy certified, α=0.10, shock 0.3× @ chunks 28–37):**

| فاز | Coverage | n chunks |
|-----|----------|----------|
| pre (steady, post warm-up) | 0.928 | 1,632 |
| during shock | 0.929 | 2,040 |
| post recovery | 0.944 | 2,040 |

**متن:** §5 `Coverage under injected drift` + Fig.~\ref{fig:cps:coverage_drift}

---

## ۵. اقدام باقیمانده

| اولویت | کار |
|--------|-----|
| DEFER | Lumos5G/Ghent primary eval |
| اختیاری | تعدیل عنوان «Certified» |
| مکانیکی | regenerate `overleaf_cps_v19_prose.zip` پس از compile |

---

*آخرین تغییر: PDF stylistic fixes + coverage-under-drift experiment integrated*
