
**فایل‌ها:** `new/src/paper/main.ltx` + `new/src/paper/overleaf_upload/main.tex` (همگام)  
**Overleaf:** `overleaf_cps_v27_phaseA.zip` (FMT-01/02 + SCI-06/07/05/15/21; plain refs + twocolumn)

## ۰. Related work 2024–26 (RW-001–004) ✅ v25
- پارagraph Conformal-ABR / SafeSABR / Kairos + پل CPS در §2
- سه سطر در `tab:rw_safety_vmaf` + intro جدول (Conf./Bank.)
- `references.bib`: `zhong2026conformalabr`, `liu2025safesabr`, `zhong2025kairos` (metadata با Crossref/arXiv چک شد)

## ۰b. Phase A brief (Computer Networks) ✅ v27
- FMT-01 abstract ≤250; FMT-02 `Highlights.txt`
- SCI-06 ranges; SCI-07 intervention rates; SCI-05 cite Fig for per-chunk
- SCI-15/21 ε_risk + soften claims; CPS-P naming; SCI-03 headers already OK (`0.00` fix)
- **SCI-01** monotonicity aligned with data/Table 3 (Prop 2 scoped; mid-ladder vs top-3)
- **SCI-02 Path A:** ablation default row = `greedy_5g` (4.5% / −9.3% / 70.4%); captions updated
- **SCI-04 Path C:** coverage-drift figure (warm-up band) + phase vs rolling prose/caption
- Changelog: `new/src/paper/CHANGELOG_revision.md`

## ۱. زمان فعل (p.4) ✅
- افعال کار انجام‌شده (evaluate, report, trained, was, …) → **ماضی** در abstract تا conclusion
- توصیف روش CPS (CPS selects, …) → **حال** ماند

## ۲. ارجاع Fig/Table (p.19) ✅ → **v26 final:** plain (بدون bold)
- ماکرو: `\figref{}` و `\tabref{}` → `Fig.~\ref` / `Table~\ref` (Elsevier)
- `Figure~\ref` → `Fig.`

## ۳. وضوح و نگارش ✅
- **SI/TI** تعریف شد (spatial/temporal complexity indices)
- **LOS/NLOS** باز شد (line-of-sight / non-line-of-sight)
- **Pairing** یک جمله در §5.1
- **Fig overview ↔ Table full** جمله پل در §5.2
- **namely** + «The shield then projects» برای knee rule
- **Algorithm** → «computes the knee index via eq.»
- **`X%/Y%`** → «X% bitrate and Y% rebuffering»
- **0.919** در sensitivity شفاف‌تر شد
- **versus** → `vs.\` (جاهای کلیدی)
- **proposed rung** در abstract
- intro قبل از جدول related-work positioning (RW-004؛ گسترش‌یافته در v25)
- caption co-design: diamond markers = episode means

## ۴. انجام نشده (باقیمانده)
- جابجایی بخش‌ها (→ §2، §5، conclusion، بعد از references)
- حذف/جابجایی بلوک‌های mark-only
- citation اضافه، بازنویسی telegraphic §5.3
- resolve «؟؟؟» با PDF باز
