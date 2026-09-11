# راهنمای کامل ارائه — Certified Perceptual Shield (CPS)

**مقاله:** *Certified Perceptual Shielding for Adaptive Bitrate Streaming*  
**نویسندگان:** Saeed Zarbi, Leili Farzinvash, Pedram Salehpour — دانشگاه تبریز  
**هدف این فایل:** توضیح کامل کار انجام‌شده + پاسخ به سوالات محتمل استاد

---

## ۰. یک جمله — ما چه کردیم؟

یک **wrapper سبک و model-agnostic** ساختیم که بین policy پایه ABR و player می‌نشیند؛ با **conformal prediction** ایمنی chunk-level را certify می‌کند و با **VMAF-knee banking** روی saturation ladder، bandwidth را bank می‌کند — **بدون retrain اجباری**.

---

## ۱. مسئله (Problem)

### ۱.۱ ABR چیست؟
- Client هر **chunk** (مثلاً ۴ ثانیه) یک **representation index** از ladder انتخاب می‌کند.
- Hostها: greedy، BBA، BOLA، RobustMPC، PPO، …

### ۱.۲ دو failure mode در سطح chunk
| Failure mode | علت | پیامد |
|--------------|-----|--------|
| **Unsafe** | throughput ناگهان پایین می‌آید | download > buffer → **stall / rebuffering** |
| **Wasteful** | ladder در VMAF **اشباع** شده | bitrate بالاتر کیفیت محسوس اضافه نمی‌کند |

### ۱.۳ چرا shield کلاسیک کافی نیست؟
- معمولاً index را **decrement** می‌کند تا buffer test پاس شود.
- **Content-blind:** saturation ladder را exploit نمی‌کند.
- **No banking:** buffer headroom را از قبل ذخیره نمی‌کند.
- **No formal guarantee:** coverage/statistical certificate ندارد.

---

## ۲. راه‌حل — CPS (Certified Perceptual Shield)

### ۲.۱ تعریف
CPS تابع deterministic \(\mathcal{S}\) است:
\[
(a^{\mathrm{exec}}_k, \iota_k) = \mathcal{S}(s_k, a^{\mathrm{raw}}_k)
\]
- \(a^{\mathrm{raw}}\): proposal از policy
- \(a^{\mathrm{exec}}\): rung واقعاً دانلودشده
- \(\iota_k\): آیا shield مداخله کرد؟

### ۲.۲ سه جزء (حفظ کن — استاد می‌پرسد)

#### (A) Conformal throughput lower bound
- **Predictor:** harmonic mean از \(\kappa=5\) throughput اخیر (RobustMPC-style)
- **Residual:** \(r = C^{\mathrm{act}} / \hat{C}\) در window \(W=200\)
- **Quantile:** split-conformal با rank correction **(n+1)** — نه linear interpolation (under-cover می‌کند)
- **Lower bound:** \(\underline{C}_k = \hat{C}_k \cdot q_\alpha\)
- **Claim:** تحت **exchangeability** → \(\mathbb{P}(C^{\mathrm{act}} \ge \underline{C}_k) \ge 1-\alpha\)

**پارامتر پیش‌فرض:** \(\alpha=0.10\) → target coverage **0.90**

#### (B) Certified feasibility (Safety set)
\[
\mathcal{A}_{\mathrm{safe},k}(a) = \{ j \le a : d_k(j) \le \max(B_k - m, 0.1) \}
\]
- \(d_k(j)\): زمان دانلود rung \(j\) **تحت** \(\underline{C}_k\)
- \(m = 0.5\) s margin
- اگر \(B_k \le B_{\mathrm{crit}}=0.3\) s → force lowest rung

#### (C) VMAF-knee bandwidth banking
\[
j^{\mathrm{knee}}_k(a) = \min\{ j \le a : V_k(a) - V_k(j) \le \varepsilon \}
\]
- \(\varepsilon = 1.0\) VMAF point (بودجه perceptual)
- **حتی اگر proposal feasible باشد**، به knee می‌رود
- سپس روی \(\mathcal{A}_{\mathrm{safe}}\) project می‌شود — **safety همیشه اولویت دارد**

**Intuition:** rungهای بالا VMAF تقریباً یکسان → پایین آمدن = bitrate کمتر + افت کیفیت ناچیز → bytes bank

### ۲.۳ Proposition 1 — Stall certificate (Prop. feasibility)
- اگر \(d_k(j) \le B_k - m\) و \(C^{\mathrm{act}} \ge \underline{C}_k\) → **stall این chunk = 0**
- Coverage **marginal** است، نه conditional روی buffer/regime
- **مهم:** certificate ≠ «هیچ‌وقت stall نمی‌شود» — فقط bound درست بودن throughput

### ۲.۴ Proposition 2 — Ranking collapse (Prop. collapse)
- اگر ladder VMAF **monotonic** و download time **monotonic**:
- Shield که «max VMAF among feasible» را برگرداند = **highest feasible index**
- بودجه \(\varepsilon\) **هیچ اثری ندارد** → banking صفر
- **CPS با knee rule** این collapse را می‌شکند

### ۲.۵ Predictive extension (Improved shield)
- \(\varepsilon_{\mathrm{eff}}\) وقتی buffer < \(B_{\mathrm{risk}}=8\) s یا look-ahead \(H\ge6\) dip پیش‌بینی کند
- \(\varepsilon_{\mathrm{risk}} = 4.0\) VMAF
- **Pre-banking** قبل از dip

### ۲.۶ Co-design (supporting — headline نیست)
- PPO با CPS در training loop vs shield-at-eval-only
- **Single seed** — claim اصلی نیست
- crossover QoE weight \(w^\star \approx 5.31\)

---

## ۳. System model — چیزهایی که استاد می‌پرسد

| پارامتر | مقدار |
|---------|--------|
| Chunk duration \(T_c\) | 4 s |
| Chunks per episode \(K_e\) | 48 |
| Buffer cap | 12 s |
| Episodes (paired) | **204** |
| Videos | **12** (6 Blender + 6 Xiph DERF) |
| Ladder rungs \(L\) | 6 |
| QoE weights | \(\beta=4.3\) per stall s, \(\mu=1.0\) smoothness |
| Latency shield | median **131 µs**, p99 **310 µs** per decision |

### ۳.۱ سه arm (طراحی آزمایش)
| Arm | معنی |
|-----|------|
| **Raw** | بدون shield |
| **Safety** | فقط conformal feasibility |
| **Certified** | feasibility + VMAF-knee banking |

**Pairing:** همان seed و (video, trace) برای هر arm → تفاوت فقط از shielding است.

### ۳.۲ دو suite شبکه
| Suite | منبع | نقش |
|-------|------|-----|
| **Primary 5G** | synthetic (dip/outage) | claim اصلی banking |
| **Broadband** | FCC MBA + Norway/Nornet (27 trace) | regime boundary — banking collapse |

---

## ۴. Research Questions (RQ1–RQ6)

| RQ | سوال | پاسخ کوتاه |
|----|------|------------|
| **RQ1** | Banking vs Safety: bitrate/reb down، VMAF در ±ε؟ | ✅ بله، همه hostهای deterministic |
| **RQ2** | Coverage ≥ 1−α؟ | ✅ 0.919 vs 0.90 روی 5G؛ broadband 0.890 |
| **RQ3** | Predictive banking بهتر؟ | ✅ ~−14.6% bitrate greedy |
| **RQ4** | Co-design چه می‌کند؟ | +2.93 VMAF، single-seed supporting |
| **RQ5** | Per-chunk non-monotone ladder؟ | knee sound؛ per-chunk بیشتر bank می‌کند |
| **RQ6** | Sensitivity ε, α؟ | banking با ε scale؛ coverage با α track |

---

## ۵. اعداد کلیدی — حفظ کن

### ۵.۱ Certified vs Safety (5G synthetic) — headline
| Host | Δ Bitrate | Δ Rebuffering | Δ VMAF (TOST ±1.0) |
|------|-----------|---------------|---------------------|
| Greedy | **−4.5%** | **−9.3%** | −0.01 |
| BBA | −4.1% | −8.9% | +0.17 |
| BOLA | −4.1% | −8.9% | +0.18 |
| RobustMPC | −3.8% | −9.2% | +0.02 |

- Wilcoxon rebuffering greedy: \(p = 8.5\times10^{-4}\)
- Empirical coverage: **0.919** (همه hostها روی 5G یکسان — وابسته به trace نه policy)

### ۵.۲ Certified vs Raw (greedy) — نقش safety
- Rebuffering cut: **−55.5%** (Safety conformal بخش بزرگ stall را حذف می‌کند؛ banking بقیه headroom)

### ۵.۳ Absolute numbers (greedy, Table 5)
| Arm | Rebuffer (s) | VMAF | Bitrate (kb/s) | Coverage | Intervention |
|-----|--------------|------|----------------|----------|--------------|
| Raw | 85.5 | 89.1 | 6000 | — | 0% |
| Safety | 41.9 | 79.4 | 3131 | 0.919 | 65.8% |
| Certified | 38.0 | 79.4 | 2990 | 0.919 | 70.4% |

### ۵.۴ Predictive (Improved shield)
| Host | Δ Bitrate | Δ Rebuffering |
|------|-----------|---------------|
| Greedy | **−14.6%** | **−12.6%** |
| BBA | −12.5% | −13.0% |

### ۵.۵ Broadband (real traces)
- Banking Certified vs Safety: **−0.9%** bitrate (collapse)
- Safety vs Raw rebuffering: **−98.7%**
- Coverage: **0.890** (زیر target 0.90)

### ۵.۶ Saturation split (greedy Certified vs Safety)
| Ladder type | Episodes | Δ Bitrate | Δ Rebuffering |
|-------------|----------|-----------|---------------|
| Top gain ≤ 2 VMAF (saturated) | n=60 | **4.6%** | **9.9%** |
| Top gain ≥ 9 VMAF (steep) | n=51 | 2.1% | 3.7% |

### ۵.۷ PPO content-aware (primary learned host)
- Certified vs Safety: **−35.3%** bitrate, **−79.3%** rebuffering
- Improved: −21.9% / −29.6%

### ۵.۸ Co-design (supporting)
| | Shield-at-eval | Co-trained | Δ |
|--|----------------|------------|---|
| VMAF | 72.80 | 75.73 | **+2.93** |
| Rebuffer (s) | 0.31 | 0.86 | +0.55 |
| Bitrate (kb/s) | 801 | 1463 | — |
| \(w^\star\) | — | — | **5.31** |

### ۵.۹ Per-chunk ablation (RQ5)
- Inverted chunks: **26.5%**
- Banked bitrate: pooled 44.6% vs per-chunk **45.8%**
- Pooled budget violation: **11.5%** of chunks
- Per-chunk cost: **0.32** VMAF (≤ ε)

### ۵.۱۰ Sensitivity (RQ6)
- ε sweep 0.5→4.0: bitrate saved 2.0%→24.6%; coverage **invariant**
- α sweep: coverage 0.945 (α=0.05) → **0.919** (α=0.10) → 0.854 (α=0.20)

### ۵.۱۱ Coverage under drift (injected shock)
- Chunks 28–37: throughput × 0.3 (LOS/NLOS dip)
- Pre-shock coverage: 0.928 → during 0.929 → post 0.944 (target 0.90)

---

## ۶. Contributions (۵ bullet مقاله)

1. **CPS:** banking + conformal + certified feasibility (model-agnostic wrapper)
2. **Isolated evaluation:** Raw/Safety/Certified paired arms + Wilcoxon + TOST
3. **Predictive banking:** risk-aware ε + look-ahead
4. **Regime characterization:** 5G vs broadband — banking regime-dependent
5. **Co-design (supporting):** optional PPO + CPS in loop

---

## ۷. محدودیت‌ها — صادقانه بگو

| Threat | توضیح برای استاد |
|--------|------------------|
| Synthetic 5G | trace واقعی packet-level نیست؛ emulator chunk-level |
| 12 titles | diversity محدود؛ clustering by title |
| VMAF surrogate | subjective QoE ممکن است diverge کند |
| Exchangeability | drift → coverage ممکن است بشکند (drift test نشان داد sliding window کمک می‌کند) |
| Co-design | single seed — training variance گزارش نشده |
| TOST margin = ε | engineering choice، نه clinical equivalence |
| Pensieve negative control | policy conservative → banking فرصت ندارد |

---

## ۸. سوالات محتمل استاد + پاسخ آماده

### Q1: «چرا conformal prediction؟ چرا نه fixed safety factor؟»
**پاسخ:** fixed factor (مثلاً ρ=0.8) coverage guarantee ندارد و hand-tuned است. Conformal با finite-sample (n+1) correction، **distribution-free** coverage 1−α تحت exchangeability می‌دهد — auditable و data-driven.

### Q2: «Certificate یعنی stall نمی‌شود؟»
**پاسخ:** **نه.** Certificate = \(\mathbb{P}(C^{\mathrm{act}} \ge \underline{C}_k) \ge 1-\alpha\). اگر bound درست باشد و feasibility پاس شود، آن chunk stall نمی‌شود. ولی ~8% chunks ممکن است throughput زیر bound بیاید (coverage 0.919). همچنین cold start و \(B_{\mathrm{crit}}\) force-downshift باعث residual rebuffering می‌شود.

### Q3: «چرا knee rule و نه maximize VMAF among feasible؟»
**پاسخ:** Prop. collapse: روی monotone ladder، max-VMAF feasible = highest index → **banking صفر**. Knee از proposal پایین می‌آید و saturation را exploit می‌کند.

### Q4: «ε=1.0 از کجا آمد؟»
**پاسخ:** engineering design aligned با eq. knee و TOST margin. یک VMAF point کسری از gap بین rungهای adjacent است (Table ladder spacing). Conservative نسبت به perceptual encoding practice — preregistered clinical margin نیست.

### Q5: «چرا 204 episode؟ paired design چرا؟»
**پاسخ:** paired Wilcoxon/TOST روی همان (video, trace, seed) → effect از shielding isolate می‌شود. 204 = held-out test pool. caveat: effective content diversity = 12 title نه 204.

### Q6: «چرا banking روی broadband collapse شد؟»
**پاسخ:** top rungs rarely **jointly feasible and saturated** — knee خارج از operating range. Conformal safety هنوز stall را کم می‌کند (−98.7% vs raw). **Regime-dependent** by design.

### Q7: «Coverage 0.919 vs target 0.90 — خوب است؟ 0.890 broadband؟»
**پاسخ:** روی 5G synthetic بالاتر از target — finite-sample OK. Broadband 0.890 کمی زیر 0.90 — nonstationarity + n=204. α=0.05 case هم 0.945 vs 0.95 (marginally below). صادقانه در limitations گفته شده.

### Q8: «Intervention rate 70% greedy — shield همه چیز را override می‌کند؟»
**پاسخ:** **نه claim dominance.** Greedy همیشه top rung → هر downshift = intervention. Rate نشان می‌دهد host چقدر از certified operating point دور است. RobustMPC intervention ~31% — policy خودش conservative است.

### Q9: «Model-agnostic یعنی چی؟ retrain لازم نیست؟»
**پاسخ:** یک wrapper inference-time روی greedy/BBA/BOLA/MPC/PPO — بدون تغییر weights. Co-design **optional** است برای learned hosts.

### Q10: «PPO نتیجه −35% bitrate — چرا بیشتر از greedy؟»
**پاسخ:** learned policy هنوز often high rungs → saturation headroom بیشتر. Greedy همیشه top → Safety layer قبلاً heavily project کرده.

### Q11: «Synthetic 5G credible است؟»
**پاسخ:** public per-chunk 5G/mmWave logs scarce. Generator tuned to reported dip/outage dynamics. Broadband real traces as counterpoint. Future work: packet-level / field 5G.

### Q12: «Per-chunk ladder vs pooled — کدام در production؟»
**پاسخ:** headline runs pooled session-mean (monotone). Per-chunk دقیق‌تر، بیشتر bank می‌کند، violation صفر. Production نیاز به per-chunk VMAF feed دارد.

### Q13: «Complexity و real-time؟»
**پاسخ:** O(L+H) per decision، ~131 µs median — ~10⁻⁵ of 4s chunk. Inline feasible. Production challenges: noisy HTTP throughput, buffer signal delay, exchangeability under QUIC/TCP.

### Q14: «تفاوت با shielding قبلی (Alshiekh et al.)؟»
**پاسخ:** prior shields: decrement until feasible، content-blind، no conformal coverage، no VMAF-knee banking. CPS: formal coverage + exploit saturation + auditable parameters.

### Q15: «Co-design headline claim است؟»
**پاسخ:** **خیر.** Single seed، supporting study. Abstract و conclusion صریح: optional، not headline.

### Q16: «Holm–Bonferroni correction؟»
**پاسخ:** همه reported bitrate/rebuffering effects survive Holm–Bonferroni at 0.05 across host families.

### Q17: «چرا TOST و نه فقط t-test؟»
**پاسخ:** می‌خواهیم نشان دهیم VMAF **non-inferior** است within ±ε — equivalence testing، نه فقط «تفاوت معنی‌دار نیست».

### Q18: «Drift test چه نشان داد؟»
**پاسخ:** injected 0.3× shock chunks 28–37: rolling coverage زیر target نرفت (0.929 during shock). sliding window defense empirically کار می‌کند — theory: TV-bound under drift (Barber et al.).

### Q19: «چرا greedy/BBA/BOLA/MPC و Pensieve؟»
**پاسخ:** spectrum از aggressive (greedy) تا conservative (Pensieve). Pensieve = negative control — banking useless when policy never requests saturated rungs.

### Q20: «Contribution اصلی برای journal چیست؟»
**پاسخ:** ترکیب **certified safety** (conformal) + **perceptual banking** (VMAF-knee) در یک wrapper سبک، با isolated paired evaluation و regime characterization — نه «بهترین ABR algorithm».

---

## ۹. چیزهایی که **نباید** بگویی

- ❌ «CPS همیشه بهتر از همه ABRهاست»
- ❌ «هیچ stallی رخ نمی‌دهد»
- ❌ «co-design با confidence کامل claim می‌کنیم»
- ❌ «5G results = real network guaranteed»
- ❌ «coverage conditional روی هر buffer state»
- ❌ intervention rate بالا = shield bad

---

## ۱۰. روایت‌های آماده

### ۱۰.۱ یک دقیقه
ما CPS را برای ABR client-side ساختیم: wrapper بین policy و player. سه جزء — conformal lower bound برای throughput، certified feasibility برای buffer، و VMAF-knee banking برای exploit saturation. روی 204 paired episode و 12 video، banking نسبت به safety-only حدود 4–5% bitrate و 9% rebuffering کم کرد بدون افت VMAF beyond ±1. Coverage 0.919. روی broadband واقعی banking collapse شد ولی safety مفید ماند. Payoff regime-dependent است.

### ۱۰.۲ سه دقیقه (ساختار)
1. **Motivation** (30s): per-chunk risk + saturation waste  
2. **CPS design** (60s): 3 components + collapse proposition  
3. **Evaluation** (45s): 3 arms, 204 paired, 12 titles  
4. **Results** (45s): numbers 5G + broadband contrast  
5. **Honest limits** (30s): synthetic, exchangeability, co-design supporting  
6. **Take-home** (10s): lightweight certifiable wrapper, no retrain required

---

## ۱۱. نقشه مقاله (اگر بپرسد «کجا چیست؟»)

| Section | محتوا |
|---------|--------|
| §1 Intro | problem, CPS overview, contributions |
| §2 Related | ABR, RL, shielding, conformal — positioning table |
| §3 System | notation, arch fig, ladder saturation |
| §4 Method | conformal, knee, algorithm, predictive, co-design |
| §5 Eval | RQ1–6, tables/figures |
| §6 Limits | threats, inversions, future (dash.js deployment) |
| §7 Conclusion | summary |
| Appendix | CMDP for PPO training |

---

## ۱۲. Figure/Table برای ارجاع شفاهی

| Ref | چه نشان می‌دهد |
|-----|----------------|
| Fig. arch | control loop raw → CPS → exec |
| Fig. overview | banking 5G vs broadband collapse |
| Table 5 (full) | Raw/Safety/Certified همه hostها |
| Fig. coverage | RQ2 — host-invariant coverage |
| Fig. tradeoff / CDF | per-episode operating points |
| Fig. codesign | w* crossover (supporting) |
| Fig. perchunk | RQ5 pooled vs per-chunk |
| Fig. ablation | RQ6 ε and α sensitivity |
| Fig. coverage_drift | shock test |
| Table ladder spacing | saturation diversity |

---

*آخرین sync با `overleaf_upload/main.tex` و macros v18 — 2026-09-10*
