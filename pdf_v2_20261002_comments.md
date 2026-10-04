# PDF Comments: `Conformal_Perceptual_Shielding_for_Adaptive_Bitrate_Streaming__20261002-(v2).pdf`

- **Pages:** 16

## Page 1

### 1. **Text** · ASUS

> NOTE: Adaptive Bitrate

---

### 2. **Text** · ASUS

> NOTE: ارتباط بین chunk, request, rung
مبهم است

---

### 3. **Text** · ASUS

> NOTE: solutions for these challenges مثلا
repairs به نظرم فعل آمد

---

### 4. **Text** · ASUS

> NOTE: coverage برای کسی که قبلا مقاله مشابه ندیده باشد مشخص نیست چیه

---

### 5. **Highlight** · ASUS

> TEXT: executed rung

---

### 6. **Highlight** · ASUS

> TEXT: least 0.1 s

---

### 7. **Text** · ASUS

> NOTE: منظور چی هست؟

---

### 8. **Text** · ASUS

> NOTE: تعریف نشده تا حالا

---

### 9. **Highlight** · ASUS

> TEXT: savings versus safety at the default budget; a larger static ε=2 cuts rebuffering further. On real fixed and mobile broadband traces, banking gains fall to about 0.9%. A feasibility projection removes about 98.7% of stall time versus unshielded greedy, which already stalls for most of those sessions. Banking therefore pays off where ladders saturate. The broadband stall cut is a feasibility effect, not evidence that the conformal bound transfers.

> NOTE: کلا چی شد؟

---

### 10. **Text** · ASUS

> NOTE: we three contributions مثلا

---

### 11. **Text** · ASUS

> NOTE: در چکیده نیست

---

### 12. **Text** · ASUS

> NOTE: Adaptive Bitrate

---

### 13. **Text** · ASUS

> NOTE: Quality of Experience

---

### 14. **Text** · ASUS

> NOTE: Buffer-Based Adaption

---

### 15. **Text** · ASUS

> NOTE: buffer rules یعنی؟؟

---

### 16. **Text** · ASUS

> NOTE: used بقیه افعال ماضی

---

### 17. **Highlight** · ASUS

> TEXT: plans

> NOTE: planed

---

### 18. **Text** · ASUS

> NOTE: مرجع

---

### 19. **Highlight** · ASUS

> TEXT: representation index

> NOTE: دفعه اول است امده

---

### 20. **Highlight** · ASUS

> TEXT: CPS

> NOTE: اینجا CPS گفتید پاراگراف بعد گفتید 
We introduce ...

---

### 21. **Highlight** · ASUS

> TEXT: ,

> NOTE: ویرگول بردارید

---

### 22. **Highlight** · ASUS

> TEXT: calib=20

> NOTE: تعریف نشده

---

### 23. **Highlight** · ASUS

> TEXT: Bk ≤Bcrit

> NOTE: ؟؟؟؟؟

---

## Page 2

### 1. **Highlight** · ASUS

> TEXT: Bk ≥0.1 s

---

### 2. **Highlight** · ASUS

> TEXT: highest feasible rung, so a perceptual-loss budget cannot change its choice. The same projection attaches to bitrate-greedy, BBA,

> NOTE: متوجه نشدم؟

---

### 3. **Highlight** · ASUS

> TEXT: Raw, Safety

> NOTE: معرفی کنید چی هستند

---

### 4. **Highlight** · ASUS

> TEXT: two one-sided tests

> NOTE: Two One-sided Tests

---

### 5. **Highlight** · ASUS

> TEXT: a larger static bu

> NOTE: ارتباط با جمله قبلی

---

### 6. **Highlight** · ASUS

> TEXT: shield-at-eval-only

---

## Page 1 full text (for Abstract mapping)

```
Conformal Perceptual Shielding for Adaptive Bitrate Streaming
Saeed Zarbia, Leili Farzinvasha,∗, Pedram Salehpoura
aDepartment of Computer Engineering, Faculty of Electrical and Computer Engineering, University of Tabriz, Tabriz, Iran
Abstract
Client-side adaptive bitrate (ABR) controllers decide chunk by chunk. A single request can stall playback when throughput drops,
or waste bandwidth when the top rungs of the bitrate ladder are perceptually saturated. Standard runtime repairs address neither
problem: they ignore perceptual saturation and offer no coverage guarantee. We propose the Certified Perceptual Shield (CPS), a
model-agnostic projection that wraps heuristic, model-based, or learned ABR policies. First, knee-based bandwidth banking lowers
even a feasible proposal to the smallest rung whose Video Multimethod Assessment Fusion (VMAF) score lies within a perceptual
budget of the proposal. Second, an online split-conformal lower bound on throughput provides distribution-free coverage under
residual exchangeability. Third, a feasibility check prefers rungs whose download time under that bound fits the buffer margin. If
none does, or if the buffer is below a critical level, CPS still emits the lowest rung. The stall statement is then a per-chunk probabilistic
certificate only when the executed rung meets the test and the buffer is at least 0.1 s, not a zero-stall promise. We evaluate CPS on
204 paired synthetic 5G episodes over twelve titles with diverse rate–quality ladders. Compared with a safety-only shield, banking
reduces delivered bitrate by 3.8–4.5% and rebuffering by 8.9–9.3% for four deterministic controllers. Mean VMAF stays within one
point (equivalence test). Empirical coverage is 0.919 against a 0.90 target, and the first 20 of 48 chunks in each episode use a fixed
fallback scale. As a supporting mechanism study, risk-aware predictive banking (CPS-P) reaches 14.6% bitrate and 12.6% rebuffering
savings versus safety at the default budget; a larger static ε=2 cuts rebuffering further. On real fixed and mobile broadband traces,
banking gains fall to about 0.9%. A feasibility projection removes about 98.7% of stall time versus unshielded greedy, which already
stalls for most of those sessions. Banking therefore pays off where ladders saturate. The broadband stall cut is a feasibility effect, not
evidence that the conformal bound transfers.
Keywords: adaptive bitrate streaming, runtime shielding, VMAF, conformal prediction, certified safety, video QoE
1. Introduction
Client-side adaptive bitrate (ABR) policies target average
quality of experience (QoE). Several controller families pursue
this target. Buffer-based adaptation (BBA) [1] and BOLA [2]
use buffer or Lyapunov rules. RobustMPC [3] plans over a short
throughput horizon. Bitrate-greedy selection always requests the
top rung, and proximal policy optimization (PPO) [4, 5] learns
a policy from traces. Session averages, however, can obscure
transient risks associated with individual chunks. A single seg-
ment can still stall under volatile sub-6 GHz and mmWave 5G
throughput [6–8]. A comfortable buffer creates the opposite
problem: wasted bandwidth. Commercial rate ladders often
saturate in Video Multimethod Assessment Fusion (VMAF) [9]:
further bitrate increases at the top often provide only marginal
perceptual gains. Requesting those rungs when the buffer is
healthy consumes bandwidth that could cushion the next dip.
A runtime shield sits between the base policy and the
player [10–12]. It checks each proposed representation before
download and replaces an unsafe proposal. The standard repair
decrements the representation index until a buffer-feasibility
test passes. That repair is content-blind, because it ignores
∗Corresponding author.
Email address: l.farzinvash@tabrizu.ac.ir (Leili Farzinvash)
VMAF saturation. Prior shields often apply a fixed throughput
scale without a stated coverage guarantee. CPS uses the same
kind of fallback scale only during a short warm-up (Wcalib=20
chunks), then switches to an online split-conformal lower bound
with finite-sample coverage. Deployed ladders carry exploitable
structure—rate–quality saturation near the top. That structure
supports perceptually negligible bitrate reductions while freeing
additional buffer capacity.
We introduce a Certified Perceptual Shield (CPS) that exploits
this structure while stating a formal coverage guarantee:
1. VMAF-knee bandwidth banking. Even a feasible action
is lowered to the smallest rung whose VMAF lies within a
budget ε, in VMAF points, of the proposal. The downshift
is perceptually bounded, and reclaimed bytes bank as buffer
for later chunks.
2. Conformal throughput lower bound. Download-time
tests use an online split-conformal lower bound on through-
put. Under residual exchangeability, the bound covers the
next chunk with probability at least 1−α, where α is the
tolerated miss rate. Calibration uses recent residual ra-
tios [13, 14].
3. Certified feasibility. After banking, CPS prefers a rung
whose download time under the lower bound fits the buffer
margin. If no such rung exists, or if Bk ≤Bcrit, it still emits
the lowest rung. When the executed rung meets the test

```

Total annotations: 29