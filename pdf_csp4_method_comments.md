# PDF: CSP-4.pdf
- Pages: 16

## Page 5
_head:_ Table 3: Session-mean representation ladder across twelve evaluation titles. | Video | Held-out | SI | TI | Top gain | Span | Mono. | Big Buck Bunny | —

### 1. **Highlight** y=402.2
> TEXT: A base policy proposes a ladder rung; CPS applies a deter- ministic projection before download; the player then updates the buffer under realized throughput. CPS is a model-agnostic

> NOTE: برود بعد از علامت صورتی

> NEAR: decisions use session-mean ladders, but per-chunk VMAF can | invert locally. Chunk-level inversions appear in Fig. 2 for the | logged subset. Banking does not depend on them; Section 5.5 | shows that the knee rule remains effective and banks more when | the shield observes the raw per-chunk ladder. | 40 | 60 | 80 | 100

---

### 2. **Highlight** y=438.0
> TEXT: .

> NEAR: shows that the knee rule remains effective and banks more when | the shield observes the raw per-chunk ladder. | 300 | 750 | 1200 | 1850 | 2850 | 6000 | 20 | 40 | 60 | 80 | 100 | VMAF | (a) Per-video ladder vs. per-chunk range | 00→750 | 0→1200 | 0→1850 | 0→2850 | 0→6000 | 0 | 10 | 20 | 30 | Per-chunk VMAF gap | (b) Sp

---

### 3. **StrikeOut** y=485.9
> TEXT: that holds with probability at least 1−α, where α is the tolerated miss rate. The rest of this section covers the conformal estimator,

> NOTE: قبلا بود

> NEAR: 300 | 750 | 1200 | 1850 | 2850 | 6000 | Representation bitrate (kb/s) | 20 | 40 | 60 | VMAF | 300→750 | 750→1200 | 1200→1850 | 1850→2850 | 2850→6000 | Adjacent representation pair (kb/s) | 0 | 10 | Per-chunk VMA | Big Buck Bunny | Tears of Steel | Sintel | Figure 2: Per-chunk VMAF ladders on three titles with frame log

---

### 4. **StrikeOut** y=509.8
> TEXT: (Algorithm 1)

> NEAR: 300 | 750 | 1200 | 1850 | 2850 | 6000 | Representation bitrate (kb/s) | 20 | 300→750 | 750→1200 | 1200→1850 | 1850→2850 | 2850→6000 | Adjacent representation pair (kb/s) | 0 | Pe | Big Buck Bunny | Tears of Steel | Sintel | Figure 2: Per-chunk VMAF ladders on three titles with frame logs (Big Buck | Bunny, Tears of Ste

---

### 5. **Highlight** y=572.5
> TEXT: s, RobustMPC-style

> NOTE: بازنویسی شود

> NEAR: Bunny, Tears of Steel, Sintel; 171 chunks). (a) Session-mean pooled ladder | (markers) and per-chunk min–max range (shaded). (b) Adjacent-rung VMAF | gaps; the dashed line marks zero gain. Section 5.5 uses 113 chunks from a | separate per-chunk export for the banking ablation. | 3.4. Chunk-timescale dynamics | 3.4.1. T

---

### 6. **Highlight** y=596.5
> TEXT: window of size W. Given n residuals, the split-conformal α- quantile uses the finite-sample rank ⌊α(n+1)⌋(1-based); the (n+1) term is the standard finite-sample correction. We avoid a

> NOTE: اصلا متوجه نشدم

> NEAR: separate per-chunk export for the banking ablation. | 3.4. Chunk-timescale dynamics | 3.4.1. Throughput and latency | Let Ctr | k > 0 be the trace-derived effective throughput for | chunk k. Section 4 defines predictor ˆCk (harmonic mean of | recent realizations); the conformal construction yields lower | bound Ck fed 

---

### 7. **Highlight** y=737.1
> TEXT: σk(j  103

> NOTE: 1000و   
60 از کجا آمده اند؟

> NEAR: k |   | k  | for a small constant ε0>0 (the same numerical guard as in (5)), | and dk( j) = σk( j)/(103 ˜Ck) for the nominal download duration of | rung j. The executed chunk uses dk ≡dk(aexec | k | ) [3]. Feasibility | timing under Ck is defined in Section 4. | Ck) ≥1 −α once the residual window has length at least Wc

---

## Page 6
_head:_ Proposition 1 (Per-chunk stall certificate). Fix chunk k, exe- | cuted index j, buffer Bk ≥0.1 s, and conformal lower bound | Ck computed with miscoverage α. If dk(j) ≤max{Bk −m, 0.1} | via (5) and the realized throughpu

### 1. **Highlight** y=84.3
> TEXT: (Per-chunk stall certificate).

> NOTE: بولد شود

> NEAR: Proposition 1 (Per-chunk stall certificate). Fix chunk k, exe- | cuted index j, buffer Bk ≥0.1 s, and conformal lower bound | Ck computed with miscoverage α. If dk(j) ≤max{Bk −m, 0.1} | via (5) and the realized throughput satisfies Cact | k | ≥Ck, then | τstall | k | = 0. The buffer floor is required: max{Bk −m, 0.1} ≤

---

### 2. **Text** y=85.2
> TEXT: . Fix  ormal

> NOTE: Assume chunk ...

> NEAR: Proposition 1 (Per-chunk stall certificate). Fix chunk k, exe- | cuted index j, buffer Bk ≥0.1 s, and conformal lower bound | Ck computed with miscoverage α. If dk(j) ≤max{Bk −m, 0.1} | via (5) and the realized throughput satisfies Cact | k | ≥Ck, then | τstall | k | = 0. The buffer floor is required: max{Bk −m, 0.1} ≤

---

### 3. **Highlight** y=132.2
> TEXT: The buffer floor is required:

> NOTE: the condition ... should be satisfied only when ...
شرط سوم است؟

> NEAR: cuted index j, buffer Bk ≥0.1 s, and conformal lower bound | Ck computed with miscoverage α. If dk(j) ≤max{Bk −m, 0.1} | via (5) and the realized throughput satisfies Cact | k | ≥Ck, then | τstall | k | = 0. The buffer floor is required: max{Bk −m, 0.1} ≤Bk | only when Bk ≥0.1. Under exchangeability of conformal resid-

---

### 4. **Highlight** y=198.4
> TEXT: (4) gi

> NOTE: یا 3؟

> NEAR: bound is defined (n ≥Wcalib). | Proof sketch. When Cact | k | ≥Ck and Bk ≥0.1 s, the realized | download time is at most dk(j) ≤Bk; hence (4) gives τstall | k | = 0. | Coverage follows from standard split-conformal prediction with | the (n+1) rank correction [18, 19]. | Scope of the guarantee. Two parts of the certific

---

### 5. **Highlight** y=279.6
> TEXT: throughput then clears

> NOTE: ؟؟؟

> NEAR: Scope of the guarantee. Two parts of the certificate behave | differently. The feasibility test is deterministic only when the | executed rung satisfies dk( j) ≤max{Bk −m, 0.1} and Bk ≥0.1 s. | If the realized throughput then clears Ck, that chunk cannot | stall. Algorithm 1 can still return rung 0 when Bk ≤Bcrit, or |

---

### 6. **Highlight** y=291.6
> TEXT: return

> NOTE: get 
کد نیست شبه کد است

> NEAR: differently. The feasibility test is deterministic only when the | executed rung satisfies dk( j) ≤max{Bk −m, 0.1} and Bk ≥0.1 s. | If the realized throughput then clears Ck, that chunk cannot | stall. Algorithm 1 can still return rung 0 when Bk ≤Bcrit, or | when the decrement loop reaches 0 while dk(0) fails the test.

---

### 7. **Highlight** y=291.6
> TEXT: Algorithm 1

> NOTE: قبل از استفاده باید اشاره شود چی هست

> NEAR: differently. The feasibility test is deterministic only when the | executed rung satisfies dk( j) ≤max{Bk −m, 0.1} and Bk ≥0.1 s. | If the realized throughput then clears Ck, that chunk cannot | stall. Algorithm 1 can still return rung 0 when Bk ≤Bcrit, or | when the decrement loop reaches 0 while dk(0) fails the test.

---

### 8. **Highlight** y=411.2
> TEXT: ssed the test;

> NOTE: احتمال fail چقدر هست؟

> NEAR: or throughput regime. Distribution-free split conformal cannot | do so without stronger assumptions [19]. CPS therefore certifies | feasibility on every chunk whose bound holds and whose rung | passed the test; it does not promise a stall-free session. | Abrupt 5G/mmWave LOS/NLOS and scheduling shifts can | break resid

---

### 9. **Highlight** y=119.4
> TEXT: satisfies

> NOTE: این دو شرط دو رابطه بنویسید که مشخص تر باشند بهتر است

> NEAR: Proposition 1 (Per-chunk stall certificate). Fix chunk k, exe- | cuted index j, buffer Bk ≥0.1 s, and conformal lower bound | Ck computed with miscoverage α. If dk(j) ≤max{Bk −m, 0.1} | via (5) and the realized throughput satisfies Cact | k | ≥Ck, then | τstall | k | = 0. The buffer floor is required: max{Bk −m, 0.1} ≤

---

### 10. **Highlight** y=155.3
> TEXT: P(Cact k ≥Ck)

> NOTE: این به stall ربط ندارد موضوع جداگانه است

> NEAR: via (5) and the realized throughput satisfies C | k | ≥Ck, then | τstall | k | = 0. The buffer floor is required: max{Bk −m, 0.1} ≤Bk | only when Bk ≥0.1. Under exchangeability of conformal resid- | uals, P(Cact | k | ≥Ck) ≥1 −α at each step where the finite-sample | bound is defined (n ≥Wcalib). | Proof sketch. When C

---

### 11. **Highlight** y=590.5
> TEXT: (0.919 vs. 0.90).

> NOTE: یعنی چه؟

> NEAR: as the operational check of Proposition 1. That rate averages over | every chunk with a defined bound, including the Wcalib warm-up | steps that use ρfb. On the primary 5G pool, the certified arm | meets 1 −α (0.919 vs. 0.90). Real broadband traces and tighter | α can sit marginally below the target (Section 5.3). This

---

### 12. **Highlight** y=289.6
> TEXT: Remark.

> NOTE: برود خط بعدی

> NEAR: differently. The feasibility test is deterministic only when the | executed rung satisfies dk( j) ≤max{Bk −m, 0.1} and Bk ≥0.1 s. | If the realized throughput then clears Ck, that chunk cannot | stall. Algorithm 1 can still return rung 0 when Bk ≤Bcrit, or | when the decrement loop reaches 0 while dk(0) fails the test.

---

### 13. **Highlight** y=393.2
> TEXT: Fix

> NOTE: Assume

> NEAR: up. It does not provide coverage conditional on the current buffer | or throughput regime. Distribution-free split conformal cannot | do so without stronger assumptions [19]. CPS therefore certifies | feasibility on every chunk whose bound holds and whose rung | passed the test; it does not promise a stall-free session

---

### 14. **Highlight** y=484.6
> TEXT: jmax = max F .

> NOTE: maxF کلا اضافی است
محاسبه ندارد فقط تعریف است

> NEAR: sliding residual window of length W. The recent ratios play | two roles: they calibrate qα, and they are the ratios immediately | preceding the chunk we predict. The bound therefore tracks | local nonstationarity, even though finite-sample exactness holds | only on an exchangeable window. If drift is large, coverage | 

---

### 15. **Highlight** y=496.8
> TEXT: arg maxj∈F Vk( j) = jmax; the budget ε never enters the maxi- mization.

> NOTE: ؟؟؟

> NEAR: two roles: they calibrate qα, and they are the ratios immediately | preceding the chunk we predict. The bound therefore tracks | local nonstationarity, even though finite-sample exactness holds | only on an exchangeable window. If drift is large, coverage | decays. The decay is bounded by a total-variation term between

---

### 16. **Highlight** y=564.5
> TEXT: within ε of it (7)

> NOTE: جمله اصلاح شود
(Equation (7))

> NEAR: that down-weights stale residuals is a natural extension, which | we leave to future work. Empirically, we treat reported coverage | as the operational check of Proposition 1. That rate averages over | every chunk with a defined bound, including the Wcalib warm-up | steps that use ρfb. On the primary 5G pool, the certi

---

### 17. **StrikeOut** y=660.1
> TEXT: (5)

> NEAR: traces both erode finite-sample coverage. | 4.2. Banking and certified feasibility | Safety set. Indices that never exceed the proposal and fit the | buffer margin m form | Asafe,k(a) = {︁j ≤a : dk(j) ≤max{Bk −m, 0.1}}︁. | (6) | lowest rung with no feasibility test. Otherwise it computes the | knee index from (7) under

---

### 18. **Highlight** y=672.1
> TEXT: ,

> NOTE: (Equation (5))

> NEAR: 4.2. Banking and certified feasibility | Safety set. Indices that never exceed the proposal and fit the | buffer margin m form | Asafe,k(a) = {︁j ≤a : dk(j) ≤max{Bk −m, 0.1}}︁. | (6) | VMAF-knee banking. The perceptual budget ε is a tolerance | knee index from (7) under the effective budget εeff. This budget | equals ε

---

### 19. **Highlight** y=684.1
> TEXT: the algorithm returns it anyway. The certificate in Proposition 1 does not cover that return, nor the Bcrit return. The shield never

> NOTE: پس چکار می کند؟

> NEAR: 4.2. Banking and certified feasibility | Safety set. Indices that never exceed the proposal and fit the | buffer margin m form | Asafe,k(a) = {︁j ≤a : dk(j) ≤max{Bk −m, 0.1}}︁. | (6) | VMAF-knee banking. The perceptual budget ε is a tolerance | in VMAF points. Even when a is feasible, CPS steps down to | equals ε unles

---

## Page 7
_head:_ Evaluation arms.. Raw: we run araw unshielded. Safety: Algo- | rithm 1 with banking switched off (jknee=a), leaving conformal | feasibility alone. Certified: the full CPS, banking plus feasibil- | ity. Paired seeds keep 

### 1. **Highlight** y=84.4
> TEXT: Evaluation arms.. Raw

> NOTE: مال کجاست اینجا آمده؟

> NEAR: Evaluation arms.. Raw: we run araw unshielded. Safety: Algo- | rithm 1 with banking switched off (jknee=a), leaving conformal | feasibility alone. Certified: the full CPS, banking plus feasibil- | ity. Paired seeds keep the Wilcoxon and TOST tests valid across | arms. | Table 4: CPS default hyperparameters (primary 5G 

---

### 2. **Highlight** y=255.0
> TEXT: Brisk,

> NOTE: بقیه جمله
در این دو حالت eff به صورت زیر محاسبه می شود

> NEAR: is a drop of Bk below a risk threshold Brisk. The second is a | short H-step look-ahead under a planning throughput floor (a | low quantile of recent observations) that forecasts a dip below | Brisk, | εeff = ε + (εrisk −ε) · f(Bk, lookahead), | (8) | where f ∈[0, 1] ramps between the two budgets and εrisk > ε is | the

---

### 3. **Highlight** y=320.5
> TEXT: (CPS-P)

> NOTE: فرق با CPS?
فقط در تعریف اپسیلون است؟

> NEAR: where f ∈[0, 1] ramps between the two budgets and εrisk > ε is | the wider risk budget. The shield can therefore pre-bank more | deeply, before a stall forces an emergency cut. The predictive- | banking experiments (CPS-P) use ε=1.0, εrisk=4, Brisk=8 s, and | H≥6 at forecast quantile 0.2. | 4.4. Optional learned hosts 

---

### 4. **Highlight** y=419.1
> TEXT: Co-design:

> NOTE: منظور از co-design مشخص نیست.
یعنی با learning ترکیب شود

> NEAR: policy, we train PPO (optionally Lagrangian-shaped) with Stable- | Baselines 3 [5, 43] on content-aware observations: throughput | history, buffer, spatial and temporal complexity (SI/TI), and the | VMAF ladder. Co-design: wrapping the training environment in | CPS changes the action distribution observed by the actor–

---

### 5. **Highlight** y=395.2
> TEXT: nes 3 [5

> NOTE: ؟؟

> NEAR: 4.4. Optional learned hosts and co-design | CPS does not require a learned policy. When the host is an RL | policy, we train PPO (optionally Lagrangian-shaped) with Stable- | Baselines 3 [5, 43] on content-aware observations: throughput | history, buffer, spatial and temporal complexity (SI/TI), and the | VMAF ladder. 

---

### 6. **Highlight** y=371.3
> TEXT: CPS does not require a learned policy. When the host is an RL policy, we train PPO (optionally Lagrangian-shaped) with Stable-

> NOTE: یعنی خودتان الگوریتم learning پیشنهاد می دهید دیگر؟
پس RL policy چیست؟
از الگوریتم دیگر استفاده نمی شود؟

> NEAR: H6 at forecast quantile 02. | 4.4. Optional learned hosts and co-design | CPS does not require a learned policy. When the host is an RL | policy, we train PPO (optionally Lagrangian-shaped) with Stable- | Baselines 3 [5, 43] on content-aware observations: throughput | history, buffer, spatial and temporal complexity (S

---

### 7. **Highlight** y=589.5
> TEXT: use

> NOTE: For the first ...., fallback is set to ...

> NEAR: Wcalib=20, fallback ρfb=0.80, predictor κ=5, margin m=0.5 s, | Bcrit=0.3 s, and buffer cap 12 s. Each episode has 48 chunks and | the residual window resets at the episode boundary, so W=200 | never fills. The first Wcalib chunks of every episode use ρfb. Ta- | ble 4 collects the shield and predictive-banking defaults 

---

### 8. **Highlight** y=685.1
> TEXT: wall time

> NOTE: ??

> NEAR: rungs and an H-step look-ahead. Conformal-quantile upkeep | on the length-W residual window is O(1) amortized. Across | 20,000 rollout decisions on a commodity CPU (L=6, full pre- | dictive shield), per-decision wall time is 131 µs at the median | and 310 µs at the 99th percentile. This is roughly 10−5 of the 4 s | chu

---

### 9. **Highlight** y=695.5
> TEXT: the

> NOTE: a

> NEAR: on the length-W residual window is O(1) amortized. Across | 20,000 rollout decisions on a commodity CPU (L=6, full pre- | dictive shield), per-decision wall time is 131 µs at the median | and 310 µs at the 99th percentile. This is roughly 10−5 of the 4 s | chunk budget. CPS therefore runs inline, and the control loop i

---

### 10. **Highlight** y=514.8
> TEXT: mplement

> NEAR: the shield (shield-at-eval-only), one co-trained with CPS. Both | face CPS-P at test time. | 4.5. Implementation notes | At the primary 5G operating point the defaults are α=0.10 | (coverage target 0.90), residual window W=200, warm-up | Wcalib=20, fallback ρfb=0.80, predictor κ=5, margin m=0.5 s, | Bcrit=0.3 s, and bu

---

### 11. **Highlight** y=514.8
> TEXT: Implementation

> NOTE: مقادیر برود بخش 5

> NEAR: the shield (shield-at-eval-only), one co-trained with CPS. Both | face CPS-P at test time. | 4.5. Implementation notes | At the primary 5G operating point the defaults are α=0.10 | (coverage target 0.90), residual window W=200, warm-up | Wcalib=20, fallback ρfb=0.80, predictor κ=5, margin m=0.5 s, | Bcrit=0.3 s, and bu

---

Total annotations: 37
