### 1. One-paragraph verdict

* **Likely outcome:** Major revision
* **Acceptance probability after one revision round:** 45–60%
* **Probability of desk reject or reject without revision:** 15%

The paper offers a theoretically interesting fusion of conformal prediction and VMAF-aware shielding. However, the core theoretical guarantee—conformal coverage—relies heavily on residual exchangeability, which is notoriously violated by the highly volatile nature of 5G networks. Furthermore, evaluating a 5G-targeted mechanism primarily on synthetic traces, while limiting real-world trace evaluation to older broadband datasets where the mechanism admittedly shows little benefit, weakens the empirical claims. The authors' transparency regarding these regime limitations is a strong positive for *Computer Networks*, making the manuscript salvageable. It requires a more robust evaluation on real 5G traces and a stronger defense of the exchangeability assumption to cross the acceptance bar.

### 2. What the paper actually contributes

* The authors present a model-agnostic runtime shield (CPS) that combines VMAF-knee bandwidth banking with a conformal throughput lower bound.


* The manuscript proves that ranking-based shields collapse to highest-feasible-index selection on monotone VMAF ladders, requiring a perceptual-loss budget to actually free bandwidth.


* The evaluation shows that VMAF-knee banking reduces bitrate and rebuffering over safety-only baselines on synthetic 5G traces.


* The authors explicitly identify a regime boundary, showing that banking benefits collapse almost entirely (-0.9% bitrate) on constrained real-measured broadband traces where top rungs are rarely feasible.


* **Incremental flag:** The PPO co-training evaluation is packaged as a supporting study but relies entirely on a single training seed. This renders the RL co-design claim statistically insignificant and incremental.



### 3. Claim–evidence audit

| Claim | Supported? | Evidence strength | Main risk |
| --- | --- | --- | --- |
| **Model-agnostic mechanism** | Yes | High: Tested successfully across greedy, BBA, BOLA, RobustMPC, and PPO hosts.

 | Low: The deterministic wrapper clearly functions independently of the host. |
| **Banking gains on 5G** | Partial | Medium: Shows -4.5% bitrate and -9.3% rebuffering for greedy hosts, but only on synthetic 5G traces.

 | High: Synthetic traces may exaggerate the frequency of saturation opportunities compared to real networks. |
| **Conformal coverage certificate** | Partial | Medium: Achieves 0.919 empirical coverage on synthetic 5G, but drops to 0.890 on real broadband against a 0.90 target.

 | Medium: The underlying assumption of residual exchangeability routinely breaks under abrupt real-world distribution shifts.

 |
| **Broadband boundary** | Yes | High: Clearly demonstrated using FCC and Norway/Nornet traces, showing virtually no banking headroom.

 | Low: This is an honest and well-supported limitation. |
| **Per-chunk robustness** | Yes | Medium: Ablation on 113 chunks confirms banking works on non-monotone ladders.

 | Low: The mechanism structurally supports inversions.

 |
| **Policy-shield co-design** | No | Very Low: Evaluated using a single PPO training seed.

 | Fatal: Deep RL requires multiple seeds to prove algorithmic significance. |

### 4. Reviewer attack surface (ranked)

* **FATAL: Residual Exchangeability in 5G.** A hostile reviewer will immediately attack the core conformal guarantee. 5G throughput exhibits bursty, non-stationary behaviors (LOS/NLOS shifts) that break the exchangeability assumption required for split-conformal prediction. The paper attempts to mitigate this using a sliding window $W=200$, but openly admits coverage decay under large drift.


* **MAJOR: Synthetic 5G Evaluation.** The paper justifies its banking mechanism based on 5G volatility but relies exclusively on *synthetic* 5G traces for its positive results. Reviewers will point out that public 5G datasets (e.g., Lumos5G) exist. The paper's mitigation—stating public logs "remain scarce"—is insufficient for a modern networking venue.


* **MAJOR: Single-seed RL Co-design.** The PPO co-training results are based on exactly one training seed per policy. Reviewers will flag this as statistically invalid for DRL evaluations. The authors try to mitigate this by framing it as a "supporting" rather than "headline" claim, but its inclusion invites unnecessary attacks.


* **MINOR: Session-mean vs. Per-chunk VMAF.** The primary evaluation uses pooled session-mean ladders. While the authors provide a 113-chunk ablation to prove viability on non-monotone per-chunk ladders, reviewers may argue the primary experiments should have utilized per-chunk ladders natively.



### 5. Novelty relative to closest prior work

The closest baselines are safe RL approaches like COREL, runtime shielding by Alshiekh et al., and VMAF-aware imitation learning like Comyco. The distinct delta here is the fusion of decision-time *conformal prediction* (for throughput pessimism) with *VMAF-knee banking* (for proactive buffer headroom). Prior constrained RL systems bake safety into training rather than runtime execution, and prior VMAF-aware systems use perceptual data for objective formulation rather than deterministic emergency projection.

**Is the delta "enough for Computer Networks"? Borderline Yes.** The architectural split between host and shield is highly practical. Furthermore, the critique showing that ranking-based shields inherently collapse to highest-feasible-index selection on monotone ladders (Proposition 2) is a sharp systems observation. However, the heavy reliance on synthetic traces to prove the mechanism's value significantly dilutes the impact.

### 6. Evaluation credibility

* **Experimental design:** 7/10. Good use of paired seeds across multiple baseline hosts, combined with TOST equivalence testing.


* **Statistics:** 6/10. Solid non-inferiority testing, but fails drastically on the single-seed RL claim.


* **Baselines:** 8/10. Thorough baseline coverage across heuristic (BBA), Lyapunov (BOLA), MPC (RobustMPC), and learned (Pensieve-style) hosts.


* **Generality:** 6/10. Effective diversity is restricted to 12 titles, and the results heavily depend on ladder steepness.


* **Realism of traces:** 4/10. The authors evaluate on older real broadband traces where the method yields no banking gains, and rely on synthetic 5G traces to demonstrate where it does.


* **Reproducibility:** 9/10. Hyperparameters, network architectures, and shield parameters are meticulously documented.


* **Flags:** The reliance on synthetic 5G traces to demonstrate banking gains borders on regime cherry-picking, especially since the broadband tests confirm the gains vanish when top rungs are generally infeasible.



### 7. Writing / structure / presentation

**Score: 8/10.**
The paper is well-structured and refreshingly candid about its limitations. However, sections 3.2 and 4.1 are dense and heavily rely on jargon. The distinction between the conformal coverage guarantee (which is marginal over the residual window) and the per-chunk guarantee (which is conditional) could easily be misinterpreted. Finally, the decision to include the unverified PPO co-design (Section 5.4) feels bolted-on and detracts from the core systems contribution.

### 8. Decision factors that would move the needle

**What would most increase acceptance odds:**

1. Evaluate the CPS mechanism on real, publicly available 5G datasets (e.g., Lumos5G) to prove the banking claims translate outside of synthetic simulations.
2. Remove Section 5.4 (PPO co-design) entirely; it is a distraction and a glaring target for rejection due to its lack of statistical rigor.
3. Provide a deeper empirical analysis of conformal coverage decay during sudden simulated 5G LOS/NLOS shifts to directly address the exchangeability flaw.
4. Transition the primary 204-episode evaluation from session-mean ladders to actual per-chunk ladders, as this is the exact regime where perceptual budgets actually bind.
5. Clarify early in the abstract and introduction that the stall "certificate" is a probabilistic and marginal guarantee, not an absolute one.

**What would most decrease acceptance odds if left as-is:**

* Leaving the synthetic 5G traces as the sole environment where the banking mechanism demonstrates substantive bitrate and rebuffering benefits.

### 9. Final scores

* **Novelty (7/10):** Clever intersection of conformal prediction and ABR shielding, though the component parts themselves are standard.
* **Technical soundness (6/10):** The conformal exchangeability assumption is theoretically shaky for 5G, and the single-seed RL evaluation lacks rigor.
* **Empirical evaluation (5/10):** Severely hampered by synthetic 5G traces and a limited 12-title video catalog.
* **Clarity (8/10):** Mathematical definitions are precise, and the authors are transparent regarding negative results on broadband traces.
* **Significance / fit for Computer Networks (7/10):** Matches the journal's scope perfectly by offering a practical, deployable middleware approach rather than just another RL algorithm.
* **Overall recommendation score (6/10):** A promising core idea that requires a heavier, more realistic empirical lift to cross the publication threshold.

### 10. Unbiased bottom line

This manuscript presents an auditable and theoretically interesting VMAF-aware shield, but its reliance on synthetic 5G traces to demonstrate its primary banking benefits—while simultaneously conceding those benefits collapse on real broadband traces—leaves the core utility unproven in the wild.

---

### Author action checklist

1. Replace synthetic 5G traces with real 5G/mmWave datasets (e.g., Lumos5G or Ghent) for the primary evaluation.
2. Strip Section 5.4 (PPO co-design) from the paper to avoid immediate attacks on single-seed statistical validity.
3. Run the primary evaluation using per-chunk (non-monotone) VMAF ladders natively rather than relying on a localized ablation.
4. Add an empirical analysis of conformal coverage specifically during abrupt, bursty throughput drops to defend the exchangeability assumption.
5. Tone down the word "certificate" in the title and abstract to emphasize its probabilistic, finite-sample nature.