# ABR-Awareness: Research Framework (Overview)

This document summarizes the **conceptual framework** of our adaptive bitrate (ABR) study: what we optimize, under which constraints, how safety is enforced, and how we evaluate. It is meant to align the **paper narrative** with the **implementation** (e.g., Version 12).

---

## 1. Problem setting

We consider **chunk-level bitrate selection** for HTTP adaptive streaming. At each step the agent chooses a representation (bitrate index). The environment provides:

- **Quality proxy**: VMAF (per-chunk).
- **Stall cost**: rebuffering time when download exceeds available buffer.
- **Smoothness**: penalty for abrupt quality changes (VMAF jumps between chunks).

**QoE-style evaluation** (for reporting) uses a weighted sum consistent with our simulator, e.g.  
`QoE ≈ VMAF − α·rebuffer − β·smoothness` with fixed weights for **fair comparison across methods**.

---

## 2. Optimization objective (training): Constrained MDP (CMDP)

Instead of hand-tuning fixed penalty weights for the whole state space, we formulate streaming as a **CMDP**:

- **Maximize** expected per-step **VMAF** (primary reward signal).
- **Subject to** soft budgets on:
  - **Mean rebuffer intensity** over an episode (stall risk).
  - **Mean smoothness penalty** (bitrate/quality stability).

We use **Lagrangian relaxation**:

- The policy maximizes a **Lagrangian reward**: VMAF minus **adaptive** penalties for rebuffer and smoothness, with dual variables \( \lambda_{\text{rebuf}}, \lambda_{\text{smooth}} \).
- Dual variables are updated from measured constraint gaps (primal–dual style), so penalties **adapt** during training rather than staying fixed.

**Algorithm**: PPO (Stable-Baselines3) trains the policy on the Lagrangian reward when the method uses the Lagrangian wrapper.

---

## 3. Safety: shield-in-the-loop (inference-time projection)

For **shielded** variants, a **deterministic safety mapping** may adjust the policy action before the environment step (e.g., reduce quality when estimated download time is incompatible with buffer). This is **not** a learned component; it is a **runtime guard** that makes the deployed policy safer under stress.

We distinguish:

- **Classic shield**: always applies the projection rule (subject to configured strength).
- **Risk-gated shield**: intervenes when a **risk condition** holds (e.g., estimated download time vs. buffer), potentially reducing unnecessary interventions.

Optional training-time additions (some variants):

- **Shield-aware reward shaping**: small penalties when the shield intervenes or when the executed action differs from the raw policy action (encourages agreement with safe behavior).
- **Hysteresis**: limits aggressive up-switches / large jumps to reduce oscillation (method-dependent).

---

## 4. Method families in our experiments

| Family | Role in the story |
|--------|-------------------|
| **Proposed (unshielded)** | CMDP + full features; shows **QoE potential** but can be **less safe** without a shield. |
| **Proposed + shield** | Same learning setup, but **safe action projection** at execution; primary **safety-centric** candidate. |
| **Proposed + shield + QoE shaping** | Ablates whether shield-aware shaping improves **QoE vs. interventions / switching** under the shield. |
| **Proposed + risk-gated shield** | Ablates **intervention frequency** vs. classic shield while targeting low stall. |
| **Ablations** | Remove Lyapunov signal, future/chunk lookahead, or Lagrangian training—**component analysis**. |
| **Pensieve-style RL** | Classic RL baseline with blinded content features (fair Pensieve-like setting). |
| **Non-RL baselines** | RobustMPC, BBA, Fugu, and **Genie** (oracle / upper-bound style baseline in simulation). |

---

## 5. Evaluation protocol (paper-facing)

- **Fair environment settings** per method (e.g., observation masking for Pensieve-like training).
- **Episode-level metrics**: average VMAF, rebuffer ratio, QoE, bitrate switches.
- **Paired statistics** where appropriate (e.g., Wilcoxon vs. a strong baseline on paired episodes).
- **Genie** is treated as an **oracle / upper reference**, not a practical competitor.

**Recommended positioning with current results:** emphasize a **safety–QoE trade-off**:

- Shielded methods **sharply reduce rebuffer** vs. unshielded Proposed.
- QoE may remain **competitive** with common RL/heuristic baselines, while **stall risk** is the headline improvement—this is a credible Q1-friendly claim if stated precisely.

---

## 6. One-sentence contribution template (use in Abstract)

> We train ABR policies with **primal–dual constrained RL** and deploy them with a **deterministic runtime shield** to reduce stalling, and we report the **resulting safety–QoE trade-offs** against standard streaming baselines under a trace-driven simulator.

---

## 7. File map (implementation)

- Training entry: `new/src/training/train_all_models_v12.py` (outputs under `master_v12/`).
- Evaluation entry: `new/src/evaluation/evaluate_all_models_v12.py` (results under `new/results/`).
- CMDP / Lagrangian core: `new/src/training/constrained_abr.py` (base), versioned wrappers in `constrained_abr_v12.py`.
- Shields: `new/src/training/safety_shield_v12.py`; optional shaping/hysteresis: `new/src/training/shield_aware_wrappers_v12.py`.

---

*Last aligned with the Version 12 experimental line. Update this document if targets, shields, or evaluation weights change materially.*
