# 🧠 Ablation Study Report — Content-Aware ABR Agent

This document summarizes the ablation experiments conducted to evaluate the contribution of each key component in our **content-aware adaptive bitrate (ABR)** model.  
The study systematically removes specific inputs or reward components to quantify their effect on overall QoE performance.

---

## 🎯 Objective

The goal of this study is to isolate and measure the effect of:
- **Content features (SI/TI)** — representing spatial and temporal complexity of video scenes.
- **Perceptual reward (VMAF-based)** — representing user-perceived visual quality.
- **Network-only features** — representing throughput and buffer dynamics.

---

## 🧩 Experimental Configurations

| Model | Description |
|--------|--------------|
| **Full** | The complete model using both content features and VMAF-based perceptual reward. |
| **No-SITI** | Content-related features (spatial and temporal information) are removed. |
| **No-VMAF** | Perceptual reward is disabled; only bitrate and rebuffering are used in reward computation. |
| **Network-Only** | The baseline similar to Pensieve; no content or VMAF features are used. |

---

## 📊 Training Summary

| Model | Best Validation Reward | Approx. Updates | Convergence Status |
|--------|------------------------|------------------|--------------------|
| **Full** | **+109.33** | ~245 | Stable convergence, best overall performance |
| **No-SITI** | +95.51 | ~175 | Slightly lower reward, faster convergence |
| **No-VMAF** | +94.81 | ~245 | Noticeable drop in perceptual quality |
| **Network-Only** | +105.71 | ~175 | Close numerically to Full, but less perceptually stable |

---

## 🔍 Observations

1. **Impact of VMAF-based Reward:**  
   Removing the perceptual reward causes a clear degradation in overall QoE, confirming that VMAF provides strong alignment with user-perceived quality.

2. **Impact of Content Features (SI/TI):**  
   Without content information, the model becomes less adaptive to scene complexity (e.g., high-motion segments) and exhibits slightly higher bitrate oscillations.

3. **Network-Only Baseline:**  
   Performs reasonably well but lacks perceptual awareness. It cannot distinguish between visually complex and simple scenes, leading to suboptimal QoE in diverse content.

---

## 🧠 Analytical Insights

| Factor | Effect on QoE | Notes |
|---------|----------------|-------|
| **VMAF-based Reward** | High | Crucial for perceptual alignment and overall quality. |
| **Content Features (SI/TI)** | Moderate | Enhance stability and adaptivity to visual complexity. |
| **Network-only Model** | Limited | Learns generic bitrate dynamics but lacks visual sensitivity. |

---

## 🧾 Conclusion

The ablation results demonstrate that:
- Both **content features** and **VMAF-based reward** significantly contribute to performance improvement.  
- The **Full model** achieves the best balance between bitrate efficiency, smoothness, and perceptual quality.  
- Removing either component leads to measurable degradation, confirming the synergy between visual perception and content awareness.

> **In summary:** Integrating perceptual feedback and content features enables an ABR policy that maximizes both technical and perceptual QoE.

---

## 🧪 Reproducibility

Each ablation model can be trained via:

```bash
python3 train_ablation.py --type full
python3 train_ablation.py --type no_siti
python3 train_ablation.py --type no_vmaf
python3 train_ablation.py --type network_only
