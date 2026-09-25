# Cover letter — Computer Networks

**To:** The Editors, *Computer Networks*  
**From:** Pedram Salehpour (corresponding author), on behalf of Saeed Zarbi and Leili Farzinvash  
**Affiliation:** Department of Computer Engineering, Faculty of Electrical and Computer Engineering, University of Tabriz, Tabriz, Iran  
**Email:** psalehpour@tabrizu.ac.ir  
**Manuscript title:** Certified Perceptual Shielding for Adaptive Bitrate Streaming  

---

Dear Editors,

Please consider our manuscript, *Certified Perceptual Shielding for Adaptive Bitrate Streaming*, for publication in *Computer Networks*.

Client-side ABR controllers still risk per-chunk stalls under volatile throughput and waste bandwidth when commercial rate ladders are perceptually saturated. We propose the Certified Perceptual Shield (CPS), a model-agnostic runtime projection that wraps any ABR policy. CPS combines VMAF-knee bandwidth banking, an online split-conformal throughput lower bound with a per-chunk stall coverage target, and a buffer-feasibility check. On synthetic 5G episodes, banking reduces bitrate and rebuffering versus a safety-only shield while keeping mean VMAF within one point; on real broadband traces, banking gains shrink and conformal safety drives stall reduction. These results align with the journal’s interest in measured networking systems, QoE, and principled online control under uncertainty.

The work is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the submission. We declare that generative AI (Claude) was used only to improve language and clarity; the authors reviewed the text and take full responsibility for the content.

Suggested highlights (also in `Highlights.txt`):

- CPS is a model-agnostic runtime shield that wraps any client-side ABR controller  
- VMAF-knee banking trims bitrate on saturated ladders within a 1-point VMAF budget  
- An online conformal throughput lower bound yields a per-chunk stall coverage bound  
- On synthetic 5G traces, banking cuts rebuffering by ~9% vs. safety-only projection  
- On real broadband traces, gains shrink and conformal safety drives stall reduction  

Thank you for your consideration.

Sincerely,  
Pedram Salehpour  
Corresponding author  
psalehpour@tabrizu.ac.ir  

---

*Paste the body (from “Dear Editors” through the signature) into Editorial Manager. Attach the compiled PDF and source zip separately as required by the journal.*
