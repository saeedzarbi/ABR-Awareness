# Revision Brief: Preparing the Manuscript for Elsevier *Computer Networks*

This brief is for an AI agent that will edit the paper's LaTeX source. It stands on its own: the brief gives everything you need, so you do not need any earlier review or conversation.

---

## 0. Context

| Field | Value |
|---|---|
| Paper title | *Certified Perceptual Shielding for Adaptive Bitrate Streaming* |
| Authors | Saeed Zarbi, Leili Farzinvash, Pedram Salehpour (corresponding: psalehpour@tabrizu.ac.ir), Department of Computer Engineering, Faculty of Electrical and Computer Engineering, University of Tabriz, Iran |
| Target journal | Elsevier *Computer Networks* (ISSN 1389-1286) |
| Current template | `elsarticle`, two-column layout (similar to the `5p` option), Times fonts via newtx, numbered citations (`elsarticle-num` look). The default "Preprint submitted to ..." footer has been removed. |
| Current size | 14 A4 pages (about 12 pages of text and 2 of references), about 11,750 words in total, 45 references, 10 figures, 9 tables, 1 algorithm, 2 propositions, Eqs. (1)-(8) and (A.1)-(A.2), and Appendix A |
| Goal | Make the manuscript ready to submit: meet the journal's formal requirements, remove internal contradictions, and flag every item that needs input from the authors |
| Reference template on disk | `/workspace/computer_networks_template/starter_cas/main.tex` (a `cas-dc` starter that already has highlights, keywords, `\credit`, `\printcredits`, and a placeholder for the generative-AI declaration) |

### Paper map (page numbers as printed in the PDF)

| Page(s) | Content |
|---|---|
| 1 | Title, authors, Abstract, Keywords, §1 Introduction (start) |
| 2 | §1 (contributions list, roadmap), §2 Related work, §2.1 |
| 3 | §2.2, §2.3, Table 1 (positioning), Table 2 (notation), §3 start, §3.1 |
| 4 | §3.1 (Eq. 1), Fig. 1 (control loop), §3.2 (Eq. 2 QoE), §3.3 Ladder saturation |
| 5 | Fig. 2, Table 3 (ladders), §3.4 (Eqs. 3-4), §3.5, §4 start, §4.1 (Eq. 5), Proposition 1 |
| 6 | Algorithm 1, §4.2 (Eqs. 6-7), Proposition 2, "Evaluation arms", §4.3 (Eq. 8), §4.4, §4.5 |
| 7 | Table 4 (defaults), §5 (RQ1-RQ6), §5.1 setup |
| 8 | §5.1 (continued), §5.2, Fig. 3, Fig. 4, §5.3 start |
| 9 | Fig. 5, Fig. 6, Table 5 (main results), §5.3 (broadband, injected drift), Fig. 7, §5.4 start |
| 10 | Table 6 (co-design), Fig. 8, Table 7, Fig. 9, §5.5, §5.6 start |
| 11 | Table 8, Table 9, Fig. 10, §6 Discussion (Local ladder inversions, Threats to validity) |
| 12 | §6 (continued, Future work), §7 Conclusion, Appendix A, Data availability, Funding, Declaration of competing interest, CRediT |
| 13-14 | References [1]-[45] |

---

## 1. Journal requirements (checked against the live Guide for Authors on 2026-09-25)

| # | Requirement | Rule |
|---|---|---|
| J1 | Template | The official LaTeX template is Elsevier **CAS** (`cas-dc` double-column or `cas-sc` single-column, from `els-cas-templates.zip`). **`elsarticle` is an accepted older alternative.** |
| J2 | Source files | An **editable source** (`.tex` or `.docx`) is required. A PDF is **not** accepted as the source. Word files must be single-column; double-column is allowed only for LaTeX. |
| J3 | Abstract | Required. Concise and factual, **at most 250 words**, readable on its own, no references, no undefined abbreviations. |
| J4 | Keywords | **1 to 7**, in English. Avoid multi-word keywords that contain "and" or "of". |
| J5 | Highlights | Encouraged. **3 to 5 bullets, each at most 85 characters including spaces**, supplied as a **separate file whose name contains "highlights"**. |
| J6 | Graphical abstract | Optional. **531 x 1328 px (h x w)** or proportionally larger. |
| J7 | Citations | Any consistent style is acceptable at submission. In-text citations must be **numbered in square brackets in order of first appearance**. |
| J8 | Reference completeness | Each reference must include authors, title, year, volume, and pages or article number. **Abbreviate journal names per LTWA.** |
| J9 | CRediT | An author contribution statement using the 14 CRediT roles. |
| J10 | Competing interest | A "Declaration of competing interest" is required. |
| J11 | Data availability | Data statement **Option C**: deposit the data and cite it, **or** explain why it cannot be shared. |
| J12 | Generative AI | If AI tools were used, add a section titled exactly **"Declaration of generative AI and AI-assisted technologies in the manuscript preparation process"**, placed **before the References**. |
| J13 | Author biographies | Up to **100 words per author, with a photo**. |
| J14 | Section numbering | 1, 1.1, 1.1.1 ... |
| J15 | Review model | Single anonymized, so author names may stay in the manuscript. |
| J16 | Length | No page limit is stated. |

---

## 2. Guardrails for the agent (read before editing)

1. **Do not change scientific claims, numbers, statistics, or results without author confirmation.** Where a task says `Needs-author-input: yes`, insert a visible marker instead of inventing content: `\authorcheck{...}` (define it once, see Section 7.1), and also record it in the changelog.
2. **Never fabricate** numbers, experimental results, datasets, citations, DOIs, page numbers, repository URLs, biographies, or AI-use statements.
3. **Flag contradictions; do not resolve them silently.** If two places disagree, do not pick one yourself. Mark both places and list them in the changelog, unless the task gives explicit evidence for the resolution.
4. **Keep a changelog.** Create `CHANGELOG_revision.md` next to `main.tex`. Each entry records: task ID, file and line, before, after, and a note (including any "awaiting author" status).
5. **Keep the document compilable.** Compile after each group of edits (`latexmk -pdf main.tex`). Add no new errors, undefined references, or undefined citations, and no overfull boxes wider than 5 pt in the edited regions.
6. **Keep existing citation keys and labels** (`\cite{...}`, `\label{...}`). If you must rename one, update every use and log the rename.
7. **Do not delete content without noting it.** If a task asks you to remove or merge something (for example a redundant figure), comment it out with `%% REMOVED (TASK-ID): ...` or move it to `removed_content.tex`, and log it.
8. **Verify every new citation** before adding it (see Section 9). Unverifiable items must not be cited.
9. **Keep spelling consistent** with the existing text, which currently uses American spelling ("behavior", "optimize").
10. When a task is complete, tick its box in Section 10 and record the verification you performed.

---

## 3. Task legend

- **Priority:** **P0** = blocks submission, **P1** = important (likely reviewer objection), **P2** = minor or polish.
- **ID prefixes:** `FMT` format and journal compliance, `SCI` scientific content and consistency, `FIG` figures and tables, `REF` references, `LANG` language.
- **Needs-author-input:** `yes` means the agent must not complete the fix alone. Insert an `\authorcheck{}` marker, prepare any scaffolding, and log it.
- Quotes in the "Problem" fields are copied from the PDF text layer. Math may appear flattened (for example "Bk" for $B_k$ and "10−4" for $10^{-4}$), so search the `.tex` for the surrounding words.

---

## 4. P0: blocking tasks

### FMT-01: Abstract exceeds 250 words
- **Location:** Abstract, p.1
- **Problem:** The abstract is **319 words** (whitespace tokens; about 322 if words joined by em-dashes are counted separately). It uses undefined abbreviations (`TOST`, `PPO`, `FCC`, `RobustMPC`, `(n+1)`) and unexplained symbols (ε, α). It also contains reversed and incomplete ranges: "reduced delivered bitrate by about 4.5–3.8% and rebuffering by about 9.3–9.2%".
- **Required action:** Replace the abstract with the drop-in draft in Section 6.2, or with an author-approved equivalent. Cross-check every number against the final Table 5 and §5 after SCI-02 is resolved.
- **Acceptance criteria:** Word count ≤ 250 (e.g. `detex` on the abstract piped to `wc -w`). No citations. Every abbreviation is expanded or removed. Ranges are written low-high and match Table 5.
- **Needs-author-input:** no for the text; **yes** for confirming the numbers if SCI-02 changes them.

### FMT-02: Highlights file missing
- **Location:** Submission package
- **Problem:** No highlights exist.
- **Required action:** Create `Highlights.txt` (the name must contain "highlights") with the 5 bullets from Section 6.1. If you migrate to CAS (FMT-07), also add them in `\begin{highlights}...\end{highlights}`.
- **Acceptance criteria:** 3-5 bullets, each ≤ 85 characters including spaces (check with `awk '{print length}'`). Every number matches the final paper.
- **Needs-author-input:** no (but check the numbers after SCI-02).

### FMT-03: Editable source package
- **Location:** Submission package
- **Problem:** Journal rule J2. The PDF cannot serve as the source.
- **Required action:** Prepare a flat folder containing `main.tex`, the `.bib` file and the generated `.bbl`, the class and style files if they are non-standard, all figure files (vector PDF or EPS preferred), `Highlights.txt`, and author photos. Make sure it compiles from a clean copy.
- **Acceptance criteria:** `latexmk -pdf` in a fresh copy of the folder produces the PDF with no missing files and no undefined references.
- **Needs-author-input:** no.

### FMT-04: Data availability statement is vague
- **Location:** "Data availability", p.12
- **Problem:** "The code, evaluation scripts, and exported CSV files underlying Table 5–Table 6 and Fig. 3–Fig. 8 are available with the article artifacts." There is no repository, DOI, or licence, so it does not meet Option C.
- **Required action:** Replace it with the template in Section 7.2. Insert `\authorcheck{}` markers for the repository URL, DOI, licence, and any data that cannot be shared (and why). Add the deposited artifact as a dataset or software reference in the bibliography once the authors supply its details.
- **Acceptance criteria:** The statement names a persistent repository (such as Zenodo or Figshare) with a DOI, or gives an explicit reason for not sharing. The artifact is cited in the references. There are no invented URLs.
- **Needs-author-input:** **yes**

### FMT-05: Generative-AI declaration
- **Location:** End matter, before the References
- **Problem:** No declaration is present. Whether AI tools were used cannot be determined from the PDF.
- **Required action:** Ask the authors. If AI was used, insert the section from Section 7.3, titled exactly "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", immediately before the References. If AI was not used, add nothing and log the authors' answer.
- **Acceptance criteria:** Either the section exists with the exact title and placement and author-supplied content, or the changelog records the authors' confirmation that no AI was used.
- **Needs-author-input:** **yes**

### FMT-06: Author biographies and photos missing
- **Location:** End matter
- **Problem:** Journal rule J13. No biographies are present.
- **Required action:** Add biography scaffolding for all three authors (Section 7.4) with `\authorcheck{}` placeholders and photo file placeholders.
- **Acceptance criteria:** Three biographies, each ≤ 100 words, supplied by the authors, each with a photo file included and rendered.
- **Needs-author-input:** **yes**

### SCI-01: Contradiction about ladder monotonicity
- **Location:** §3.3 (p.4); Table 3 column "Mono." (p.5); §5.5 (p.10); §6 "Local ladder inversions" (p.11)
- **Problem:**
  - Table 3 marks only 4 of 12 titles as monotone ("Yes"), and §3.3 says "Across the twelve-title suite, 67% exhibited a non-monotone session-mean ladder."
  - The same section then says "For the monotone per-video ladder evaluated here, ranking feasible indices by VMAF produced the same choice as projection to the highest feasible index."
  - §5.5: "The headline runs applied the knee rule to the pooled session-mean ladder, which is monotone."
  - §6: "Structurally, the evaluated shield ran on the monotone session-mean ladder, so Proposition 2 applied outright."
- **Required action:** Do not choose a side. Insert `\authorcheck{}` markers at all four locations asking whether the evaluated session-mean ladders are monotone. Once the authors confirm, rewrite the sentences consistently. If some ladders are non-monotone, restrict the scope statements for Proposition 2 accordingly.
- **Acceptance criteria:** Every statement about monotonicity agrees with Table 3. The scope of Proposition 2 matches the evaluated ladders. The changelog records the authors' decision.
- **Needs-author-input:** **yes**

### SCI-02: Inconsistent headline numbers across tables and text
- **Location:** Table 5 and §5.2 (pp.8-9) versus Table 8, Table 9, and §5.6 (pp.10-11)
- **Problem:**
  - Tables 8 and 9 describe the same configuration: "same greedy host on the synthetic 5G test pool at 204 paired episodes", ε=1.0, α=0.10. For it they report bitrate saved **4.4%**, rebuffering **8.2%**, and intervention **71.0%**.
  - Table 5 and §5.2 report **4.5%** (3131 to 2990 kb/s), **9.3%** (41.9 s to 38.0 s), and **70.4%**.
- **Required action:** Flag these with `\authorcheck{}` in Table 8, Table 9, and §5.6. Ask the authors either to regenerate the numbers from one run or to state explicitly how the configurations differ (seed set, shield version, and so on). Do not edit the numbers yourself.
- **Acceptance criteria:** Identical configurations report identical numbers, or the difference in configuration is stated in the captions and text.
- **Needs-author-input:** **yes**

### SCI-03: Table 8 column headers are swapped
- **Location:** Table 8, p.11
- **Problem:** The column "Rebuffer ∆ (%)" holds +0.02, -0.00, -0.17, -0.48, and the column "VMAF ∆ (pts)" holds -3.5, -8.2, -30.8, -46.1. §5.6 says "the mean VMAF change remained negligible (at most 0.48 points)", so the first set are VMAF values.
- **Required action:** Swap the two headers (or swap the two data columns so the header order stays the same). Replace "-0.00" with "0.00". Log the change and mark it for author confirmation.
- **Acceptance criteria:** The VMAF column holds values with magnitude ≤ 0.48 and the rebuffering column holds -3.5 ... -46.1, consistent with the §5.6 text.
- **Needs-author-input:** no (the text is the evidence), but record it in the changelog for author sign-off.

### SCI-04: Fig. 7 contradicts the text on coverage under injected drift
- **Location:** §5.3 "Coverage under injected drift" (p.9); Fig. 7 (p.9)
- **Problem:**
  - The text says: "per-chunk coverage in the steady pre-shock window was 0.928 (n=1,632 chunks). During the shock it was 0.929 (n=2,040); in the post-shock recovery window it rose to 0.944", and "the bound tracked the shock without collapsing below the target".
  - Fig. 7 shows "Rolling coverage" starting near 0.6 and staying **below** the 0.90 target line until about chunk 38.
  - Coverage rising *during* a 0.3x throughput shock is also counter-intuitive and is not explained.
  - The rolling-window definition is not given.
- **Required action:** Insert `\authorcheck{}` markers in the caption and the paragraph. Ask the authors to reconcile the figure with the numbers (for example: cumulative versus rolling coverage, whether the warm-up is included, the window length), to explain why coverage increases during the shock, and to state the rolling-window length in the caption.
- **Acceptance criteria:** The figure and the text report consistent values. The caption defines the rolling window. The explanation is present.
- **Needs-author-input:** **yes**

### SCI-05: Chunk counts disagree (173 vs. 113) and wrong table citations
- **Location:** Fig. 2 caption (p.5), §3.3 (p.4), §5.5 and Table 7 (p.10), §6 (p.11)
- **Problem:**
  - The Fig. 2 caption says "across all 173 chunks", but §5.5 says "(26.5% of the 113 chunks held at least one inverted adjacent pair; Table 3)" and Table 7 says "113 chunks".
  - Table 3 contains only session-mean data, yet §5.5 cites it for per-chunk statistics, and §6 says "Table 3 showed that, chunk by chunk, a sizable share of adjacent rungs in titles like Sintel and Big Buck Bunny ran near-flat or inverted."
  - The per-chunk analysis covers only "a legacy four-title subset" (§3.3), and RQ5 and the conclusions do not say so.
- **Required action:** Flag the count (`\authorcheck{}`). Change the §5.5 and §6 citations from "Table 3" to "Fig. 2" where the claim concerns per-chunk data. Add a sentence to §5.5 stating that the per-chunk ablation uses the four titles shown in Fig. 2. Replace the internal term "legacy four-title subset" with a neutral description (see LANG-20).
- **Acceptance criteria:** One chunk count is used consistently (confirmed by the authors). No per-chunk claim cites Table 3. The four-title scope is stated in §5.5.
- **Needs-author-input:** **yes** for the count; no for the citation fixes and the scope sentence.

### SCI-06: Reversed and incomplete ranges in the abstract and conclusion
- **Location:** Abstract (p.1), §7 Conclusion (p.12)
- **Problem:** "by about 4.5–3.8% and rebuffering by about 9.3–9.2%". Per §5.2 the host values are bitrate 4.5, 4.1, 4.1, 3.8% and rebuffering 9.3, 8.9, 8.9, 9.2%.
- **Required action:** Rewrite as "by 3.8–4.5% ... by 8.9–9.3%" in both places.
- **Acceptance criteria:** Ranges are ordered low to high and include the minimum and maximum from §5.2 and Table 5.
- **Needs-author-input:** no (recheck after SCI-02)

### SCI-07: Intervention-rate text contradicts Table 5
- **Location:** §5.2, p.8, paragraph starting "Certified intervention decreased across all hosts"
- **Problem:** "70.4% for always-top greedy, 61.1% for BOLA, 31.1% for RobustMPC ... and near zero for the conservative Pensieve-style control." Table 5 gives Pensieve 7.4% and BBA 66.1% (BBA is omitted from the text).
- **Required action:** Replace the sentence with the "after" text of LANG-12, which uses the Table 5 values.
- **Acceptance criteria:** Every rate in the sentence equals its Table 5 value. BBA is included. "near zero" is removed.
- **Needs-author-input:** no

### SCI-08: Suspicious identical p-values and an unclear test procedure
- **Location:** Note under Table 5 (p.9); note under Table 6 and §5.4 (p.10)
- **Problem:** "greedy bitrate p=8.5×10−4, rebuffering p=8.5×10−4" and "VMAF p=5.0×10−5, rebuffering p=5.0×10−5". Identical values suggest the resolution floor of a permutation or bootstrap test rather than an exact Wilcoxon test. Also, §5.4 says "Paired Wilcoxon tests and episode-bootstrap intervals were reported as 5.0×10−5 (VMAF) and 5.0×10−5 (rebuffering)", which mixes up intervals and p-values.
- **Required action:** Insert `\authorcheck{}` asking the authors to confirm the exact test (exact or normal-approximation Wilcoxon, or a permutation test with N resamples) and the true p-values. State the method once in §5.1.
- **Acceptance criteria:** §5.1 names the test, the zero-handling method, and any resampling count. The reported p-values are confirmed by the authors.
- **Needs-author-input:** **yes**

### SCI-09: Gap in Proposition 1 and its proof
- **Location:** §4.1 Proposition 1 and proof sketch (p.5); Algorithm 1 lines 4-5 (p.6); Proposition 2 (p.6)
- **Problem:**
  - The condition is "If dk ( j) ≤ max{Bk − m, 0.1} via (5)", but the proof says "the realized download time is at most dk ( j), which by assumption does not exceed Bk". When $B_k < 0.1$, $\max\{B_k-m, 0.1\} = 0.1 > B_k$, so this step fails.
  - Algorithm 1 (line 4: "if Bk ≤ Bcrit then", line 5: return rung 0) returns rung 0 without any feasibility test, so the certificate does not cover those steps.
  - Proposition 2 defines "F = { j : dk ( j) ≤ Bk − m}", which differs from Eq. (6) (`max{B_k − m, 0.1}`).
- **Required action:** Propose a corrected statement to the authors. For example, add the hypothesis "$B_k > B_{crit}$ and $B_k - m \ge 0.1$" (or replace the 0.1 floor in the certificate condition by $B_k$), and state explicitly that steps with $B_k \le B_{crit}$ are not certified. Make the definition of F consistent with Eq. (6), or explain the difference. Put the proposed text in an `\authorcheck{}` block; do not change the proposition silently.
- **Acceptance criteria:** Every hypothesis used in the proof is in the statement. The scope excludes (or handles) the $B_{crit}$ branch. F and Eq. (6) are consistent. The authors have approved.
- **Needs-author-input:** **yes** (the mathematical intent must be confirmed)

### SCI-10: Missing baselines (non-conformal safety and direct competitors)
- **Location:** §1 (p.1), §2 and Table 1 (pp.2-3), §5 (pp.7-11)
- **Problem:**
  - The introduction criticizes the standard repair because it "reduces throughput pessimism to a hand-tuned factor with no coverage guarantee", but no evaluation arm uses such a fixed-factor safety rule. CPS itself falls back to exactly that (§4.1: "we fell back to a fixed scale ρfb =0.80").
  - Direct competitors listed in Table 1 (Conformal-ABR [21], SafeSABR [22], Kairos [23]) are not compared experimentally.
  - Adaptive conformal inference (ACI) is not discussed.
  - The Pensieve-style baseline sits near the bottom rung (Table 5: 722 kb/s, 0.1 s rebuffering), which reviewers may read as an under-trained strawman.
- **Required action:** Insert an `\authorcheck{}` in §5.1 listing the missing arms: fixed-factor safety (for example 0.8 x harmonic mean); at least one of [21]-[23], or a justified reason why it is infeasible; optionally an ACI-based bound. Ask the authors to comment on how the Pensieve-style policy was trained. Add a short paragraph in §6 on why the competitors were or were not compared, with the text supplied by the authors.
- **Acceptance criteria:** Either the new arms appear in Table 5 (or a new table) with author-supplied results, or §6 explicitly justifies their absence. The Pensieve training adequacy is addressed.
- **Needs-author-input:** **yes** (new experiments)

### SCI-11: Uncited "prior VMAF-aware ranking shields"
- **Location:** §1 (pp.1-2), §4.2 and Proposition 2 (p.6)
- **Problem:** "Prior VMAF-aware ranking shields provably collapse to highest-feasible-index projection on monotone ladders (Proposition 2)." There is no citation for these prior shields, and §4.2 says "Banking distinguishes CPS from prior VMAF-ranking shields".
- **Required action:** Add `\authorcheck{cite the prior VMAF-ranking shield(s)}`. If the authors have no citable prior work, rephrase to "A natural VMAF-ranking shield ..." so it no longer implies published work.
- **Acceptance criteria:** Each mention is either backed by a verified citation or rephrased so it does not claim prior published work.
- **Needs-author-input:** **yes**

### SCI-12: The QoE metric in Eq. (2) is defined but never reported
- **Location:** §3.2 Eq. (2) (p.4); Table 5 (p.9)
- **Problem:** Eq. (2) defines the per-chunk QoE ("Our shared harness fixed β=4.3 per stall second and µ=1.0 per absolute VMAF-point change"), but no table or figure reports session QoE. Lower bitrate is not a benefit in itself.
- **Required action:** Add a "QoE" column placeholder to Table 5 with `\authorcheck{}` cells, and a sentence in §5.2 that the authors will complete.
- **Acceptance criteria:** Table 5 reports the session QoE per arm, with author-supplied values, and §5.2 discusses it.
- **Needs-author-input:** **yes**

### SCI-13: Synthetic 5G traces are undescribed and realism is questionable
- **Location:** §5.1 "Primary 5G suite" (p.7)
- **Problem:** "We generated them synthetically because public per-chunk 5G/mmWave throughput logs suitable for chunk-level ABR replay remained scarce." The generator (dip and outage model, parameters, seeds) is not described, and no validation against real traces is shown. Public 5G throughput datasets do exist (see Section 9, unverified).
- **Required action:** Add a paragraph or subsection placeholder (`\authorcheck{}`) for the generator description: model, parameters, distributions, and a comparison with real measurements. Soften the scarcity claim to something the authors can defend, or add an evaluation on a real 5G dataset (authors decide).
- **Acceptance criteria:** The generator is fully specified so it can be reproduced. The scarcity claim is supported, qualified, or replaced.
- **Needs-author-input:** **yes**

### SCI-14: Residual window (W=200) is longer than an episode (K_e=48)
- **Location:** §4.5 (p.6), §5.1 (p.7), §5.3 (p.9)
- **Problem:** "window W=200" versus "Ke =48 chunks/episode", and elsewhere "After a 20-chunk conformal warm-up". It is unclear whether the residual window carries over across episodes and traces, which affects the exchangeability assumption.
- **Required action:** Insert `\authorcheck{}` in §4.5 asking whether the window persists across episodes. Once answered, add one explicit sentence about it and about how it relates to the 20-chunk warm-up.
- **Acceptance criteria:** The window's behavior across episodes is stated explicitly and matches the warm-up description.
- **Needs-author-input:** **yes**

---

## 5. P1: important tasks

### SCI-15: "Same ε ceiling" claim conflicts with ε_risk = 4
- **Location:** Contributions bullet 3 (p.2), RQ3 (p.7), §4.3 (p.6), §7 (p.12)
- **Problem:** "It improves the trade-off between bandwidth and stalls under the same ε ceiling at comfortable buffers" and RQ3 "without leaving the perceptual envelope", while §4.3 says "The “improved” experiments ran with ε=1.0, εrisk =4, Brisk =8 s, and H≥6 at forecast quantile 0.2." The TOST margin is ±1.0.
- **Required action:** Rephrase these to say the predictive variant widens the budget up to ε_risk = 4 VMAF points under risk. In §7, qualify "Predictive banking further improved the margin by about 14.6% greedy bitrate" (see SCI-21).
- **Acceptance criteria:** No sentence claims the predictive variant stays within ε = 1. The budget of 4 is stated wherever the 14.6% result appears.
- **Needs-author-input:** no

### SCI-16: Unrealistic rebuffering levels and a small buffer
- **Location:** Table 5 (p.9), §4.5 (p.6), §5.2 (p.8)
- **Problem:** The certified greedy arm has "38.0 s" of rebuffering per 192 s episode (48 x 4 s, about 20% rebuffering ratio), and Raw has "85.5 s", with "buffer cap 12 s". The explanation "The forced downshift at Bcrit and the first-chunk cold start explained the residual rebuffering in the certified arm" does not plausibly account for 38 s.
- **Required action:** Insert `\authorcheck{}` asking the authors to (a) justify the 12 s cap and the trace severity, (b) consider adding a 30 s or 60 s buffer experiment, and (c) give a quantitative breakdown of the residual rebuffering. Report the rebuffering ratio alongside total seconds.
- **Acceptance criteria:** The text justifies the regime or reports an additional buffer setting, and the residual-rebuffering explanation is quantitative.
- **Needs-author-input:** **yes**

### SCI-17: How 48-chunk episodes were built from short clips
- **Location:** §5.1 (p.7)
- **Problem:** The episodes use "Ke =48 chunks/episode, T c =4 s" with Xiph DERF clips (Park Joy, Old Town Cross, ...), which are typically about 10 s long. The construction (looping or otherwise) is not described.
- **Required action:** Add `\authorcheck{describe episode construction from source clips}` in §5.1.
- **Acceptance criteria:** One or two sentences state how the chunk sequences were built for each title.
- **Needs-author-input:** **yes**

### SCI-18: Missing reproducibility details
- **Location:** §3.3, §4.3 (Eq. 8), §4.4, §4.5, §5.1, Appendix A
- **Problem:**
  - The encoding ladder is not specified (codec, resolution per rung, encoder settings, VMAF model version). The rung bitrates 300/750/1200/1850/2850/6000 kb/s can only be inferred from the Fig. 2 axis.
  - `f` in Eq. (8) is given only as "with f ∈ [0, 1]".
  - "H≥6" (§4.3, Table 4) is not a specific value.
  - PPO training budget, learning rate, and batch size are missing, as are the Stable-Baselines3, Python, and library versions.
  - The hardware is described only as "on a commodity CPU".
- **Required action:** Add a "Reproducibility details" paragraph (or a table in the appendix) with an `\authorcheck{}` for each missing item. Replace "H≥6" with the exact value the authors supply.
- **Acceptance criteria:** Every listed item is specified with author-supplied values. There are no `≥` placeholders for configuration values.
- **Needs-author-input:** **yes**

### SCI-19: Statistical rigor
- **Location:** §5.1 (pp.7-8), §5.2, Table 5
- **Problem:**
  - The paper acknowledges that "effective independent-content diversity was twelve, not 204", yet it uses only episode-level Wilcoxon tests.
  - No p-values are given for BOLA or RobustMPC.
  - The Holm-Bonferroni correction is claimed without a table.
  - The headline percentages have no confidence intervals.
- **Required action:** Add `\authorcheck{}` requesting cluster (per-title) bootstrap confidence intervals or a mixed-effects analysis, p-values for all hosts, and a supplementary Holm table. Prepare the table skeleton.
- **Acceptance criteria:** Every headline effect has a confidence interval. All hosts have p-values. Holm-adjusted results are reported.
- **Needs-author-input:** **yes**

### SCI-20: Broadband episode count vs. trace count
- **Location:** §5.1 (p.7), §5.3 (p.9)
- **Problem:** "(27 held-out test traces, source-disjoint from training)" versus "empirical conformal coverage was 0.890 (below the 0.90 target at n=204 episodes)".
- **Required action:** Add `\authorcheck{}` asking how 204 episodes were drawn from 27 traces, and add the explanation to §5.1.
- **Acceptance criteria:** The sampling of broadband episodes is stated explicitly.
- **Needs-author-input:** **yes**

### SCI-21: Overclaiming language
- **Location:** as listed in the table below
- **Problem and required action** (rewording only; no numbers change):

| Location | Current text | Replace with |
|---|---|---|
| Fig. 6 caption (p.9) | "The certified arm stochastically dominates safety and raw on the stall axis." | "The certified arm's rebuffering distribution lies at or left of the safety arm and well left of raw." (keep the original claim only if the authors supply a dominance test) |
| §5.2 (p.8) | "Conformal safety eliminated the catastrophic stalls" | "Conformal safety removed most of the stall time relative to Raw" |
| §5.1 (p.7) | "The negative broadband result (Section 5.3) showed that we did not select regimes that favor banking." | "We also report a regime (broadband, Section 5.3) in which banking yields little benefit." |
| §7 (p.12) | "and extended naturally to learned policies." | "and can wrap learned policies; our co-design evidence is limited to a single training seed." |
| §7 (p.12) | "Predictive banking further improved the margin by about 14.6% greedy bitrate." | "With a wider risk budget (ε_risk = 4), predictive banking raised greedy bitrate savings to 14.6%." |

- **Acceptance criteria:** None of the "current" phrases remain unless the authors supply supporting evidence.
- **Needs-author-input:** no (yes only if the authors want to keep a claim and supply a test)

### SCI-22: Future-work item contradicts the evaluation
- **Location:** §6 "Future work" (p.12)
- **Problem:** "applying CPS to RobustMPC and Comyco hosts", although RobustMPC is evaluated in Table 5.
- **Required action:** Change to "applying CPS to Comyco and other learned hosts" (drop RobustMPC).
- **Acceptance criteria:** Future work no longer lists a host that is already evaluated.
- **Needs-author-input:** no

### SCI-23: Conflicting notation
- **Location:** §3.1 (p.4), §3.4.1 (p.5), Table 2 (p.3), §4.1 (p.5), Proposition 2 (p.6)
- **Problem:** "The effective trace throughput Ĉk is exogenous to the controller" (§3.1) and "Let Ĉk > 0 be the trace-derived effective throughput" (§3.4.1), whereas Table 2 and §4.1 use Ĉ for the "throughput predictor" (harmonic mean) and `C^act` for the realized throughput.
- **Required action:** Use `C_k^{act}` (realized, trace-derived) in §3.1 and §3.4.1, including the nominal download-time formula and `\tilde{C}_k`, and keep `\hat{C}_k` only for the predictor. Check every occurrence (the text around Eq. (1), §3.4.1, Eq. (5), Table 2). Log the macro changes. Fix the definition of F in Proposition 2 together with SCI-09.
- **Acceptance criteria:** Each symbol has exactly one meaning throughout, consistent with Table 2.
- **Needs-author-input:** no (confirm with the authors if the meaning is ambiguous in any equation)

### SCI-24: Primary PPO host missing from the results tables
- **Location:** §5.4 (p.9), Table 5
- **Problem:** "certified banking vs. safety saved 35.3% bitrate and reduced rebuffering by 79.3% (p=6.2 × 10−18 )". These results appear in no table.
- **Required action:** Add a "PPO (content-aware)" row block to Table 5 with `\authorcheck{}` cells.
- **Acceptance criteria:** The Raw/Safety/Certified metrics for this host are tabulated, with author-supplied values.
- **Needs-author-input:** **yes**

### SCI-25: Problem statement and structure of the contributions
- **Location:** §1 (pp.1-2)
- **Problem:**
  - The introduction repeats many §5 numbers but has no crisp problem formulation.
  - Two of the five contribution bullets ("Regime characterization", "Policy–shield co-design (supporting)") are findings, not contributions.
  - The contributions list is placed after a long results paragraph.
- **Required action:**
  - Add a one- or two-sentence problem statement ("Given a base ABR policy π and a ladder ..., design a projection S such that ...") before the contributions.
  - Reduce the list to three core contributions and move regime characterization and co-design into one "Findings" sentence.
  - Shorten the results paragraph by keeping at most three headline numbers.
  - Do not change any number.
- **Acceptance criteria:** A problem statement exists. There are three contribution bullets. The introduction's numbers are a subset of those in §5 and match them.
- **Needs-author-input:** no (the authors should review)

### SCI-26: Related-work gaps and journal fit
- **Location:** §2 (pp.2-3)
- **Problem:**
  - Missing lines of work: safety fallback and uncertainty for learned ABR, recent learned ABR, sensitivity-aware QoE, and online conformal methods under distribution shift.
  - Only one *Computer Networks* paper is cited ([39]).
  - The related work does include 12 references from 2023-2026, so recency is partly addressed.
- **Required action:** Use the **unverified** list in Section 9. Verify each item, and add only verified and relevant items with author approval. Search for recent (2023-2026) *Computer Networks* papers on ABR, video streaming, or safe RL for networking, verify them, and propose them to the authors.
- **Acceptance criteria:** Every added citation is verified (title, authors, venue, year, DOI) and approved by the authors. At least two relevant *Computer Networks* papers are considered.
- **Needs-author-input:** **yes** (approval of additions)

### SCI-27: Single-seed co-design result
- **Location:** §5.4, Table 6, Fig. 8, §6 "Threats to validity" item 5
- **Problem:** "the co-design contrast used a single training seed per policy".
- **Required action:** Add `\authorcheck{}` offering two options: add 3-5 training seeds, or shorten §5.4 and Fig. 8 and keep it as a clearly labeled supporting result (the current framing).
- **Acceptance criteria:** Either multi-seed results are reported or the section is shortened and consistently labeled "supporting".
- **Needs-author-input:** **yes**

### FIG-01: Fig. 1 text is unreadable
- **Location:** Fig. 1, p.4
- **Problem:** The box labels are about 4 pt at print size. The figure embeds a Type 3 TimesNewRoman font.
- **Required action:** Redraw or re-export the figure with every text ≥ 7 pt at final size, or make it full width with `figure*`. Use embedded Type 1 or TrueType fonts.
- **Acceptance criteria:** The smallest glyph is ≥ 7 pt at print size (check by zooming to 100% on A4), and `pdffonts` shows no Type 3 fonts for this figure.
- **Needs-author-input:** no (needs the figure source file; if it is unavailable, flag it)

### FIG-02: Fig. 2 has small, overlapping labels and a count mismatch
- **Location:** Fig. 2, p.5
- **Problem:** The axis tick labels and legend are too small. The x tick labels in panel (b) overlap. The caption says "173 chunks" (see SCI-05).
- **Required action:** Regenerate with a larger font (≥ 7 pt) and rotated or abbreviated tick labels. Fix the count after SCI-05.
- **Acceptance criteria:** The labels are legible and do not overlap, and the count is consistent.
- **Needs-author-input:** no for the plot (needs the plotting script); **yes** for the count

### FIG-03: Fig. 3 mixes units and uses an undefined label
- **Location:** Fig. 3, p.8
- **Problem:** Percentages and "VMAF cost (pts)" share one y-axis. The VMAF bars at 0.0 are invisible. The group label "BB greedy" is undefined and easily confused with BBA or Big Buck Bunny.
- **Required action:** Split the figure into two panels (or use a secondary axis). Rename the group to "Broadband greedy". Keep the data unchanged.
- **Acceptance criteria:** Each axis has a single unit, and all labels are defined in the caption or text.
- **Needs-author-input:** no

### FIG-04: Fig. 10 caption claims something not plotted
- **Location:** Fig. 10, p.11
- **Problem:** The caption says "Left: banked bitrate rises and the mean VMAF change stays near zero as the budget ε grows (α=0.10)." The left panel plots only "bitrate saved (%)".
- **Required action:** Either add the mean VMAF change on a secondary axis (data from Table 8, after SCI-03) or remove that clause from the caption.
- **Acceptance criteria:** The caption describes only what is plotted.
- **Needs-author-input:** no

### FIG-05: Table 5 structure
- **Location:** Table 5, p.9
- **Problem:** No QoE column (SCI-12) and no primary PPO host (SCI-24).
- **Required action:** Implement the skeletons from SCI-12 and SCI-24. Keep the booktabs style.
- **Acceptance criteria:** The table compiles, fits the column (or uses `table*`), and has its placeholders marked.
- **Needs-author-input:** **yes** (values)

### LANG-19: Global tense and style pass
- **Location:** Whole manuscript
- **Problem:** The method, paper structure, and claims are written in the past tense ("This section formalized", "Table 2 collected", "We introduced", "Section 5.5 showed"). Many short, choppy sentences appear, especially in the Abstract and §1.
- **Required action:**
  - Use the present tense for method descriptions, paper structure, figure and table descriptions, and claims.
  - Keep the past tense only for experiments actually carried out.
  - Merge very short consecutive sentences where the meaning allows.
  - Apply the specific fixes in Section 8.1.
  - Do not change technical content.
- **Acceptance criteria:** Sections 1, 3, and 4 describe the method in the present tense, and every before/after item in Section 8.1 is applied.
- **Needs-author-input:** no

### LANG-20: Remove internal jargon and working notes
- **Location:** p.4 "Our shared harness"; p.4 "legacy four-title subset"; p.6 "The “improved” experiments"; p.10 Table 6 note "(requires synced episodes.csv)"; p.9 Table 5 note "Improved predictive banking defaults in Table 4."
- **Required action:**
  - Delete "(requires synced episodes.csv)".
  - Replace "harness" with "emulator".
  - Replace "legacy four-title subset" with "a four-title subset for which frame-level VMAF logs are available".
  - Give the predictive variant a proper name (for example `CPS-P`) and use it wherever "improved shield" or "improved" refers to it (§4.3, §5.3, §5.4, Table 5 note, Table 6 caption, Fig. 8 caption).
- **Acceptance criteria:** `grep -n -i "improved\|harness\|legacy\|episodes.csv"` in the `.tex` finds no internal-jargon hits (ordinary uses of "improved" as a verb are fine).
- **Needs-author-input:** no

### REF tasks (P1)
The P1 reference tasks are REF-03, REF-04, REF-06, REF-07, REF-08, REF-10, REF-11, REF-12, REF-13, and REF-15. See the per-reference table in Section 8.2.

### LANG tasks (P1)
LANG-01 to LANG-06, LANG-09, and LANG-11 to LANG-18 are P1. See the table in Section 8.1.

---

## 6. Drop-in text blocks (DRAFTS: check against the paper's final numbers)

> These drafts use only numbers that appear in the current PDF. If SCI-02, SCI-06, or any other task changes a number, update these blocks before submission.

### 6.1 Highlights (DRAFT): 5 bullets, each ≤ 85 characters

| # | Highlight | Characters (incl. spaces) |
|---|---|---|
| 1 | CPS is a model-agnostic runtime shield that wraps any client-side ABR controller | 80 |
| 2 | VMAF-knee banking trims bitrate on saturated ladders within a 1-point VMAF budget | 81 |
| 3 | An online conformal throughput lower bound yields a per-chunk stall coverage bound | 82 |
| 4 | On synthetic 5G traces, banking cuts rebuffering by ~9% vs. safety-only projection | 82 |
| 5 | On real broadband traces, gains shrink and conformal safety drives stall reduction | 82 |

`Highlights.txt` (plain text, separate file; the leading "- " is not counted in the lengths above):

```text
- CPS is a model-agnostic runtime shield that wraps any client-side ABR controller
- VMAF-knee banking trims bitrate on saturated ladders within a 1-point VMAF budget
- An online conformal throughput lower bound yields a per-chunk stall coverage bound
- On synthetic 5G traces, banking cuts rebuffering by ~9% vs. safety-only projection
- On real broadband traces, gains shrink and conformal safety drives stall reduction
```

Optional CAS version (only if migrating per FMT-07):

```latex
\begin{highlights}
\item CPS is a model-agnostic runtime shield that wraps any client-side ABR controller
\item VMAF-knee banking trims bitrate on saturated ladders within a 1-point VMAF budget
\item An online conformal throughput lower bound yields a per-chunk stall coverage bound
\item On synthetic 5G traces, banking cuts rebuffering by $\sim$9\% vs. safety-only projection
\item On real broadband traces, gains shrink and conformal safety drives stall reduction
\end{highlights}
```

### 6.2 Abstract (DRAFT): 246 words (current abstract: 319 words)

Numbers used and their sources:
- 204 episodes and twelve titles: §5.1
- 3.8-4.5% bitrate and 8.9-9.3% rebuffering: §5.2 and Table 5
- 0.919 coverage vs. a 0.90 target: Table 5
- 14.6% and 12.6%: §5.3
- 0.9% and 98.7%: §5.3

```latex
\begin{abstract}
Client-side adaptive bitrate (ABR) controllers decide chunk by chunk, yet a
single request can stall playback when throughput drops, or waste bandwidth
when the top rungs of the bitrate ladder are perceptually saturated. We propose
the Certified Perceptual Shield (CPS), a model-agnostic runtime projection that
wraps any heuristic, model-based, or learned ABR policy. CPS combines three
components. First, knee-based bandwidth banking lowers even a feasible proposal
to the smallest rung whose Video Multimethod Assessment Fusion (VMAF) score lies
within a perceptual budget of the proposal. Second, an online split-conformal
lower bound on throughput provides distribution-free coverage under residual
exchangeability. Third, a feasibility check rejects any rung whose download time
under that bound exceeds the buffer margin, which yields a per-chunk
probabilistic stall guarantee rather than a zero-stall promise. We evaluate CPS
in a chunk-level emulator on 204 paired synthetic 5G episodes over twelve
reference titles with diverse rate-quality ladders. Compared with a safety-only
shield, banking reduces delivered bitrate by 3.8--4.5\% and rebuffering by
8.9--9.3\% for four deterministic controllers, while mean VMAF stays within one
point (equivalence test) and empirical coverage is 0.919 against a 0.90 target.
A risk-aware predictive variant raises the savings to 14.6\% bitrate and 12.6\%
rebuffering for a greedy controller. On real fixed and mobile broadband traces,
banking gains fall to about 0.9\%, whereas conformal safety still removes about
98.7\% of stall time relative to the unshielded controller. These results show
that the benefit of banking depends on ladder saturation and network regime.
\end{abstract}
```

Note: the 14.6% / 12.6% predictive result uses a wider risk budget (ε_risk = 4; see SCI-15). If the authors prefer, append "under a wider risk budget" to that sentence; the abstract stays under 250 words.

### 6.3 Keywords (current: 6, compliant)

The current keywords are: "Adaptive Bitrate Streaming, Runtime Shielding, VMAF, Conformal Prediction, Certified Safety, Video QoE". Optionally (FMT-08, P2) replace the vague "Certified Safety" with "5G" or "Safe reinforcement learning", subject to author approval. Do not exceed 7.

---

## 7. Drop-in end-matter templates (require author input)

### 7.1 Placeholder macro (add to the preamble)

```latex
\usepackage{xcolor}
\newcommand{\authorcheck}[1]{\textcolor{red}{[AUTHOR: #1]}}
```

Remove every `\authorcheck` and the macro itself before the final submission (see the checklist).

### 7.2 Data availability (Option C)

```latex
\section*{Data availability}
The emulator, CPS implementation, evaluation scripts, trace lists, per-title VMAF
ladders, and per-episode result files underlying Tables~5--9 and Figs.~2--10 are
openly available at \authorcheck{repository name} under the
\authorcheck{licence} licence (DOI: \authorcheck{DOI})~\cite{ARTIFACT_KEY}.
The FCC and Norway throughput traces are third-party public datasets available
from their original providers~\cite{FCC_KEY,NORWAY_KEY}.
\authorcheck{State here any data that cannot be shared and the reason.}
```

Replace `ARTIFACT_KEY` with a new bib entry once the authors supply the deposit details. Replace `FCC_KEY` and `NORWAY_KEY` with the existing citation keys for [38] and [39], but first see REF-13 on the Norway source. Until then, leave the `\cite` commented out so the document still compiles.

### 7.3 Generative-AI declaration (insert only if the authors confirm AI use)

```latex
\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
During the preparation of this work the authors used \authorcheck{tool name and
version} in order to \authorcheck{purpose, e.g., improve language and readability}.
After using this tool, the authors reviewed and edited the content as needed and
take full responsibility for the content of the published article.
```

Place it immediately before the References (`\bibliography{...}` or `\begin{thebibliography}`).

### 7.4 Author biographies (≤ 100 words each, with photo)

For `elsarticle` (after the References, or wherever the submission system asks):

```latex
\section*{Author biographies}
\noindent\begin{minipage}[t]{0.25\linewidth}
  \includegraphics[width=\linewidth]{photo_zarbi.jpg}% AUTHOR: supply photo
\end{minipage}\hfill
\begin{minipage}[t]{0.72\linewidth}
  \textbf{Saeed Zarbi} \authorcheck{biography, max 100 words}
\end{minipage}
% Repeat for Leili Farzinvash and Pedram Salehpour.
```

For CAS (`cas-dc`):

```latex
\bio{photo_zarbi.jpg}
\textbf{Saeed Zarbi} \authorcheck{biography, max 100 words}
\endbio
```

Until the photo files exist, comment out the `\includegraphics` and `\bio{...}` lines, or use a placeholder image, so the document still compiles.

### 7.5 Existing end matter to keep (already compliant)

- "Funding" (p.12): compliant.
- "Declaration of competing interest" (p.12): compliant.
- "CRediT authorship contribution statement" (p.12): compliant (it uses valid roles). If migrating to CAS, move the roles into `\credit{}` per author and use `\printcredits`.
- Suggested order of end matter (FMT-11, P2): Conclusion → CRediT → Declaration of competing interest → Funding → Data availability → Acknowledgements (if any) → Generative-AI declaration (if any) → Appendix A → References → Biographies.

---

## 8. Language, reference, and minor format tasks

### 8.1 Language fixes: before/after

Acceptance for every LANG item: the "before" string no longer appears in the `.tex`, and the "after" string (or an author-approved equivalent) does. Needs-author-input: no, except where noted.

| ID | Priority | Page / location | Before (exact) | After |
|---|---|---|---|---|
| LANG-01 | P1 | p.1 Abstract | We addressed both failure modes with a Certified Perceptual Shield (CPS). | We address both failure modes with a Certified Perceptual Shield (CPS). |
| LANG-02 | P1 | p.1 Abstract; p.12 §7 | banking significantly reduced delivered bitrate by about 4.5–3.8% and rebuffering by about 9.3–9.2% | banking reduced delivered bitrate by 3.8–4.5% and rebuffering by 8.9–9.3% |
| LANG-03 | P1 | p.1 Abstract | The result is a per-chunk probabilistic stall certificate under the conformal assumption. The certificate is a coverage guarantee, not a promise of zero stalls. | This yields a per-chunk probabilistic stall guarantee under the conformal assumption, not a zero-stall promise. |
| LANG-04 | P1 | p.1 §1 | We introduced a Certified Perceptual Shield (CPS) that targets this structure | We introduce a Certified Perceptual Shield (CPS) that exploits this structure |
| LANG-05 | P1 | p.1 §1; p.4 §3.1 | volatile sub-6 GHz and 5G/mmWave throughput | volatile sub-6 GHz and mmWave 5G throughput |
| LANG-06 | P1 | p.3 §3 | This section formalized client-side ABR at the chunk timescale. ... Table 2 collected the symbols used throughout. | This section formalizes client-side ABR at the chunk timescale. ... Table 2 lists the notation. |
| LANG-07 | P2 | p.3 Table 2 | logged-QoE smoothness and stall weights ((2)) | logged-QoE smoothness and stall weights (Eq. (2)) |
| LANG-08 | P2 | p.5 Table 3 note | Top gain is VMAF(6000)−VMAF(2850) in kbps | Top gain is VMAF at 6000 kbps minus VMAF at 2850 kbps |
| LANG-09 | P1 | p.5 §4.1 | we fell back to a fixed scale ρfb =0.80 | we fall back to a fixed scale $\rho_{\mathrm{fb}} = 0.80$ |
| LANG-10 | P2 | p.6 and p.9 paragraph heads | Evaluation arms.. / Coverage under injected drift.. | Evaluation arms. / Coverage under injected drift. (fix the `\paragraph{...}` so there is no double period) |
| LANG-11 | P1 | p.6 §4.3 | The “improved” experiments ran with ε=1.0, εrisk =4, Brisk =8 s, and H≥6 at forecast quantile 0.2. | The predictive-banking experiments (CPS-P) use ε = 1.0, ε_risk = 4, B_risk = 8 s, H = \authorcheck{exact value}, and forecast quantile 0.2. (Needs-author-input: yes, for H) |
| LANG-12 | P1 | p.8 §5.2 | Certified intervention decreased across all hosts: 70.4% for always-top greedy, 61.1% for BOLA, 31.1% for RobustMPC— ... —and near zero for the conservative Pensieve-style control. | The certified intervention rate falls as the host becomes more conservative: 70.4% (greedy), 66.1% (BBA), 61.1% (BOLA), 31.1% (RobustMPC), and 7.4% (Pensieve-style). |
| LANG-13 | P1 | p.10 §5.4 | Paired Wilcoxon tests and episode-bootstrap intervals were reported as 5.0×10−5 (VMAF) and 5.0×10−5 (rebuffering). | Paired Wilcoxon tests give $p = 5.0\times10^{-5}$ for both VMAF and rebuffering. (confirm per SCI-08; Needs-author-input: yes) |
| LANG-14 | P1 | p.10 Table 6 note | ... over 204 episodes (requires synced episodes.csv). | ... over 204 episodes. |
| LANG-15 | P1 | p.11 §6 | These omissions were deliberate. The method did not claim dominance over every ABR controller ... | We deliberately limit the scope: CPS does not claim dominance over every ABR controller ... |
| LANG-16 | P1 | p.12 §6 Future work | Real-player deployment was the highest-priority direction. | Real-player deployment is the highest-priority direction. |
| LANG-17 | P1 | p.4 §3.3 | Section 5.5 showed that the knee rule remained effective. | Section 5.5 shows that the knee rule remains effective. |
| LANG-18 | P1 | p.9 §5.4 | Two PPO settings warranted separate reporting. | We report two PPO settings separately. |

(LANG-19, the global tense pass, and LANG-20, internal jargon, are in Section 5.)

### 8.2 Reference fixes, per reference number

Rule for every row: **do not invent missing bibliographic data.** Retrieve it from the DOI, the publisher page, or DBLP. If you cannot verify it, add an `\authorcheck{}` and log it.

| ID | Ref | Priority | Problem (as printed) | Required action | Acceptance | Needs-author-input |
|---|---|---|---|---|---|---|
| REF-01 | [3] | P2 | "Neural adaptive video streaming with pensieve" (lowercase proper noun) | Protect the case in BibTeX: `{P}ensieve` | The rendered title shows "Pensieve" | no |
| REF-02 | [4] | P2 | "arXiv preprint arXiv:1707.06347 (2017). arXiv:1707.06347." (duplicated identifier) | Keep a single arXiv identifier (use either the `eprint` field or the `journal` field, not both) | The identifier appears once | no |
| REF-03 | [6] | P1 | Cisco annual internet report with "accessed: 2025 (2023)"; cited for "prediction accuracy that varies across networks and devices [5, 6]" and for volatile 5G throughput "[5–7]" | Fix the web-reference format (title, organization, year, URL, access date). Ask the authors whether to replace it with a peer-reviewed measurement source for those claims | Well-formed entry, and the claims it supports are appropriate | yes (choice of replacement) |
| REF-04 | [8] | P1 | The Netflix tech-blog URL overflows the column and overlaps [21] (p.13) | Use `\usepackage{xurl}` or break the URL. Optionally add a verified peer-reviewed VMAF reference (see Section 9) | No overflow in the References | no |
| REF-05 | [13] | P2 | "IEEE Communications Surveys & Tutorials 21 (1) (2018) 562–585"; vol. 21(1) is usually dated 2019 | Check the year via the DOI and add the DOI | Year matches the DOI record | no |
| REF-06 | [14] | P1 | "ACM Computing Surveys 57 (12) (2025). doi:10.1145/3742472." (no article number) | Add the article number from the DOI record | Article number present | no |
| REF-07 | [15] | P1 | SIGCOMM 2024, no pages | Add pages from the DOI record (10.1145/3651890.3672269) | Pages present | no |
| REF-08 | [18] | P1 | SODA, SIGCOMM 2024, no pages | Add pages from the DOI record (10.1145/3651890.3672260) | Pages present | no |
| REF-09 | [22], [25] | P2 | arXiv-only preprints ([22] is a 2026 preprint listed as a main competitor in Table 1) | Check for peer-reviewed versions. If none exist, keep them but label them clearly as preprints | The peer-reviewed version is cited where one exists | no |
| REF-10 | [26] | P1 | The Sutton & Barto textbook is cited for "Their neural policies combine throughput histories, buffer state, and manifest features [26]" (p.3) | At that location, replace the citation with the existing keys for [3] and [27] (Pensieve, Comyco), or remove it there | The claim cites ABR systems, not a textbook | no |
| REF-11 | [33], [35] | P1 | NeurIPS 2019 / 2018, no volume or pages | Add volume and pages from the proceedings | Volume and pages present | no |
| REF-12 | [38] | P1 | "Measuring broadband america - fixed broadband reports, ... ongoing program; data used for training and evaluation (2015)"; the URL is broken ("ongoreports") | Remove the free-text note. Format it as a dataset (organization, title, year, URL, access date) | Clean dataset entry | no |
| REF-13 | [39] | P1 | The NorNet Edge platform paper (Computer Networks 61, 2014) is cited as the source of the "Norway/Nornet mobile-broadband mobility logs". The Norway traces common in ABR work are usually the 3G/HSDPA commute traces of Riiser et al. (MMSys 2013; unverified, see Section 9). [39] is also cited for "capped catalogue diversity [39]" (p.11), which is unrelated | Ask the authors which dataset was actually used and cite that dataset. Remove [39] from the catalogue-diversity sentence | The citation matches the actual data source, and there is no unrelated use | **yes** |
| REF-14 | [17] vs. Table 1 | P2 | Table 1 lists BOLA as 2018; ref [17] is the 2020 IEEE/ACM ToN version | Make the year consistent (use the year of the cited version) | Table 1 and [17] agree | no |
| REF-15 | all | P1 | Journal names are not LTWA-abbreviated (e.g., "IEEE Communications Surveys & Tutorials" should be "IEEE Commun. Surv. Tutor."; "Computer Networks" should be "Comput. Netw."). Proceedings names are inconsistent ("Proc. AAAI Conf. Artificial Intelligence", "Proc. International Conf. Machine Learning"). Many DOIs are missing | Apply LTWA abbreviations consistently. Use one style for proceedings names. Add verified DOIs | Consistent abbreviations, and a DOI wherever one exists | no |

### 8.3 Minor format and figure tasks (P2)

| ID | Location | Problem | Required action | Acceptance | Needs-author-input |
|---|---|---|---|---|---|
| FMT-07 | Whole document | Uses `elsarticle` rather than the recommended CAS | Optionally migrate to `cas-dc` using `/workspace/computer_networks_template/starter_cas/main.tex` (use `\credit`, `\printcredits`, `highlights`, and `\bio`). Keep `natbib` numeric citations with `elsarticle-num.bst` | Compiles, and all content is preserved | no (the authors decide whether to migrate) |
| FMT-08 | Keywords | "Certified Safety" is vague | Offer an alternative (Section 6.3) | The authors decide | yes |
| FMT-09 | p.1 email footnote; PDF metadata | A superscript "a" follows each name in the email list ("Saeed Zarbi^a"). The PDF metadata author field reads "Saeed Zarbia; Leili Farzinvasha; Pedram Salehpoura" | Remove the superscripts in the email list. Set `\hypersetup{pdfauthor={Saeed Zarbi, Leili Farzinvash, Pedram Salehpour}}` | `pdfinfo` shows the correct authors | no |
| FMT-10 | p.6 §4.2; p.3 Table 2 | "Algorithm 1 realizes (7)–(6)."; "((2))" | Change to "(6)–(7)" and use `\eqref` / "Eq.~\eqref" consistently | Fixed | no |
| FMT-11 | p.12 | Appendix A sits between the Conclusion and the declarations | Reorder per Section 7.5 | Order as specified | no |
| FMT-12 | p.6 | The "Evaluation arms" paragraph is in §4 (Method) | Move it to the start of §5.1 | Moved, and references still resolve | no |
| FIG-06 | Fig. 4, p.8 | Seven identical violins (the text says "host-invariant by construction") | Remove the figure (comment it out per the guardrails) and keep one sentence, or replace it with a 5G vs. broadband comparison (authors decide) | Logged | yes (choice) |
| FIG-07 | Fig. 5, p.9 | Heavy overplotting and small legend markers | Add alpha transparency and a larger legend | Legible | no |
| FIG-08 | Fig. 6, p.9 | The Safety and Certified CDFs overlap | Add an inset zoomed on the region where they differ | The difference is visible | no |
| FIG-09 | Fig. 8 right panel, p.10 | A straight line ΔQoE = 2.93 − 0.55w that adds little | Remove the panel (report w* in the text) or add bootstrap bands (authors decide) | Logged | yes (choice) |
| FIG-10 | Fig. 9 vs. Table 7; Fig. 10 vs. Tables 8-9 | Duplicate content | Keep one form of each pair, or state that the figure visualizes the table | Logged | yes (choice) |
| FIG-11 | Table 1, p.3 | The "Paradigm" column wraps ("Risk-controlled / ABR"). The rows are not in chronological order (RobustMPC 2015 after BOLA 2018). The meaning of "≈" is unclear | Use `table*` or abbreviate; sort rows by year; define "≈" explicitly in the table note | Clean layout | no |
| FIG-12 | All plots | Fonts are DejaVu Sans Type 3 and do not match the Times body text | Regenerate with matplotlib `rcParams['pdf.fonttype']=42`, `rcParams['ps.fonttype']=42`, a serif/Times family, and 7-9 pt text at the final column width | `pdffonts main.pdf` lists no Type 3 fonts from figures | no (needs the plotting scripts) |

Things that already work (do not break them): all figures are vector graphics with no raster images, the axes have labels and units, section numbering follows 1 / 1.1 / 1.1.1, equations are numbered, and citation numbering is in order of first appearance.

---

## 9. Related-work suggestions: UNVERIFIED

> **UNVERIFIED: the agent must check existence, bibliographic details, and relevance before citing. Do not fabricate.** These come from reviewer memory, not from the paper. For each item, confirm the exact title, authors, venue, year, pages, and DOI from an authoritative source (publisher page, DOI resolver, DBLP). Get author approval before adding anything. If an item cannot be verified, drop it and log that.

| Topic | Suggested work (UNVERIFIED) | Why it may be relevant |
|---|---|---|
| Safety fallback for learned ABR | H. Mao et al., "Towards Safe Online Reinforcement Learning in Computer Systems", NeurIPS ML for Systems Workshop, 2019 | An early runtime safety fallback for RL-based ABR; a direct precedent for a shield |
| Uncertainty-triggered fallback | N. H. Rotman, M. Schapira, A. Tamar, "Online Safety Assurance for Learning-Augmented Systems", ACM HotNets, 2020 | Switches to a safe default policy under uncertainty in ABR; close to CPS's motivation |
| Online conformal under shift | I. Gibbs, E. Candès, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS, 2021 | The standard online-conformal method for non-exchangeable time series; relevant to SCI-10 and SCI-14 |
| Recent learned ABR | Z. Xia et al., "Genet: Automatic Curriculum Generation for Learning Adaptation in Networking", ACM SIGCOMM, 2022 | Recent learned ABR with a focus on generalization |
| LLM-based networking and ABR | D. Wu et al., "NetLLM: Adapting Large Language Models for Networking", ACM SIGCOMM, 2024 | Recent state of the art for learned ABR |
| QoE sensitivity | X. Zhang et al., "SENSEI: Aligning Video Streaming Quality with Dynamic User Sensitivity", USENIX NSDI, 2021 | Quality-sensitivity-aware ABR, related to perceptual banking |
| Real 5G throughput datasets | A. Narayanan et al., "Lumos5G: Mapping and Predicting Commercial mmWave 5G Throughput", ACM IMC, 2020 (and its dataset) | Counters the "public 5G logs are scarce" claim (SCI-13) |
| Real 5G throughput datasets | D. Raca et al., "Beyond Throughput, The Next Generation: A 5G Dataset with Channel and Context Metrics", ACM MMSys, 2020 | Public 5G traces usable for ABR replay |
| Norway traces | H. Riiser et al., "Commute Path Bandwidth Traces from 3G Networks: Analysis and Applications", ACM MMSys, 2013 | Probable source of the Norway traces (REF-13); confirm with the authors |
| VMAF (peer-reviewed) | A peer-reviewed VMAF paper by the Netflix and academic authors (find via DBLP) | Supplements the blog reference [8] |
| Journal fit | Recent (2023-2026) *Computer Networks* papers on ABR, video streaming QoE, or safe RL for networking | Shows fit with the journal; search the journal site and verify each |

---

## 10. Final submission checklist

### Formal (journal rules)
- [ ] FMT-01 Abstract ≤ 250 words, no citations, no undefined abbreviations, numbers match Table 5
- [ ] FMT-02 `Highlights.txt`: 3-5 bullets, each ≤ 85 characters, numbers verified
- [ ] FMT-03 Editable source package compiles from a clean folder (`.tex`, `.bib` and `.bbl`, figures, class files)
- [ ] FMT-04 Data availability names a repository with a DOI, or gives a reason; the artifact is cited
- [ ] FMT-05 Generative-AI declaration added with the exact title before the References, or the authors' confirmation of no AI use is logged
- [ ] FMT-06 Three biographies (≤ 100 words each) with photos
- [ ] Keywords: 1-7, none with "and"/"of" (currently 6)
- [ ] Citations numbered in square brackets in order of first appearance (recheck after adding references)
- [ ] REF-01 to REF-15 done; every reference has authors, title, year, volume, and pages or article number; LTWA abbreviations applied
- [ ] CRediT, competing-interest, and Funding statements present
- [ ] Optional: graphical abstract (531 x 1328 px or larger)
- [ ] Optional: migration to `cas-dc` (FMT-07); FMT-08 to FMT-12 done

### Scientific consistency
- [ ] SCI-01 Monotonicity statements consistent (author-confirmed)
- [ ] SCI-02 Table 8/9 vs. Table 5 numbers reconciled
- [ ] SCI-03 Table 8 headers corrected
- [ ] SCI-04 Fig. 7 and text reconciled; rolling window defined
- [ ] SCI-05 Chunk count consistent; per-chunk claims cite Fig. 2, not Table 3
- [ ] SCI-06 / SCI-07 Ranges and intervention rates match Table 5
- [ ] SCI-08 Test procedure and p-values confirmed
- [ ] SCI-09 Proposition 1 hypotheses and scope corrected (author-approved)
- [ ] SCI-10 Missing baselines added or their absence justified
- [ ] SCI-11 "Prior VMAF-ranking shields" cited or rephrased
- [ ] SCI-12 QoE reported in Table 5
- [ ] SCI-13 Synthetic 5G generator fully described; scarcity claim fixed
- [ ] SCI-14 Residual-window behavior across episodes stated
- [ ] SCI-15 to SCI-27 addressed, or explicitly deferred with the authors' agreement

### Figures, tables, and language
- [ ] FIG-01 to FIG-12 done; `pdffonts` shows no Type 3 fonts from figures; the smallest figure text is ≥ 7 pt
- [ ] LANG-19 present-tense pass done; LANG-01 to LANG-18 applied
- [ ] LANG-20 internal jargon removed (`episodes.csv`, "improved", "harness", "legacy")

### Hygiene
- [ ] `grep -n "authorcheck" *.tex` returns nothing (all author inputs resolved and the macro definition removed)
- [ ] `latexmk -pdf` finishes with no errors, no undefined references or citations, and no overfull boxes > 5 pt
- [ ] `CHANGELOG_revision.md` lists every change with its task ID
- [ ] No content removed without a changelog entry
- [ ] Every new citation verified (Section 9) and approved by the authors
- [ ] The final PDF metadata (`pdfinfo`) shows the correct title and authors
