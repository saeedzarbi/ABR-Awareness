# Revision Brief, Phase B: *Certified Perceptual Shielding for Adaptive Bitrate Streaming*

This brief is for the AI agent that edits the LaTeX source. It replaces the Phase A brief (`paper_revision_brief_computer_networks_EN.md`) as your working document. Use the Phase A brief only to look up detail that is not repeated here. If the two disagree, **this brief wins**.

---

## 1. Context

| Field | Value |
|---|---|
| Paper | *Certified Perceptual Shielding for Adaptive Bitrate Streaming*, by Saeed Zarbi, Leili Farzinvash, and Pedram Salehpour (corresponding author: psalehpour@tabrizu.ac.ir), University of Tabriz |
| Target journal | Elsevier *Computer Networks* (ISSN 1389-1286). Abstract ≤ 250 words; highlights 3–5 bullets, each ≤ 85 characters, in a separate file; keywords 1–7; data statement Option C; AI declaration titled exactly "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", placed before the References; CRediT; competing interests; biographies ≤ 100 words each with a photo |
| Source files | `overleaf_upload/main.tex` (primary), `main.ltx` (keep in sync), `tables/table_ablation_eps.tex`, `overleaf_upload/Highlights.txt` |
| Current PDF | Revision v2: 15 pages, `elsarticle`, 45 references, 10 figures, 9 tables |
| Phase A output | `CHANGELOG_revision.md` (Phase A) and the v2 PDF |
| Rung bitrates (for reference) | 300 / 750 / 1200 / 1850 / 2850 / 6000 kb/s |

### What Phase A did

**Done well:**
- Replaced the abstract. The new one is 251 words, so it is still one word over the limit.
- Added `Highlights.txt`.
- Corrected the ranges (3.8–4.5% and 8.9–9.3%).
- Made the intervention rates match Table 5.
- Toned down overclaims (SCI-15, SCI-21).
- Renamed the predictive variant to CPS-P.
- Removed `episodes.csv`, "harness", and "legacy".
- Added the AI declaration in the correct place.
- Regenerated Tables 8 and 9 so that the default cell matches Table 5.
- Replotted Fig. 7 with a rolling window.

**Not done:** 60 of 86 tasks, including every reference fix, every figure task, and most SCI tasks.

**Process problems we must not repeat:**
1. No `\authorcheck` markers appeared in the PDF, even though 23 tasks needed author input.
2. The Phase A changelog lists SCI-01, SCI-02, SCI-04, FMT-04, and FMT-05 **as both done and deferred**.
3. Some changes added new factual claims to the text without showing a source in the manuscript:
   - per-rung ladder structure;
   - "same seeding protocol";
   - rolling window W=20;
   - a DOI "planned upon acceptance";
   - the name of the AI tool.

   The Phase A agent says these claims are data-backed. They still need the authors to verify them, and wherever possible they must be traceable to a source file.

### Corrections to the Phase A brief (reviewer errors, now fixed)

- **Phase A brief §6.2:** it said that appending "under a wider risk budget" keeps the abstract under 250 words. **That was wrong** (246 + 5 = 251). See task FMT-01 below for the fix.
- **SCI-03 (Table 8 headers "swapped") was a false positive.** It came from a text-extraction artifact. The v1 PDF already had correct headers ("Rebuffer Δ" holds −3.5…−46.1 and "VMAF Δ" holds +0.02…−0.48). The Phase A agent was right not to swap them. Only a verification remains (task SCI-03, P2).

---

## 2. Hard rules (guardrails)

These rules override any task text. If you break one, the task is **not Done**.

1. **Never invent facts.** This covers numbers, experimental results, p-values, confidence intervals, seeds, trace or dataset details, encoder settings, hyperparameters, hardware, software versions, citations, DOIs, URLs, page or article numbers, repository names, licences, author biographies, photos, and AI tool names or versions.
2. **Any fact that only the authors know gets a visible marker.** Define the macro once in the preamble:
   ```latex
   \usepackage{xcolor}
   \newcommand{\authorcheck}[1]{\textcolor{red}{[AUTHOR: #1]}}
   ```
   Every marker must appear in the compiled PDF (red) **and** in the changelog. Do not hide markers in LaTeX comments.
3. **Do not re-run, re-train, or re-simulate anything unless the authors explicitly tell you to.** *Replotting* existing outputs (for example to fix fonts) is allowed only when:
   - the plotting script reads existing result files unchanged;
   - you record the script name, the input files, and the command in the changelog;
   - you confirm that the plotted values are unchanged.
4. **Every numeric change must cite its source of truth in the changelog.** Give the file path plus the row, column, or key, or the script and the command. If you cannot name a source, do not change the number; insert `\authorcheck{}` instead.
5. **Every new factual sentence must be traceable** to a file, to an existing sentence or table in the manuscript, or to an author instruction recorded in the changelog. Otherwise mark it with `\authorcheck{}`.
6. **Flag contradictions; do not resolve them silently.** When two places disagree and the source data does not settle it, put `\authorcheck{}` at both places.
7. **New citations need verification.** Check the title, authors, venue, year, pages or article number, and DOI against the DOI resolver, the publisher page, or DBLP. Record the verification URL in the changelog. Anything you cannot verify must not be cited. Adding related-work citations also needs author approval (use `\authorcheck{approve citation ...}`).
8. **Keep the AI declaration truthful.** It must cover everything an AI tool did to the manuscript, including regenerating tables and figures and editing scientific text, not only language (see FMT-05).
9. **A task is Done only if it is fully closed.** Use exactly one of these statuses per task in the changelog:
   - `Done`: fully applied, with evidence.
   - `Deferred-authorcheck`: a marker has been inserted and awaits the authors.
   - `Verified-no-change`: checked, and nothing needed to change.
   - `Blocked`: needs a missing file or tool. Say which.

   **Never list a task as both Done and deferred.** If only part of a task is done, split it into sub-items (for example `SCI-05a Done`, `SCI-05b Deferred-authorcheck`).
10. **Produce `CHANGELOG_revision_phaseB.md`** next to `main.tex`. Use one row per change, with these columns:

    | Task ID | Status | File | Line / section | Before (exact) | After (exact) | Source of truth | Note |
    |---|---|---|---|---|---|---|---|

    At the end, add:
    - a list of every `\authorcheck{}` with its location and the question it asks;
    - a status count table covering all 86 IDs plus NEW-01;
    - the output of the acceptance-check commands in Section 7.
11. **Keep the document compilable.** Run `latexmk -pdf main.tex` after each group of edits. It must finish with no errors, no undefined references or citations, and no new overfull boxes wider than 5 pt. Keep `main.ltx` in sync with `main.tex`.
12. **Keep existing `\label`/`\cite` keys.** Do not delete content without a trace: comment it out with `%% REMOVED (TASK-ID)` and log it.
13. **Scope.** Do only the Phase B tasks in Section 4. Do not "improve" other text.

---
## 3. Status carry-over (all 86 Phase A IDs, as of revision v2)

**Legend:**
- **Done**: closed, and no Phase B action is needed.
- **Partial**: something remains to do.
- **Not done**: untouched.
- **Needs verification**: the change was made, but the authors must confirm its source or truth.

"B" in the last column means the item appears in the Phase B task list (Section 4).

**Counts:** Done 14 · Partial 9 · Needs verification 3 · Not done 60. Phase B also adds one new task, NEW-01.

### FMT

| ID | Pri | Phase-A status | Evidence / remaining gap | B |
|---|---|---|---|---|
| FMT-01 | P0 | Partial | The abstract was replaced, but it is **251 words** | B |
| FMT-02 | P0 | Partial | `Highlights.txt` exists (5 bullets, 80–82 characters). Highlight 1 overclaims ("any") | B |
| FMT-03 | P0 | Needs verification | The source package exists in `overleaf_upload/`. A clean-folder compile has not been demonstrated | B |
| FMT-04 | P0 | Partial | The statement is "upon reasonable request" with a DOI "planned". It is circular about the 5G generators and has no deposit | B |
| FMT-05 | P0 | Partial | The title and placement are correct. The text understates AI use and has no version | B |
| FMT-06 | P0 | Not done | No biographies or photos | B |
| FMT-07 | P2 | Not done | Still `elsarticle` (optional) | B |
| FMT-08 | P2 | Not done | Keywords unchanged ("Certified Safety") | B |
| FMT-09 | P2 | Not done | The email footnote shows "Zarbi^a". PDF metadata reads "Saeed Zarbia; Leili Farzinvasha; Pedram Salehpoura" | B |
| FMT-10 | P2 | Not done | "Algorithm 1 realizes (7)–(6)" (§4.2) and "((2))" (Table 2) | B |
| FMT-11 | P2 | Not done | Appendix A still sits between the Conclusion and the end matter | B |
| FMT-12 | P2 | Not done | The "Evaluation arms.." paragraph is still in §4 | B |

### SCI

| ID | Pri | Phase-A status | Evidence / remaining gap | B |
|---|---|---|---|---|
| SCI-01 | P0 | Needs verification | Monotonicity text now agrees with Table 3. The new per-rung claims (§3.3, §5.5, §6) are not shown in any table, and the Prop. 2 "upper three rungs" extension is not proved | B |
| SCI-02 | P0 | Needs verification | Path A: the default cell was reused from `greedy_5g` and now matches Table 5. Whether the other rows share seeds and traces is unconfirmed | B |
| SCI-03 | P0→P2 | Done | Headers were already correct (the Phase A brief was wrong). The ε=1 VMAF Δ now reads −0.01, consistent with §5.2. Verify only | B (verify) |
| SCI-04 | P0 | Partial | Fig. 7 was replotted (replot-only, W=20). The caption understates how long the curve stays below target, and the symbol W clashes with the residual window W=200 | B |
| SCI-05 | P0 | Partial | The Table 3 → Fig. 2 citations and the four-title scope were fixed. The Fig. 2 caption still says "173 chunks" while §5.5 and Table 7 say 113 | B |
| SCI-06 | P0 | Done | Abstract and §7: 3.8–4.5% and 8.9–9.3% | – |
| SCI-07 | P0 | Done | §5.2 intervention rates match Table 5 | – |
| SCI-08 | P0 | Not done | Identical p-values. §5.4 still says "intervals were reported as 5.0×10⁻⁵" | B |
| SCI-09 | P0 | Not done | Prop. 1 hypothesis gap. B_crit branch not covered. F in Prop. 2 ≠ Eq. (6) | B |
| SCI-10 | P0 | Not done | No fixed-factor safety baseline, no competitor, no justification | B |
| SCI-11 | P0 | Not done | "Prior VMAF-aware ranking shields" (§1, §4.2) still have no citation | B |
| SCI-12 | P0 | Not done | QoE (Eq. 2) is still not reported | B |
| SCI-13 | P0 | Not done | 5G generator not described. The "remained scarce" claim is still there | B |
| SCI-14 | P0 | Not done | W=200 vs K_e=48 is not explained | B |
| SCI-15 | P1 | Done | ε_risk = 4 is stated in contribution 3, RQ3, §5.3, and §7 | – |
| SCI-16 | P1 | Not done | 38 s rebuffering and 12 s buffer not justified | B |
| SCI-17 | P1 | Not done | Construction of the 48-chunk episodes not described | B |
| SCI-18 | P1 | Not done | "H≥6", f, encoding ladder, and software versions are missing | B |
| SCI-19 | P1 | Not done | No cluster CIs, no BOLA/MPC p-values, no Holm table | B |
| SCI-20 | P1 | Not done | 204 episodes from 27 traces not explained | B |
| SCI-21 | P1 | Done | All five rewordings applied | – |
| SCI-22 | P1 | Not done | Future work still says "RobustMPC and Comyco" | B |
| SCI-23 | P1 | Not done | §3.4.1 "Let Ĉk > 0 be the trace-derived effective throughput" is still there | B |
| SCI-24 | P1 | Not done | No primary PPO row in Table 5 | B |
| SCI-25 | P1 | Not done | Five contribution bullets and no problem statement | B |
| SCI-26 | P1 | Not done | Related work unchanged | B |
| SCI-27 | P1 | Not done | Single seed; section not shortened | B |

### FIG

| ID | Pri | Phase-A status | Evidence / remaining gap | B |
|---|---|---|---|---|
| FIG-01 | P1 | Not done | Fig. 1 text still about 4 pt | B |
| FIG-02 | P1 | Not done | Fig. 2 labels small or overlapping; "173" | B |
| FIG-03 | P1 | Not done | Fig. 3 mixes units; "BB greedy" label | B |
| FIG-04 | P1 | Not done | Fig. 10 caption claims VMAF is shown in the left panel, but it is not plotted | B |
| FIG-05 | P1 | Not done | Table 5 has no QoE column and no PPO block | B |
| FIG-06 | P2 | Not done | Fig. 4 (seven identical violins) still present | B |
| FIG-07 | P2 | Not done | Fig. 5 overplotting | B |
| FIG-08 | P2 | Not done | Fig. 6 has no inset | B |
| FIG-09 | P2 | Not done | Fig. 8 right panel still present | B |
| FIG-10 | P2 | Not done | Fig. 9 duplicates Table 7; Fig. 10 duplicates Tables 8–9 | B |
| FIG-11 | P2 | Not done | Table 1 unchanged | B |
| FIG-12 | P2 | Not done | `pdffonts` still shows DejaVu Type 3 fonts, even in the replotted Fig. 7 | B |

### LANG

| ID | Pri | Phase-A status | Evidence / remaining gap | B |
|---|---|---|---|---|
| LANG-01 | P1 | Done | Resolved by the new abstract | – |
| LANG-02 | P1 | Done | Abstract and §7 | – |
| LANG-03 | P1 | Done | Resolved by the new abstract | – |
| LANG-04 | P1 | Not done | "We introduced a Certified Perceptual Shield (CPS) that targets this structure" | B |
| LANG-05 | P1 | Not done | "sub-6 GHz and 5G/mmWave" (§1, §3.1) | B |
| LANG-06 | P1 | Not done | "This section formalized …", "Table 2 collected …" | B |
| LANG-07 | P2 | Not done | "((2))" | B |
| LANG-08 | P2 | Not done | Table 3 note: "… in kbps" | B |
| LANG-09 | P1 | Not done | "we fell back to a fixed scale" | B |
| LANG-10 | P2 | Not done | "Evaluation arms..", "Coverage under injected drift.." | B |
| LANG-11 | P1 | Partial | Named CPS-P, but "ran with … H≥6" remains | B |
| LANG-12 | P1 | Done | §5.2 | – |
| LANG-13 | P1 | Not done | §5.4 "intervals were reported as" | B |
| LANG-14 | P1 | Done | "episodes.csv" removed | – |
| LANG-15 | P1 | Done | §6 | – |
| LANG-16 | P1 | Done | §6 | – |
| LANG-17 | P1 | Done | §3.3 | – |
| LANG-18 | P1 | Done | §5.4 | – |
| LANG-19 | P1 | Partial | Only a few tense fixes; §1, §3, and §4 are mostly still past tense | B |
| LANG-20 | P1 | Partial | Fixed in the text, but the **Fig. 8 left-panel title** still reads "Certified arm (improved shield)" | B |

### REF

| ID | Pri | Phase-A status | Evidence / remaining gap | B |
|---|---|---|---|---|
| REF-01 | P2 | Not done | [3] "pensieve" | B |
| REF-02 | P2 | Not done | [4] duplicated arXiv identifier | B |
| REF-03 | P1 | Not done | [6] Cisco web reference | B |
| REF-04 | P1 | Not done | The [8] URL overflows into [21] | B |
| REF-05 | P2 | Not done | [13] year | B |
| REF-06 | P1 | Not done | [14] article number | B |
| REF-07 | P1 | Not done | [15] pages | B |
| REF-08 | P1 | Not done | [18] pages | B |
| REF-09 | P2 | Not done | [22] and [25] preprint labels | B |
| REF-10 | P1 | Not done | [26] textbook cited for an ABR claim | B |
| REF-11 | P1 | Not done | [33] and [35] volume/pages | B |
| REF-12 | P1 | Not done | [38] free-text note and broken URL | B |
| REF-13 | P1 | Not done | [39] Norway source; "catalogue diversity [39]" | B |
| REF-14 | P2 | Not done | BOLA year: Table 1 vs [17] | B |
| REF-15 | P1 | Not done | LTWA abbreviations and DOIs | B |

---
## 4. Phase B task list

**Format of each task:**
- **Loc**: location in the v2 PDF (page / section).
- **Problem**: what is wrong.
- **Action**: what to do.
- **Accept**: the acceptance criteria.
- **Author**: `Needs-author-input` (yes / no).

Quotes come from the v2 PDF text layer. Math may appear flattened, so search the `.tex` for the surrounding words.

**Totals: 74 tasks. P0: 18 · P1: 34 · P2: 22. Of these, 32 need author input (P0: 13, P1: 14, P2: 5).**

### 4.1 P0: blocking (18)

#### NEW-01: Reconcile the Phase A changelog
- **Loc:** `CHANGELOG_revision.md` (Phase A)
- **Problem:**
  - SCI-01, SCI-02, SCI-04, FMT-04, and FMT-05 are listed as both done and "Deferred".
  - The SCI-03 entry says "fixed `-0.00` → `0.00`", but the v2 PDF shows `-0.01` in the ε=1.0 row. That value comes from the regenerated default cell and matches the greedy value in §5.2.
- **Action:** In `CHANGELOG_revision_phaseB.md`, give each of these five tasks exactly one final status (Rule 9), with evidence. Record that the SCI-03 VMAF Δ cell is `-0.01` and name its source file.
- **Accept:** No task appears with two statuses. Every Phase A number change has a source of truth.
- **Author:** no

#### FMT-01: Abstract is 251 words
- **Loc:** Abstract, p.1
- **Action:** Replace the final sentence.
  - Before: `These results show that the benefit of banking depends on ladder saturation and network regime.`
  - After: `Thus, banking benefit depends on ladder saturation and network regime.`
- **Accept:** A recount *in the file* gives ≤ 250 words (the expected count is 246). Use `detex` on the abstract environment piped to `wc -w`, and paste the command and its output into the changelog. Keep "under a wider risk budget".
- **Author:** no

#### FMT-02: Highlight 1 overclaims
- **Loc:** `Highlights.txt`, line 1
- **Problem:** "wraps any client-side ABR controller". The evaluated hosts are a finite set.
- **Action:** Replace with the text in Section 5.2.
  - Note: the literal wording "…that wraps existing client-side ABR controllers" is **86 characters, which exceeds 85**. Do not use it.
  - Keep bullets 2–5 unchanged.
- **Accept:**
  - 3–5 bullets.
  - Each bullet ≤ 85 characters including spaces, excluding the leading "- " marker. Check with `sed 's/^- //' Highlights.txt | awk '{print length}'` and paste the output.
  - The "~9%" in bullet 4 still matches Table 5 / §5.2 (9.3% for greedy; 8.9–9.3% across the four hosts).
- **Author:** no

#### FMT-03: Editable source package compiles cleanly
- **Loc:** `overleaf_upload/`
- **Action:**
  - Copy the folder to a fresh directory and run `latexmk -pdf main.tex`.
  - Include the `.bib` and `.bbl`, all figure PDFs, `Highlights.txt`, and the photo placeholders.
  - List the files in the changelog.
- **Accept:** A clean compile with no missing files, no undefined references, and no "??".
- **Author:** no

#### FMT-04: Data availability
- **Loc:** "Data availability", p.13
- **Problem:**
  - "available from the corresponding author upon reasonable request. A persistent archive with a DOI is planned for deposit upon acceptance." Option C expects the data to be deposited and cited, *or* a stated reason why it cannot be shared.
  - "Synthetic 5G trace generators … are included with the code release described above" is circular, because that release is only on request.
  - The claim about planned DOI timing came from the Phase A agent, not from the authors.
- **Action:**
  - Replace the statement with **Option A or Option B** from Section 5.4.
  - Insert `\authorcheck{}` for the repository, DOI, licence, and any data that cannot be shared.
  - Do **not** state that a deposit is planned unless the authors confirm it in writing (record their confirmation in the changelog).
  - The Norway source depends on REF-13.
- **Accept:** One of the options is filled with author-supplied facts, or clearly marked with `\authorcheck{}`. There is no circular wording and no invented URL or DOI.
- **Author:** **yes**

#### FMT-05: AI declaration must match actual use
- **Loc:** p.13, the section just before the References
- **Problem:**
  - The text says Claude was used only "to improve language, clarity, and academic wording".
  - In Phase A, AI assistance also regenerated tables and figures (Tables 8–9, Fig. 7) and edited scientific text.
  - No version is given.
- **Action:** Replace the body with the template in Section 5.3. Keep the exact title and placement.
- **Accept:**
  - The exact title is used.
  - The section sits immediately before the References.
  - The version is filled in by the authors, or `\authorcheck{}` is present.
  - The scope covers language editing and table/figure regeneration.
  - The changelog records the authors' confirmation.
- **Author:** **yes** (version, and confirmation of scope)

#### FMT-06: Author biographies and photos
- **Loc:** End matter, after the References
- **Action:** Insert the scaffold from Section 5.5 for all three authors, with `\authorcheck{}` for the text and for the photo files.
- **Accept:** Three bios, each ≤ 100 words (counted), author-supplied, each with a photo, or visible markers. The document still compiles while photos are missing (use the `\IfFileExists` guard).
- **Author:** **yes**

#### SCI-01: Per-rung ladder claims need visible evidence
- **Loc:**
  - §3.3, p.4: "Mid-ladder inversions concentrate at the 750→1200 kbps step; the top three rungs (1850, 2850, 6000 kbps) remain nondecreasing on every title." and "The same collapse argument applies to the upper three rungs on every title, which is the region where greedy banking mostly acts."
  - §5.5, p.10: "the other eight are non-monotone only at mid-ladder (750→1200 kbps)"
  - §6, p.11: "on the eight mid-ladder-inverted titles it still applies to the upper three rungs, where greedy banking concentrates"
- **Problem:**
  - Table 3 shows only whether each ladder is monotone (Yes/No), so readers cannot check these claims.
  - "greedy banking mostly acts / concentrates" in the upper rungs is an empirical claim with no reported statistic.
  - The claim that Prop. 2 "applies to the upper three rungs" is not proved. Prop. 2 assumes a nondecreasing ladder over the whole feasible set F. It carries over to a subset of rungs only if F lies entirely within that monotone segment.
- **Action (choose A or B, and log which):**
  - **A. Evidence exists.** Generate a small per-title table from the actual per-title session-mean ladder files. List the file paths and the script in the changelog. Use rows = 12 titles and columns = VMAF at the six rungs (or the sign of each adjacent difference). Put it in Appendix A (for example "Table A.1: Per-title session-mean VMAF ladder") and cite it at all three locations.
    - Do not type the numbers by hand. Export them with a script from the existing files.
    - If the rung-level distribution of banking decisions ("where greedy banking concentrates") exists in the existing per-episode outputs, report one sentence with the share of banking actions whose proposal is in the top three rungs, and cite the source file. Otherwise remove or soften the "mostly acts / concentrates" clauses.
  - **B. No accessible evidence.** Insert `\authorcheck{confirm per-rung ladder structure; supply ladder files}` at all three locations. Soften the text to what Table 3 supports, for example: "Eight of twelve session-mean ladders are non-monotone somewhere (Table 3); Proposition 2 applies in full to the four nondecreasing titles."
  - **In both cases:** replace the Prop. 2 extension with a correct scoped statement, for example: "Proposition 2 also applies at any step whose feasible set lies within a nondecreasing segment of the ladder." Add `\authorcheck{approve scoped statement}`.
- **Accept:** Every per-rung claim is backed by a table produced from files, or is softened and marked. The Prop. 2 scope is stated correctly.
- **Author:** **yes** (confirm the data, approve the wording)

#### SCI-02: Ablation rows must share the headline protocol, or say so
- **Loc:** Table 8 and Table 9 captions and §5.6, p.11
- **Current text:**
  - "The ε=1.0 row is the headline evaluation in Table 5; other rows vary ε only." (and the same wording for α)
  - "The default cell (ε=1.0, α=0.10) is the same evaluation as Table 5; the other cells are a one-parameter sensitivity sweep under the same seeding protocol."
- **Problem:** Phase A reused the default cell from the main `greedy_5g` run. Whether the other rows (ε = 0.5, 2, 4; α = 0.05, 0.20) were run with the **same seeds, trace set, and shield version** is not documented.
- **Action:**
  - Look for evidence in the run configurations or logs of the sweep (seed lists, trace manifest, commit hash).
  - If they match the `greedy_5g` run, keep the sentences and cite the config files in the changelog.
  - If they do not match, or you cannot tell, replace "under the same seeding protocol" with `\authorcheck{confirm sweep rows share seeds/traces with Table 5; else state the difference}` and add the same marker to both captions.
  - Do not change any numbers.
- **Accept:** The captions' claims are backed by cited configs, or marked. No row mixes protocols without a stated difference.
- **Author:** **yes**

#### SCI-04: Fig. 7 caption and the window symbol
- **Loc:** §5.3 "Coverage under injected drift", Fig. 7 caption (p.9), and §4.5 (p.6)
- **Problem:**
  - The caption says "the rolling curve is below target during warm-up by construction". The plotted curve stays below 0.90 until about chunk 37–38, which is the end of the shock band (chunks 28–37), not the end of warm-up (chunk 19). The §5.3 text states this correctly.
  - "W" names both the residual window (W=200, §4.5 and Table 4) and the rolling-plot window (W=20).
- **Action:**
  1. Rename the plot window to $W_{\mathrm{roll}}$ in §5.3, the Fig. 7 caption, **and the figure legend**. Replot the legend using `--replot-only` from the existing data and log the script and inputs. Leave $W=200$ unchanged.
  2. Replace the caption with the template in Section 5.6.
  3. Read the chunk where the curve first reaches 0.90 from the existing plotted data file and cite the file. If you cannot, use `\authorcheck{}`.
  4. In §5.3, add one clause explaining that a 20-chunk rolling window at chunk c still contains warm-up chunks until c ≥ 39, if that is how the rolling mean is computed. Otherwise use `\authorcheck{}`.
  5. Keep the phase means (0.928 / 0.929 / 0.944 with n = 1,632 / 2,040 / 2,040) unchanged.
- **Accept:**
  - The caption matches the plotted curve.
  - The crossing chunk is sourced or marked.
  - No symbol is used for two quantities.
  - Fig. 7 has no Type 3 fonts (see FIG-12).
- **Author:** no (yes only if the data file is unavailable)

#### SCI-05: 173 vs 113 chunks
- **Loc:** Fig. 2 caption (p.5): "across all 173 chunks". §5.5 and Table 7 (p.10) say "113 chunks".
- **Action:** Check the Fig. 2 plotting input (count the distinct chunks) and the Table 7 source.
  - If both numbers are real (for example 173 chunks across the four titles in Fig. 2, and 113 chunks in the ablation subset), keep both and add one clause saying which set each refers to.
  - Otherwise insert `\authorcheck{173 vs 113}` in both places.
- **Accept:** The counts are consistent or explained, with the source file cited, or marked.
- **Author:** **yes** (unless the source files settle it)

#### SCI-08: Identical p-values and test wording
- **Loc:** Table 5 note (p.9): "greedy bitrate p=8.5×10⁻⁴, rebuffering p=8.5×10⁻⁴". Table 6 note and §5.4 (p.10): "Paired Wilcoxon tests and episode-bootstrap intervals were reported as 5.0×10⁻⁵ (VMAF) and 5.0×10⁻⁵ (rebuffering)."
- **Action:**
  - Rewrite the §5.4 sentence as: "Paired Wilcoxon tests give $p = 5.0\times10^{-5}$ for both VMAF and rebuffering \authorcheck{confirm; identical values suggest a resampling floor}."
  - Add `\authorcheck{}` in §5.1 asking for the exact test (exact vs normal-approximation Wilcoxon, or permutation with N resamples), the zero handling ("zsplit" is already stated), and the true p-values.
- **Accept:** The wording no longer mixes up intervals and p-values. The test is named once in §5.1. The values are confirmed or marked.
- **Author:** **yes**

#### SCI-09: Proposition 1 hypothesis gap
- **Loc:** §4.1 Prop. 1 and its proof (p.5), Algorithm 1 lines 4–5, and the definition of F in Prop. 2 (p.6)
- **Problem:**
  - When $B_k - m < 0.1$, the condition $d_k(j) \le \max\{B_k - m, 0.1\}$ allows $d_k(j) > B_k$, so the proof step "does not exceed $B_k$" fails.
  - The $B_k \le B_{crit}$ branch returns rung 0 without any feasibility check.
  - F = {j : d_k(j) ≤ B_k − m} differs from Eq. (6).
- **Action:**
  - Insert the proposed corrected statement inside `\authorcheck{}`, for example: add the hypotheses "$B_k > B_{crit}$ and $B_k - m \ge 0.1$" and the sentence "Steps with $B_k \le B_{crit}$ are not certified."
  - Align F with Eq. (6), or explain the difference.
  - Do not replace the proposition text until the authors approve.
- **Accept:** The corrected text is approved by the authors and applied, or clearly marked.
- **Author:** **yes**

#### SCI-10: Missing baselines
- **Loc:** §5.1, §5.2 / Table 5, §6
- **Action:**
  - Insert `\authorcheck{}` in §5.1 listing the missing arms:
    - fixed-factor safety (for example 0.8 × harmonic mean, which is also CPS's own fallback, ρ_fb = 0.80);
    - at least one of [21]–[23], or a reason why none can be compared;
    - optionally an ACI bound.
  - Ask how the Pensieve-style policy was trained, since its operating point is 722 kb/s.
  - Add a §6 paragraph stub marked `\authorcheck{}`.
  - **Do not run any experiments.**
- **Accept:** Results supplied by the authors, or an author-written justification in §6.
- **Author:** **yes**

#### SCI-11: Uncited "prior VMAF-ranking shields"
- **Loc:** §1, p.1: "Prior VMAF-aware ranking shields provably collapse …". §4.2, p.6: "Banking distinguishes CPS from prior VMAF-ranking shields".
- **Action:** Insert `\authorcheck{cite prior VMAF-ranking shield(s) or approve rewording}` and propose rewording to "A natural VMAF-ranking shield …" / "a VMAF-ranking shield". Apply the rewording only when the authors approve, or when no citation is supplied by the end of Phase B (log which).
- **Accept:** Each mention is backed by a verified citation, or no longer implies prior published work.
- **Author:** **yes**

#### SCI-12: QoE is never reported
- **Loc:** §3.2 Eq. (2), Table 5, §5.2
- **Action:**
  - Add a "QoE" column to Table 5 with `\authorcheck{}` cells, and a stub sentence in §5.2.
  - If per-episode QoE values already exist in the result exports, you may compute session means **only** with a logged script, and you must state the β = 4.3 and μ = 1.0 weights used.
  - Otherwise leave the cells marked.
- **Accept:** The QoE values are sourced or marked.
- **Author:** **yes**

#### SCI-13: 5G trace generator
- **Loc:** §5.1 "Primary 5G suite", p.7 ("… remained scarce")
- **Action:**
  - Add a paragraph stub marked `\authorcheck{}` asking for the generator model, parameters, distributions, seeds, and any comparison with real 5G traces.
  - Soften "remained scarce" to: "we use synthetic traces to control dip and outage statistics; public 5G datasets (e.g., \authorcheck{verified citations}) are left for future evaluation".
  - Do not cite any dataset until it is verified (Rule 7).
- **Accept:** The generator is specified by the authors or marked, and the scarcity claim is removed or qualified.
- **Author:** **yes**

#### SCI-14: Residual window W=200 vs episode length K_e=48
- **Loc:** §4.5 (p.6), §5.1 (p.7)
- **Action:** Insert `\authorcheck{does the residual window persist across episodes/traces? relation to the 20-chunk warm-up?}`. Then add the one-sentence answer the authors supply.
- **Accept:** The behavior across episodes is stated explicitly and is consistent with the warm-up description.
- **Author:** **yes**

### 4.2 P1: important (34)

#### SCI and FIG tasks (P1)

| ID | Loc | Problem | Action | Accept | Author |
|---|---|---|---|---|---|
| SCI-16 | Table 5, §4.5, §5.2 | The certified greedy arm has 38.0 s of rebuffering per 192 s episode, with a 12 s buffer cap | Insert `\authorcheck{}`: justify the 12 s cap and the trace severity; give a quantitative breakdown of the residual rebuffering; optionally add a 30 s or 60 s buffer run (authors only). Add a rebuffering-ratio column **only if** it can be computed from the existing totals (ratio = s / 192 s), and log the formula | Justified or marked | yes |
| SCI-17 | §5.1 | How the 48-chunk episodes were built from short clips is not described | `\authorcheck{describe episode construction (looping/concatenation)}` | Stated by the authors | yes |
| SCI-18 | §3.3, §4.3, Table 4, §5.1, App. A | "H≥6"; f in Eq. (8) is unspecified; codec, resolutions, encoder settings, and VMAF model version are missing; PPO budget, learning rate, and batch size are missing; SB3, Python, and library versions are missing; hardware is "commodity CPU" | Add a "Reproducibility details" table in Appendix A with one `\authorcheck{}` per missing item. Fill a value **only** from config files, and cite the file | No "≥" placeholders; every value sourced or marked | yes |
| SCI-19 | §5.1, §5.2, Table 5 | Only episode-level Wilcoxon tests; no p-values for BOLA or MPC; Holm correction claimed but not shown; no CIs | Build an empty Holm / CI table skeleton marked `\authorcheck{}`. Do not compute anything new unless the authors instruct it | Skeleton plus markers, or author-supplied values | yes |
| SCI-20 | §5.1, §5.3 | 204 broadband episodes from 27 test traces is not explained | `\authorcheck{how were 204 episodes drawn from 27 traces?}` | Stated | yes |
| SCI-22 | §6 Future work, p.12 | "applying CPS to RobustMPC and Comyco hosts" (RobustMPC is already evaluated) | Change to "applying CPS to Comyco and other learned hosts" | String replaced | no |
| SCI-23 | §3.1, §3.4.1, Table 2, §4.1 | Ĉ_k means the trace throughput in §3.1 and §3.4.1 ("Let Ĉk > 0 be the trace-derived effective throughput") but the predictor elsewhere | Use $C_k^{\mathrm{act}}$ for realized or trace throughput in §3.1 and §3.4.1 (including the nominal download-time formula and $\tilde C_k$). Keep $\hat C_k$ for the predictor only. Log every occurrence | Each symbol has exactly one meaning, consistent with Table 2 | no |
| SCI-24 | §5.4, Table 5 | The primary PPO results ("35.3% … 79.3% (p=6.2×10⁻¹⁸)") appear in no table | Add a "PPO (content-aware)" block to Table 5. Copy values that already appear in §5.4 verbatim and mark the rest `\authorcheck{}` | No invented cells | yes |
| SCI-25 | §1 | No problem statement; five contribution bullets, two of which are findings | Add a 1–2 sentence problem statement ("Given a base policy π and a ladder …, design a projection S such that …"). Reduce to three contributions and put regime characterization and co-design in one "Findings" sentence. Keep at most three headline numbers, copied from §5. No new numbers | Three bullets and a problem statement | no (authors review) |
| SCI-26 | §2 | Gaps in related work; only one *Computer Networks* citation | Propose additions **in the changelog only**, each with a verification URL (Phase A brief §9 list, plus 2–3 recent *Computer Networks* papers). Insert them only after the authors approve (`\authorcheck{approve}`) | Every added citation verified and approved | yes |
| SCI-27 | §5.4, Table 6, Fig. 8, §6 | Single-seed co-design result | `\authorcheck{}` offering two options: 3–5 seeds (authors run them), or shorten §5.4 by about 30% and keep the "supporting" label | Option chosen and applied | yes |
| FIG-01 | Fig. 1, p.4 | Text is about 4 pt; Type 3 font | Re-export from the source with all text ≥ 7 pt, or use `figure*`, with TrueType fonts. If the source file is missing, mark it `Blocked` | ≥ 7 pt; no Type 3 | no |
| FIG-02 | Fig. 2, p.5 | Small, overlapping labels; "173" | Replot from the existing data with larger fonts and rotated ticks. Fix the count per SCI-05 | Legible; count consistent | yes (count) |
| FIG-03 | Fig. 3, p.8 | Mixed units on one axis; "BB greedy" label | Split into two panels (% and VMAF points). Rename the label to "Broadband greedy". Same data (replot-only) | One unit per axis | no |
| FIG-04 | Fig. 10 caption, p.11 | "Left: banked bitrate rises and the mean VMAF change stays near zero …", but only bitrate is plotted | Remove the VMAF clause, or add VMAF Δ on a secondary axis from `tables/table_ablation_eps.tex` data | Caption describes only what is plotted | no |
| FIG-05 | Table 5 | Structure | Carry out SCI-12 and SCI-24. Use `table*` if the table becomes too wide | Compiles; markers visible | yes |
| LANG-19 | §1, §3, §4 | Past tense used for the method and the paper structure | Present tense for the method, structure, and claims; past tense only for experiments. No change in content | §1, §3, and §4 use the present tense | no |
| LANG-20 | Fig. 8 left-panel title, p.10 | "Certified arm (improved shield)" | Replot the Fig. 8 panel title as "Certified arm (CPS-P)" (replot-only). Check `grep -n -i "improved\|harness\|legacy\|episodes.csv"` in the `.tex` and in the plotting scripts | No jargon in the text or figures | no |

#### LANG tasks (P1)

For every LANG item, acceptance means the "before" string is gone from the `.tex` and the "after" string is present.

| ID | Loc | Before (v2) | After | Author |
|---|---|---|---|---|
| LANG-04 | §1, p.1 | We introduced a Certified Perceptual Shield (CPS) that targets this structure | We introduce a Certified Perceptual Shield (CPS) that exploits this structure | no |
| LANG-05 | §1 and §3.1 | sub-6 GHz and 5G/mmWave | sub-6 GHz and mmWave 5G | no |
| LANG-06 | §3, p.3 | This section formalized … Table 2 collected the symbols … | This section formalizes … Table 2 lists the notation. | no |
| LANG-09 | §4.1, p.5 | we fell back to a fixed scale ρfb =0.80 | we fall back to a fixed scale $\rho_{\mathrm{fb}}=0.80$ | no |
| LANG-11 | §4.3, p.6 | The predictive-banking experiments (CPS-P) ran with ε=1.0, εrisk =4, Brisk =8 s, and H≥6 at forecast quantile 0.2. | The predictive-banking experiments (CPS-P) use $\varepsilon=1.0$, $\varepsilon_{\mathrm{risk}}=4$, $B_{\mathrm{risk}}=8$ s, $H=$\authorcheck{exact value}, and forecast quantile 0.2. (Also fix Table 4.) | yes (H) |
| LANG-13 | §5.4, p.10 | Paired Wilcoxon tests and episode-bootstrap intervals were reported as … | See SCI-08 | yes |

#### REF tasks (P1)

**Rule:** retrieve all bibliographic data from the DOI record, the publisher page, or DBLP, and log the URL. Do not guess.

| ID | Ref | Action | Accept | Author |
|---|---|---|---|---|
| REF-03 | [6] | Fix the web-reference format (organization, title, year, URL, access date). `\authorcheck{keep Cisco or replace with a peer-reviewed measurement source?}` | Well-formed entry | yes |
| REF-04 | [8] | **The URL overflows into [21] (p.14).** Add `\usepackage{xurl}` (load it after `hyperref` or on its own), or wrap it with `\url{}` inside the bib `url` field. Recompile and check the image of the References page | No overflow | no |
| REF-06 | [14] | Add the article number from DOI 10.1145/3742472 | Present | no |
| REF-07 | [15] | Add pages from DOI 10.1145/3651890.3672269 | Present | no |
| REF-08 | [18] | Add pages from DOI 10.1145/3651890.3672260 | Present | no |
| REF-10 | [26] | Replace the Sutton & Barto citation in "Their neural policies combine throughput histories … [26]" with the existing keys for [3] and [27] | Claim cites ABR systems | no |
| REF-11 | [33], [35] | Add NeurIPS volume and pages (from proceedings.neurips.cc or DBLP) | Present | no |
| REF-12 | [38] | Remove "ongoing program; data used for training and evaluation"; fix the broken URL ("ongoreports"); format as a dataset (organization, title, year, URL, access date) | Clean entry | no |
| REF-13 | [39] | NorNet Edge (Comput. Netw. 2014) is cited as the source of the "Norway/Nornet" traces. The Norway traces common in ABR work are the HSDPA commute traces of Riiser et al. (MMSys 2013). `\authorcheck{which Norway dataset was used?}` Remove [39] from "capped catalogue diversity [39]" (p.11). Update the Data availability statement to match | Citation matches the real source; no unrelated use | yes |
| REF-15 | all | LTWA abbreviations (e.g., "IEEE Commun. Surv. Tutor.", "Comput. Netw."); one consistent style for proceedings names; add verified DOIs | Consistent | no |

### 4.3 P2: minor (22)

| ID | Loc | Action | Accept | Author |
|---|---|---|---|---|
| SCI-03 | Table 8 vs §5.6 | **Verify only.** Confirm that "VMAF Δ (pts)" holds +0.02 / −0.01 / −0.17 / −0.48 and "Rebuffer Δ (%)" holds −3.5 / −9.3 / −30.8 / −46.1, consistent with §5.6 "mean VMAF change remained negligible (at most 0.48 points)" and with the §5.2 greedy VMAF of −0.01. Log as `Verified-no-change` | Logged | no |
| FMT-07 | Whole document | Optional migration to `cas-dc` only if the authors ask for it; otherwise log "not requested" | Logged | yes (decision) |
| FMT-08 | Keywords | `\authorcheck{}` proposing to replace "Certified Safety" with "5G" or "Safe reinforcement learning". Keep 1–7 keywords | Authors decide | yes |
| FMT-09 | p.1 emails; PDF metadata | Remove the superscript "a" after names in the email footnote. Add `\hypersetup{pdfauthor={Saeed Zarbi, Leili Farzinvash, Pedram Salehpour}, pdftitle={Certified Perceptual Shielding for Adaptive Bitrate Streaming}}` after `hyperref` is loaded | `pdfinfo` shows the correct names (paste the output) | no |
| FMT-10 | §4.2; Table 2 | "(7)–(6)" → "(6)–(7)"; use `\eqref` consistently | Fixed | no |
| FMT-11 | End matter | Order: Conclusion → CRediT → Competing interest → Funding → Data availability → AI declaration → Appendix A → References → Biographies. The AI declaration must stay **before** the References | Order as specified | no |
| FMT-12 | §4 "Evaluation arms" | Move the paragraph to the start of §5.1 | Moved; references resolve | no |
| FIG-06 | Fig. 4 | `\authorcheck{remove Fig. 4 (identical violins) or replace?}` | Authors decide | yes |
| FIG-07 | Fig. 5 | Replot with alpha transparency and a larger legend (replot-only) | Legible | no |
| FIG-08 | Fig. 6 | Add an inset where the Safety and Certified CDFs differ (replot-only) | Difference visible | no |
| FIG-09 | Fig. 8 right panel | `\authorcheck{remove the right panel or add bootstrap bands?}` | Authors decide | yes |
| FIG-10 | Fig. 9/Table 7; Fig. 10/Tables 8–9 | `\authorcheck{keep figure or table?}`. Meanwhile, state in each caption that the figure visualizes the table | Logged | yes |
| FIG-11 | Table 1 | Sort rows by year; define "≈" in the note; stop the "Paradigm" column from wrapping. Make the BOLA year consistent with REF-14 | Clean layout | no |
| FIG-12 | All plots | Regenerate every figure from existing data with Type 42 (TrueType) fonts and a serif/Times family (snippet in Section 5.8) | `pdffonts main.pdf` shows **no Type 3** fonts | no |
| LANG-07 | Table 2 | "((2))" → "(Eq. (2))" | Fixed | no |
| LANG-08 | Table 3 note | "Top gain is VMAF(6000)−VMAF(2850) in kbps" → "Top gain is VMAF at 6000 kb/s minus VMAF at 2850 kb/s" | Fixed | no |
| LANG-10 | §4, §5.3 paragraph heads | Remove the double periods ("Evaluation arms..", "Coverage under injected drift..") | Fixed | no |
| REF-01 | [3] | Write `{P}ensieve` in the BibTeX title | Fixed | no |
| REF-02 | [4] | Keep a single arXiv identifier | Fixed | no |
| REF-05 | [13] | Check the year against the DOI record (vol. 21(1) is usually 2019) | Year matches the DOI | no |
| REF-09 | [22], [25] | Look for peer-reviewed versions; otherwise label them clearly as preprints | Fixed | no |
| REF-14 | [17] / Table 1 | Use the year of the cited version (2020) in Table 1 | Consistent | no |

---
## 5. Ready-to-paste fixes

### 5.1 Abstract, last sentence (FMT-01)

```latex
% before
These results show that the benefit of banking depends on ladder saturation and network regime.
% after
Thus, banking benefit depends on ladder saturation and network regime.
```

The expected count after the change is 246 words. Recount in the file anyway.

### 5.2 Highlights (FMT-02)

Replace line 1 only. Character counts exclude the leading "- ".

```text
- CPS is a model-agnostic runtime shield wrapping existing client-side ABR controllers
```

This line is 84 characters. If the authors prefer a shorter one, use "CPS is a model-agnostic runtime shield for existing client-side ABR controllers" (79 characters). **Do not** use "…that wraps existing client-side ABR controllers": it is 86 characters, over the limit. Bullets 2–5 stay as they are (81, 82, 82, and 82 characters).

### 5.3 Generative-AI declaration (FMT-05)

```latex
\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
During the preparation of this work the authors used Claude [version: \authorcheck{}]
for language editing and for assistance in regenerating tables and figures from the
authors' existing experimental outputs. After using this tool, the authors reviewed
and edited the content as needed and take full responsibility for the content of the
published article.
```

Keep it immediately before the References. If the authors used other tools, or used AI in other ways (for example editing scientific text or code), add them after asking; do not guess. Ask in the marker, for example: `\authorcheck{list any other AI tools/uses}`.

### 5.4 Data availability (FMT-04): choose one option with the authors

**Option A. Deposit (preferred for Option C):**

```latex
\section*{Data availability}
The CPS implementation, chunk-level emulator, synthetic 5G trace generator and
configuration files, evaluation scripts, per-title session-mean VMAF ladders, and
per-episode result files underlying Tables~5--9 and Figs.~2--10 are openly available
at \authorcheck{repository} under the \authorcheck{licence} licence
(DOI: \authorcheck{DOI}). % \cite{ARTIFACT_KEY} once the bib entry exists
The FCC Measuring Broadband America traces~\cite{<key of [38]>} and the
\authorcheck{Norway dataset name, per REF-13} traces are third-party public datasets
available from their original providers.
```

**Option B. Not openly shared at submission (a stated reason is required):**

```latex
\section*{Data availability}
The CPS implementation, chunk-level emulator, synthetic 5G trace generator,
evaluation scripts, and per-episode result files are not publicly available at
submission because \authorcheck{reason}. They are available from the corresponding
author upon reasonable request. The FCC Measuring Broadband America
traces~\cite{<key of [38]>} and the \authorcheck{Norway dataset, per REF-13} traces are
third-party public datasets available from their original providers.
```

Add "A persistent archive with a DOI will be deposited upon acceptance" **only** if the authors confirm it in writing. Remove the circular sentence "Synthetic 5G trace generators … are included with the code release described above".

### 5.5 Author biographies (FMT-06)

```latex
\section*{Author biographies}
\newcommand{\authbio}[3]{%
  \noindent\begin{minipage}[t]{0.24\linewidth}
    \IfFileExists{#1}{\includegraphics[width=\linewidth]{#1}}{\fbox{\parbox{0.9\linewidth}{\centering photo\\missing}}}
  \end{minipage}\hfill
  \begin{minipage}[t]{0.72\linewidth}\textbf{#2} #3\end{minipage}\par\medskip}
\authbio{photo_zarbi.jpg}{Saeed Zarbi}{\authorcheck{biography, max 100 words}}
\authbio{photo_farzinvash.jpg}{Leili Farzinvash}{\authorcheck{biography, max 100 words}}
\authbio{photo_salehpour.jpg}{Pedram Salehpour}{\authorcheck{biography, max 100 words}}
```

Do not write biography text yourself, not even from public web pages.

### 5.6 Fig. 7 caption (SCI-04)

```latex
\caption{Episode-averaged rolling conformal coverage (rolling window
$W_{\mathrm{roll}}=20$ chunks; greedy certified arm, 204 episodes, synthetic 5G pool)
under an injected throughput shock (red band; $0.3\times$ scale, chunks 28--37).
Grey band: conformal warm-up (chunks 0--19). Dashed line: target $1-\alpha=0.90$.
Because the rolling window still contains warm-up chunks, the curve stays below the
target through the warm-up and most of the shock band, first reaching 0.90 at chunk
\authorcheck{N, read from the plotted data file}. Phase-mean coverage over valid
post-warm-up chunks is 0.928 / 0.929 / 0.944 (pre / shock / post).}
```

Also change "W=20" to "$W_{\mathrm{roll}}=20$" in §5.3 and in the legend of the plot ("Rolling mean ($W_{\mathrm{roll}}$ = 20)").

### 5.7 PDF metadata and the email footnote (FMT-09)

```latex
\hypersetup{pdfauthor={Saeed Zarbi, Leili Farzinvash, Pedram Salehpour},
            pdftitle={Certified Perceptual Shielding for Adaptive Bitrate Streaming}}
```

The footnote currently shows "szarbi@tabrizu.ac.ir (Saeed Zarbi^a)". Find where the email list picks up the affiliation label (usually `\ead{}` combined with the `\author[a]` label, or a manual footnote) and remove the "a". Check with `pdfinfo main.pdf`.

### 5.8 Figure fonts (FIG-12): replot-only

```python
import matplotlib as mpl
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, not Type 3
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})
# set figsize to the final column width (about 3.5 in single column, 7.2 in double)
```

Run each plotting script in replot-only mode on the existing result files. Log the script, the inputs, and the command. Confirm with `pdffonts main.pdf | grep -c "Type 3"`, which must print 0. Plotted values must not change.

### 5.9 Reference [8] overflow (REF-04)

```latex
\usepackage[hyphens]{url}  % or: \usepackage{xurl}
```

Recompile, then render the References page (for example `pdftoppm -f 14 -l 14 -png main.pdf ref`) and look at it to confirm that [8] no longer overlaps [21].

---

## 6. Items the agent must NOT do alone (author-only)

For each item below, insert `\authorcheck{}` and prepare scaffolding only.

1. Any new experiment, re-run, re-training, re-simulation, or new statistical computation:
   - new baselines (SCI-10);
   - buffer settings (SCI-16);
   - cluster CIs and Holm correction (SCI-19);
   - extra seeds (SCI-27);
   - QoE values not already present in the result files (SCI-12);
   - PPO table cells not already stated in the text (SCI-24).
2. Changing Proposition 1 or 2, their proofs, or their scope (SCI-09, and the SCI-01 scoped statement).
3. Choosing between contradictory numbers without a source file:
   - 173 vs 113 (SCI-05);
   - the shared seeds for the sweep rows (SCI-02);
   - identical p-values (SCI-08).
4. Configuration values that do not appear in a config file:
   - H (LANG-11 / SCI-18);
   - f, the encoder settings, the VMAF model version, the PPO hyperparameters, software versions, and hardware (SCI-18);
   - the window persistence (SCI-14);
   - the episode construction (SCI-17);
   - the broadband sampling (SCI-20);
   - the 5G generator (SCI-13).
5. Repository, DOI, licence, reasons data cannot be shared, and deposit plans (FMT-04).
6. The AI tool version and the full scope of AI use (FMT-05).
7. Biographies and photos (FMT-06).
8. Which Norway dataset was used (REF-13), and whether to replace the Cisco source (REF-03).
9. Adding related-work citations (SCI-26) and citing "prior shields" (SCI-11).
10. Editorial choices:
    - removing figures or panels (FIG-06, FIG-09, FIG-10);
    - the keyword change (FMT-08);
    - the CAS migration (FMT-07);
    - multi-seed vs shortening (SCI-27).

---

## 7. Final acceptance checklist

Paste the command output for each item into `CHANGELOG_revision_phaseB.md`.

- [ ] **Abstract ≤ 250 words**, counted in the file (`detex` on the abstract piped to `wc -w`), with no citations and no undefined abbreviations. Expected: 246.
- [ ] **Highlights**: 3–5 bullets, each ≤ 85 characters (`sed 's/^- //' Highlights.txt | awk '{print length}'`), in a separate file whose name contains "highlights". Numbers match Table 5.
- [ ] **Keywords**: 1–7, and none contain "and" or "of" (currently 6).
- [ ] **AI declaration**: exact title, immediately before the References, scope matching Section 5.3, and the version filled in or marked.
- [ ] **CRediT**, **Declaration of competing interest**, and **Funding** are present and unchanged unless the authors instructed otherwise.
- [ ] **Data availability**: Option A or B filled in, or marked; no circular wording; no invented URL or DOI.
- [ ] **Biographies**: three, each ≤ 100 words, each with a photo, or marked.
- [ ] **References**: all REF tasks closed. `grep -c "??"` on `pdftotext main.pdf` gives 0. No undefined citations in the `.log`. The [8] URL does not overflow. Citations are in order of first appearance.
- [ ] **Fonts**: `pdffonts main.pdf` lists no Type 3 fonts, and all fonts are embedded (`emb` = yes).
- [ ] **Metadata**: `pdfinfo main.pdf` shows "Saeed Zarbi, Leili Farzinvash, Pedram Salehpour" and the correct title.
- [ ] **Consistency**: Table 8/9 default rows = Table 5; ranges 3.8–4.5% and 8.9–9.3% everywhere; the Fig. 7 caption matches the curve; no symbol is used for two quantities (W vs $W_{\mathrm{roll}}$, $\hat C$ vs $C^{\mathrm{act}}$).
- [ ] **Markers**: every `\authorcheck{}` in the `.tex` appears in the changelog list, and the counts match (`grep -o "\\\\authorcheck{" main.tex | wc -l`). Remove the markers and the macro only after the authors have answered all of them, and never do this on your own initiative.
- [ ] **Changelog**:
  - every task has exactly one status (`Done` / `Deferred-authorcheck` / `Verified-no-change` / `Blocked`);
  - every numeric change has a source of truth;
  - every replot records its script, inputs, and command;
  - every new citation has a verification URL;
  - the status count table covers all 86 IDs plus NEW-01.
- [ ] **Build**: `latexmk -pdf` from a clean copy finishes with no errors, no undefined references or citations, and no new overfull boxes wider than 5 pt. `main.ltx` is in sync with `main.tex`.
- [ ] **Jargon**: `grep -n -i "improved shield\|harness\|legacy\|episodes.csv"` finds nothing in the `.tex` or in the figure text.
