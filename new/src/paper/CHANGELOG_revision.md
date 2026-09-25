# CHANGELOG_revision.md (Phase A)

Source brief: `paper_revision_brief_computer_networks_EN.md`
Files: `overleaf_upload/main.tex`, `main.ltx` (synced), `tables/table_ablation_eps.tex`, `Highlights.txt`

| Task | Change | Note |
|------|--------|------|
| FMT-01 | Abstract replaced with ≤250-word draft; macros for episodes/coverage; predictive sentence notes \(\varepsilon_{\mathrm{risk}}=4\) | Numbers from §5.2/Table 5 |
| FMT-02 | Added `overleaf_upload/Highlights.txt` (5 bullets ≤85 chars) | Also copied to `new/src/paper/Highlights.txt` |
| SCI-06 | Ranges → 3.8–4.5% bitrate, 8.9–9.3% rebuffering (abstract + conclusion) | Intact host list in §5.2 |
| SCI-07 / LANG-12 | Intervention rates: greedy/BBA/BOLA/RobustMPC/Pensieve via macros | Removed "near zero" |
| SCI-03 | Headers already correct vs §5.6; fixed `-0.00` → `0.00` | No column swap needed |
| SCI-05 | Per-chunk claims cite Fig. ladder (not Table spacing); four-title scope sentence in §5.5 | Count 113 vs 173 still author (deferred) |
| SCI-15 | Contribution #3, RQ3, CPS-P §4.3/§5.3/conclusion: wider risk budget | No number edits |
| SCI-21 | Softened CDF caption, "catastrophic", broadband regime, conclusion co-design | As brief table |
| SCI-01 | §3.3 / §5.5 / §6 + Prop intro: align with Table 3 (4/12 mono; mid-ladder inversions; upper 3 rungs always nondecreasing); Prop 2 scoped | Data-backed; no table number changes |
| SCI-02 Path A | Default ablation cell (ε=1, α=0.10) reused from `greedy_5g`; tables/figure regenerated; captions note headline row | 4.5% / −9.3% / 70.4% now match Table 5 |
| SCI-04 Path C | Drift prose: phase means vs rolling \(W{=}20\); warm-up shaded on fig; caption defines both metrics | No re-simulation; `--replot-only` |
| FMT-04 Path B | Data availability: upon reasonable request + DOI planned upon acceptance; FCC/Norway cited | Temporary until Zenodo DOI |
| FMT-05 | Generative-AI declaration: Claude for language/clarity/wording; before References | Author-confirmed |

**Deferred (needs author / Phase B+):** SCI-01, SCI-02, SCI-04, SCI-08–14, FMT-04–06, full LANG-19, REF fixes.
