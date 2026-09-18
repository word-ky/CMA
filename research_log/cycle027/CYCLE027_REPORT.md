# Cycle027 — manuscript results assembly

Review026 (4b49e34) implemented. Formatting and narrative integration only; no model/scorer invocation, metric recomputation, new sample/case search, baseline, statistical test, training or Layer-2 work.

## Tables and prose

`research_log/cycle027/MAIN_TABLE_LATEX.tex` contains only the four Cycle025 method/condition rows. `SUPPLEMENTARY_REPLICATION_TABLE_LATEX.tex` keeps the eight Cycle022/018 rows in separate overlap-affected blocks (37/50 and 50/50). Counts, rates and mean identity margins are formatted directly from the verified Cycle026 CSV by `research_log/build_cycle027_tables.py`. No estimates are pooled. `table_format_verification.json` records source and output hashes.

`RESULTS_LIMITATIONS_LATEX.tex` contains the requested four paragraphs. It retains the primary documented-protocol-disjoint status, exact historical exposure UNKNOWN, supplied memories, reconstructed pseudo targets, 23 review-level plus 27 accepted-pair QC provenance, residual 21/50 degraded CMSA failures, larger CMA degradation decline and unsuccessful Layer-2 results. `CONTRIBUTION_WORDING.md` contains exactly three frozen bullets: problem formulation, controlled supplied-memory evaluation, and empirical system-level comparison. No successful agent or architecture-only attribution is introduced.

## Main qualitative figure

`MAIN_QUAL_FIGURE.pdf/.png` uses only the prescribed frozen success `cf_778571dc8020b5df` (degraded memory A versus B) and limitation `acceptedpair_b914b3d4a14a9083` (first reference A, clean versus degraded). The optional second success and identity-error contrast were omitted to preserve legibility. Limitation memory B and all other conditions remain in the unchanged Cycle026 supplementary bank. This is a four-row, three-column layout with supplied memory, CMA, and SegLLM; saved target contours and frozen IoU annotations identify localization differences.

`MAIN_QUAL_FIGURE_CAPTION.md` and `MAIN_QUAL_FIGURE_LATEX.tex` preserve this scope. `main_figure_render_receipt.json` records input hashes, original QC, layout, unchanged inputs and zero model/scorer calls. All ten Cycle026 panel hashes remain unchanged. Only display copies receive overlays; brightness, mask geometry and saved predictions are unmodified.

## Build and checks

`ASSEMBLY_PREVIEW.tex/.pdf` provides a four-page generic two-column fragment build, not a complete manuscript or final venue pagination. Two final pdflatex passes succeed with resolved references and no overfull boxes. One underfull hbox remains in the comparison paragraph; it does not clip content. The initially missing grfext dependency was installed; figure heading spacing was repaired without changing cases or values. An empty supplementary heading page was removed. Tables, prose and figure were visually checked. `delivery_verification.json` records the final hashes and build result. No scientific code or evaluation protocol changed.

Exactly one next recommendation: assemble the full Introduction and Method manuscript sections from the existing implementation and frozen claim ledger, without reopening experiments or adding autonomous-memory/Agent claims.
