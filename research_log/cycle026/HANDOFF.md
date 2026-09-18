# Cycle026 — frozen paper evidence package

Review025/303c0f6 implemented. This cycle is paper production only. No CMA, SegLLM or SAM inference, scorer invocation, new split, threshold/prompt change, training, significance test or Layer-2 work ran. CPU scripts read frozen metrics and saved assets only.

## Authoritative table and claim ledger

`cycle026/PAPER_LAYER1_TABLE.md` and `.csv` contain twelve frozen method/condition rows in three separate blocks: Cycle025 primary DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION, Cycle022 historical replication with its documented overlap warning, and Cycle018 development replication with its overlap/development warning. There is no pooled estimate. Two primary-only paired CMA-minus-SegLLM delta rows appear in both formats; CSV uses fraction units and Markdown uses percentage points. The raw rows include CMSA/Fidelity/IER numerators, denominators and rates, both identity margins, and group/identity counts.

`table_verification.json` verifies all 184 numeric CSV cells and the corresponding displayed Markdown values against frozen scoring JSON, retaining source paths, JSON pointers and hashes. Counts/denominators in the paired delta rows identify the same evaluated groups, not a pooled dataset. No scorer was called to generate or verify this table.

`cycle026/CLAIM_EVIDENCE_LEDGER.md` makes Cycle025 the primary evidence and retains the exact-exposure, externally supplied memory, reconstructed pseudo-label, group-QC, larger degradation decline, system-level comparison and Layer-2 limitations. `PAPER_RESULTS_DRAFT.md` supplies the requested four paragraphs: main counterfactual-memory result, absolute degraded performance, comparison scope, and limitations. Numerical sentences link to frozen artifacts. No successful agent lifecycle/control contribution is claimed.

## Deterministic qualitative cases

`QUALITATIVE_CASES.json` follows the four requested categories in order. Within each category, it takes the first eligible previously unselected group(s) in frozen manifest order; it does not rank by appearance or gap magnitude. All categories have examples:

| Category | Frozen selected group(s) |
|---|---|
| Degraded CMA passes / SegLLM fails, first two | cf_778571dc8020b5df; cf_f6120ac9bd438a95 |
| Both fail degraded CMSA | acceptedpair_18803244d947e2f7 |
| CMA clean success becomes degraded failure | acceptedpair_b914b3d4a14a9083 |
| SegLLM degraded identity error, corresponding CMA not an identity error | cf_56a9b1ea716e19d3 |

The selection uses stored categorical outcomes only for illustration and changes no quantitative result or dataset membership. Full per-group IoU matrices and original QC records are retained in the case manifest. The last category must not be mistaken for CMA CMSA success; its caption explicitly preserves this distinction.

`research_log/render_cycle026.py` reads the saved Cycle025 observations, reference masks, pseudo-targets and raw prediction masks. It renders five full-frame panels, each with clean-A, clean-B, degraded-A and degraded-B rows and observation/memory/target/CMA/SegLLM columns. Identity-colour overlays affect the displayed copy only; no mask geometry or input brightness is changed. Captions and saved IoU annotations remain tied to the frozen records. Source/QC labels are visible above every panel and preserved in render metadata. Five PNGs, five PDFs and a contact sheet are under `cycle026/figures`; `FIGURE_CAPTIONS.md` supplies shared and per-case wording.

## Verification and artifacts

Rendering verified 70 input asset hashes. All ten downloaded PNG/PDF hashes match the remote rendering receipt. The contact sheet was visually inspected across all five panels: row/column alignment, case IDs, source labels, mask colours and failure cases are visible without clipping; no appearance-based case replacement occurred. Full-resolution individual panels remain available for manuscript assembly.

Supporting records: `table_sources.json`, `table_verification.json`, `QUALITATIVE_CASES.json`, `render_receipt.json`, and `delivery_verification.json`. Original observations/masks/predictions remain in their Cycle025 locations; this cycle adds only display artifacts and documents. Rendering used the existing CPU environment without loading model libraries or invoking the scorer. No GPU experiment was started.

Exactly one next recommendation: assemble the manuscript results section and figures from this frozen package, keeping Cycle025 primary and all exposure, pseudo-label, QC and agent-limit claims explicit. Further research experiments remain stopped unless the user explicitly reopens scientific scope or supplies genuinely independent data.
