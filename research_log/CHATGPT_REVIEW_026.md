# CHATGPT REVIEW 026 — frozen evidence package is accepted; move from research iteration to manuscript assembly

Reviewed commit: `3ae916d308ba74ac958acaec7d9ca3390757ed9e`.

## Decision

Cycle 026 is accepted. It does not add a new scientific result, and that is exactly the correct outcome after Cycle025. It converts the strongest frozen Layer-1 evidence into a publication-facing package without reopening model selection, rescoring, subset search, or Layer-2 development.

The cycle advances the memory-centric vision in three useful ways:

1. it makes Cycle025 the single primary quantitative evidence block and keeps Cycles022/018 visibly separated as overlap-affected replications rather than pooling them;
2. it freezes qualitative examples by a deterministic categorical rule, which substantially reduces figure cherry-picking risk;
3. it turns the claim boundaries into an explicit ledger, so the manuscript can emphasize identity-memory use while preventing accidental escalation into training-unseen, architecture-only, autonomous-memory-lifecycle, lower-degradation-sensitivity, or successful-agent claims.

The scientific core therefore remains unchanged and should now be treated as frozen:

> **same observation + same relational query + different supplied miner identity memory -> different intended helmet identity, with CMA retaining much stronger absolute identity-grounded performance than the pinned SegLLM historical-memory comparator under the fixed compound degradation.**

Cycle026 correctly does **not** revive the failed Layer-2 controller/verifier line. The project is now best framed as **memory-centric relational perception**. If the manuscript still uses “agentic” language, that term must describe the broader motivation/system framing rather than imply a validated autonomous memory-write/verify/repair controller.

## Quantitative package assessment

`PAPER_LAYER1_TABLE.md/.csv` is suitable as an authoritative evidence source. The hierarchy is correct:

- Cycle025: primary `DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION`;
- Cycle022: historical replication with 37/50 documented reconstructed-training overlap;
- Cycle018: development replication with 50/50 documented reconstructed-training overlap.

The main Cycle025 result remains:

- CMA target15_b: `67.53%` mIoU, `29/50` CMSA, `84%` Fidelity, `3%` IER;
- SegLLM target15_b: `36.44%` mIoU, `1/50` CMSA, `46%` Fidelity, `38%` IER;
- paired gaps: `+31.08 pp` mIoU, `+56 pp` CMSA, `+38 pp` Fidelity, `-35 pp` IER.

The table verifier checking all numeric cells against frozen source JSON is good research engineering. Do not recompute these numbers again for manuscript formatting; formatting scripts should consume the verified table artifacts.

One manuscript-level refinement is important: **do not put all three Cycle025/022/018 blocks into the main-paper performance table.** The overlap-affected replications are useful provenance/context evidence but visually mixing them with the primary block can weaken the central claim and invite readers to misread them as three independent test sets. The main table should use Cycle025 only; Cycles022/018 should move to a supplementary replication/provenance table or short supporting paragraph.

## Qualitative package assessment

The deterministic case rule is strong and should be preserved. It includes success, shared failure, degradation-induced failure, and an identity-error contrast rather than selecting only visually flattering cases. The rendering script is also appropriately non-scientific: it verifies stored hashes, reads frozen masks, and applies display-only overlays without changing mask geometry, image brightness, predictions, or scores.

For the final paper figure, however, the current five full 4x5 panels are better treated as a **supplementary qualitative bank** than as one main-paper figure. A two-column conference page will not make twenty cells per case legible.

The main manuscript should use only already frozen cases and create a compact visual narrative:

- one or two deterministic identity-switch success cases showing Memory A vs Memory B on the same observation;
- one degradation failure case to show the remaining limitation;
- optionally one SegLLM identity-error contrast if space permits.

No new case search is authorized. The selected group IDs from Cycle026 must remain the source pool for any condensed layout.

## Results wording assessment

`PAPER_RESULTS_DRAFT.md` is disciplined and largely ready to reuse. Preserve these distinctions:

- say **higher absolute degraded performance / retained identity fidelity**, not lower degradation sensitivity;
- call SegLLM a **system-level historical-memory comparator**, not architecture-controlled proof;
- describe Cycle025 as `DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION`, not training-unseen;
- state that memory is externally supplied and targets are reconstructed pseudo labels;
- retain the `23` historical-review + `27` accepted-pair-combination QC provenance;
- state explicitly that Layer-2 controller/verifier attempts are negative evidence.

The strongest paper-level interpretation is not “CMA solves degradation.” CMA still fails strict degraded CMSA on `21/50` groups. The stronger and more defensible message is that **identity memory remains useful and substantially more reliable than the direct historical-memory comparator even when small relational evidence is degraded, while localization remains the residual failure mode.**

## Research stop remains binding

No further in-repository model experiment, split search, threshold adjustment, baseline retry, significance fishing, or Layer-2 resurrection is justified by Cycle026. New scientific experiments should resume only if genuinely independent data or a materially new research question is supplied.

---

# CYCLE 027 — one focused hour

## Goal

Assemble a **manuscript-ready Layer-1 results package** from the already frozen Cycle025/Cycle026 evidence. This is formatting and narrative integration only; it must not create new empirical evidence.

## A. Produce the main-paper LaTeX table

Create `research_log/cycle027/MAIN_TABLE_LATEX.tex` using **Cycle025 only**.

Requirements:

- rows: CMA base-w15 and pinned SegLLM;
- conditions: clean and `target15_b`;
- columns: mIoU, CMSA, Memory Fidelity, IER, and mean identity margin;
- include numerator/denominator for CMSA/Fidelity/IER where space permits;
- caption must state `50` groups / `100` identity references per condition, supplied identity memories, reconstructed pseudo targets, and `DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION` provenance;
- caption must explicitly say comparison is system-level and exact run-bound/ancestor exposure is unknown;
- numbers must be read from the verified Cycle026 CSV/verification receipt, not manually retyped from prose.

Put Cycles022/018 into `SUPPLEMENTARY_REPLICATION_TABLE_LATEX.tex`, clearly labelled as overlap-affected replications. Do not pool them.

## B. Build one compact main qualitative figure from the already frozen cases

Create a compact main-paper figure layout using only Cycle026-selected cases.

Use exactly:

1. `cf_778571dc8020b5df` as the primary identity-switch success example;
2. `cf_f6120ac9bd438a95` as a second success only if the layout remains readable;
3. `acceptedpair_b914b3d4a14a9083` as the degradation limitation example;
4. `cf_56a9b1ea716e19d3` only if space permits for the SegLLM identity-error contrast.

Do not select any new case.

Prefer a compact structure that makes the scientific intervention visible within five seconds: same observation, Memory A vs Memory B, CMA output, SegLLM output, with small target/IoU annotations. Keep the existing full five-case panels unchanged as supplementary figures. Rendering may re-layout frozen assets but may not alter masks, predictions, scores, image brightness, or model outputs.

Create:

- `research_log/cycle027/MAIN_QUAL_FIGURE.pdf` and `.png`;
- `MAIN_QUAL_FIGURE_CAPTION.md`;
- a rendering receipt listing all source hashes and proving zero model/scorer calls.

## C. Convert Results + Limitations to manuscript LaTeX

Create `research_log/cycle027/RESULTS_LIMITATIONS_LATEX.tex` by converting the frozen Cycle026 prose into concise conference-paper prose.

Target structure:

1. **Counterfactual identity-memory evaluation** — define the same-image/same-query memory intervention and report the primary Cycle025 table;
2. **Robustness under compound degradation** — emphasize absolute degraded performance and the residual 21/50 CMSA failures;
3. **Comparison scope** — SegLLM as a pinned system-level historical-memory comparator;
4. **Limitations** — exact exposure unknown, supplied memory, pseudo labels/QC, larger CMA degradation drop, unsuccessful Layer-2 control.

Do not introduce a successful Agent contribution, autonomous memory writing claim, safety claim, or new headline metric.

## D. Freeze paper contribution wording

Create `research_log/cycle027/CONTRIBUTION_WORDING.md` with exactly three manuscript-level contribution bullets:

1. problem/formulation: counterfactual identity-memory relational perception for visually similar workers and small PPE targets;
2. method/evaluation contribution: supplied entity memory is causally switched under the same observation/query and measured with CMSA/Fidelity/IER plus mIoU;
3. empirical contribution: strong frozen system-level comparison under coal-mine-style compound degradation, with Cycle025 as primary evidence.

The third bullet must not claim lower degradation sensitivity or training-unseen generalization. Do not manufacture an Agent/controller contribution as bullet 2 or 3.

## E. Hard non-goals

- no model inference or scorer invocation;
- no metric recomputation except formatting/verification from frozen tables;
- no new samples or qualitative case search;
- no new baseline;
- no new statistical test;
- no method/code changes outside manuscript/figure rendering;
- no Layer-2 work;
- no title/abstract claim that exceeds the claim ledger.

## Deliverable

Before `CODEX UPDATE 027`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 027` with:

1. main and supplementary LaTeX table paths;
2. compact main qualitative figure + hash/render receipt;
3. Results/Limitations LaTeX;
4. three frozen contribution bullets;
5. confirmation that no model/scorer job or new scientific selection ran;
6. exactly one next recommendation limited to full Introduction/Method manuscript assembly.

After Cycle027, the next work should be paper writing, not another empirical cycle.