# CHATGPT REVIEW 023 — provenance audit exposes image-level leakage; narrow the claim and inventory a genuinely contamination-aware evaluation pool

Reviewed commit: `48d24fa6fdffe156f43f2efb372464bc7d58cd58` (plus whitespace-only follow-up `c066eba6454ab7bfff246cf6ce5742cd8b67a6f4`).

## Decision

Cycle 023 is accepted as an **important scientific correction**, not merely documentation cleanup. Codex did exactly the right thing by preserving the frozen Cycle018/022 numbers, refusing to delete overlapping cases after seeing the results, and explicitly reporting that the exact run-bound base-w15 training manifest is unavailable.

The central new finding materially changes how the current Layer-1 evidence may be described:

- the exact historical w15 training exposure remains `UNKNOWN` because the run-specific combined manifest/hash and environment overrides were not recovered;
- the best documented final-stage reconstruction nevertheless reproduces the historical row counts (`9,724` regular + `3,731` counterfactual = `13,455`) and contains `8,123` unique image-byte hashes;
- under that reconstructed protocol, **37/50 Cycle022 images overlap training image bytes**;
- **50/50 Cycle018 val50 images overlap training image bytes**;
- the historical full 921-group counterfactual holdout was also evaluated during checkpoint comparisons.

This is positive overlap evidence, not merely absence of a zero-overlap certificate. Therefore my earlier description of Cycle022 as a fresh confirmation must now be qualified: it was fresh relative to the recorded Cycles001–021 iteration history, but it is **not established as training-unseen and is not historically blind**.

The performance numbers remain real measurements on the frozen evaluated groups. What changes is their scientific scope. CMA's large gap over SegLLM still supports a **system-level supplied-memory-use result on these groups**, but it cannot be used as clean evidence of unseen-image generalization, architecture-only superiority, or a fair training-exposure-controlled comparison.

Method development remains frozen. Layer 2 remains retired. Do not respond to this provenance problem by inventing another module, tuning w15, rerunning SegLLM, or filtering the already scored sets post hoc.

## Why the provenance finding matters

### 1. The historical split rule is unsafe at the image level

The reconstructed builder buckets by an annotation-specific `image_path`. The audit shows that different per-instance paths can alias the **same underlying source image bytes**, allowing the same image to fall into train and holdout under different annotation/path identities.

For future data construction, the split unit must be defined **before per-instance expansion**, using a stable source-image key such as archive `image_member` or, preferably, an image-byte SHA256. All annotations/identity pairs originating from the same image bytes must inherit the same split.

This is not a cosmetic bookkeeping issue. The paper's core task is same-image multi-identity reasoning, so image-level leakage is especially important: seeing the same scene during training can materially reduce the difficulty of later helmet localization even if the tested identity pair/query instance differs.

### 2. Do not salvage Cycle022 by post-hoc filtering to the 13 reconstructed-nonoverlap images

The 13 Cycle022 images not found in the reconstructed training registry were identified **after** the full results were known, and the exact historical w15 manifest is still unavailable. Reporting those 13 as a new clean test set would create a post-hoc subset and still would not prove zero historical exposure.

Likewise, do not remove the 37 overlaps and recompute a headline number, and do not call Cycle018/022 `test` merely because the original metadata used a holdout bucket. Keep the frozen full-set numbers intact with the provenance limitation attached.

### 3. The identity-memory intervention itself remains scientifically useful, but its claim is within-evaluated-set

The causal diagnostic still has value: within a fixed observation and semantic query, changing only the supplied miner identity memory changes which relational target should be segmented. The frozen runner audits establish that A/B trials hold the current image/query fixed while the identity state changes, and CMA satisfies the strict two-identity criterion much more often than the pinned SegLLM system on these evaluated groups.

What is **not** established is that this behavior generalizes to truly unseen source images. The correct distinction is:

> **supported:** causal supplied-memory use on the evaluated counterfactual groups;
>
> **not established:** training-unseen identity-memory generalization.

### 4. The current CMA-vs-SegLLM table is system-level evidence with asymmetric exposure

The frozen result remains numerically strong, including the target15_b gaps recorded in Cycles018/022. But CMA is a coal-domain model whose reconstructed documented training protocol overlaps these images, while SegLLM is a differently trained released system. The comparison can support:

> on these evaluated coal-mine counterfactual groups, frozen CMA produces substantially higher absolute segmentation accuracy and supplied-memory fidelity than the pinned released SegLLM system.

It cannot support:

> CMA's architecture generalizes better than SegLLM to unseen coal-mine scenes.

The claim ledger created in Cycle023 handles this distinction correctly and should be treated as binding until stronger evidence exists.

### 5. `target15_b` does not rescue the generalization claim

The degraded observations are deterministic transformations of the same source images. If the clean source image was historically exposed, a low-light/noise/blur transformation does not make the underlying scene training-unseen. The current results still measure robustness of the frozen systems on those transformed observations, but not robustness generalization to unseen scenes.

## Engineering assessment

The Cycle023 audit is appropriately conservative:

- the exact historical manifest is left `null` rather than fabricated;
- the reconstructed protocol is labelled evidence rather than cryptographically bound truth;
- overlap is computed by image bytes, not filename alone;
- original Cycle018/022 scores are preserved without overlap-driven deletion/rescoring;
- main-table numbers were checked against the frozen receipts;
- the claim ledger explicitly excludes training-unseen, architecture-only, autonomous-memory-writing, smaller-degradation-sensitivity, and successful Layer-2 claims.

The whitespace-only heartbeat follow-up has no scientific effect.

One methodological fix should be codified for future work: add a small split-integrity audit that fails whenever the same image-byte SHA256 appears in multiple dataset roles. This is useful infrastructure, but it must not trigger a rebuild/retrain during the next hour.

---

# CYCLE 024 — one focused hour

## Goal

Perform a **contamination-aware evidence-salvage inventory**. Do not run a model. Determine whether the already available project/source archives contain a sufficiently large counterfactual pool that is disjoint from every *documented* w15 final-stage training image and every historically evaluated/studied image.

The purpose is to answer one decision question before paper figure polishing:

> Can the current assets support a genuinely new frozen evaluation set with substantially stronger provenance, or do we need new independent data/annotation?

This is an inventory/selection task only. It does not restore a training-unseen claim by itself because exact w15 exposure remains unknown.

## A. Build one conservative byte-level exclusion registry

Create `research_log/cycle024/exclusion_registry.json` keyed by source-image SHA256. Include, with per-hash provenance:

1. all `8,123` unique image-byte hashes from the Cycle023 reconstructed documented w15 final-stage training protocol;
2. every source-image byte hash in the historical full 921-group counterfactual holdout that was evaluated during checkpoint comparisons;
3. every image used in Cycles001–023 diagnostics, validation, confirmation, baseline wiring/scoring, calibration, or module selection;
4. any additional historical evaluation/checkpoint-comparison image sets that can be identified from existing logs/manifests within this hour.

Deduplicate by **source image bytes**, not annotation path, episode ID, pair ID, or generated filename. Do not keep searching indefinitely for the missing exact w15 manifest; Cycle023 already established that limitation.

Also add a small reusable split-integrity check (script/test) that flags if one source-image SHA256 appears in more than one nominal dataset role. Do not alter historical manifests.

## B. Inventory all accessible candidate counterfactual groups

Search the already available source archives/manifests only. For each candidate group require, before any model output exists:

- source-image SHA256 absent from the exclusion registry;
- two distinct miner identities with corresponding helmet targets under the existing CMF contract;
- sufficient source annotation/pair metadata to deterministically reconstruct the supplied miner memories and helmet pseudo-label targets with the already documented pipeline;
- one group per unique source-image SHA256.

Do not restrict the search to the historically evaluated 921-group holdout if other pre-existing source data/manifests can generate valid groups; conversely, do not fabricate new identity pairs unsupported by source annotations.

Produce `candidate_inventory.json` with counts by source/manifold and exclusion reason. No LISA/CMA/SegLLM inference and no scoring.

## C. Freeze a candidate manifest only if inventory is sufficient

If at least `30` valid source-image-disjoint candidates remain, deterministically order them by

`sha256("CYCLE024:" + stable_source_image_sha256 + ":" + counterfactual_id)`

then ID, and freeze up to the first `50` in `documented_protocol_disjoint_candidates.jsonl` with source/memory/target provenance.

Label the status exactly:

`DOCUMENTED_PROTOCOL_DISJOINT_CANDIDATES`

—not `training-unseen test set`, because exact historical w15 exposure is still unknown.

Do not restore/regenerate all masks unless needed to establish deterministic source metadata/validity; do not run any model. If target reconstruction is required for validity, hash/freeze it without looking at model outputs.

If fewer than `30` valid candidates remain, stop and record the shortfall. Do **not** relax the exclusion registry, reuse the 13 post-hoc Cycle022 nonoverlap images, or merge previously evaluated images merely to hit a round number.

## D. Decision rule for the following review

- **If >=30 candidates exist:** recommend exactly one later frozen evaluation cycle using unchanged CMA base-w15 and pinned SegLLM, with the candidate manifest selected before inference. The next review will decide whether that is worth running and how to word its provenance.
- **If <30 candidates exist:** recommend no further in-repository benchmark recycling. The next scientific step should be independent new image collection / external dataset acquisition plus image-level split and annotation, not another model or split trick.

Regardless of inventory outcome, do not resurrect Layer 2 or method search.

## Non-goals

- no CMA/SegLLM inference;
- no rescoring Cycle018/022 subsets;
- no retraining or checkpoint replacement;
- no prompt/threshold/degradation changes;
- no new recovery module;
- no Layer-2 verifier/controller/Qwen/RL;
- no bootstrap/significance analysis;
- no claim that documented-protocol disjointness proves exact historical training disjointness.

## Deliverable

Before `CODEX UPDATE 024`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 024` with:

1. exclusion-registry counts and provenance classes;
2. image-byte split-integrity audit result;
3. candidate inventory and exclusion reasons;
4. if available, frozen deterministic 30–50 group candidate manifest and status `DOCUMENTED_PROTOCOL_DISJOINT_CANDIDATES`;
5. one explicit conclusion: `ENOUGH_FOR_STRONGER_FROZEN_EVAL` or `NEED_NEW_INDEPENDENT_DATA`;
6. exactly one next recommendation under the decision rule above.
