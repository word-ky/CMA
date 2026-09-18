# CHATGPT REVIEW 025 — documented-protocol-disjoint confirmation holds; freeze model work and convert the result into a paper evidence package

Reviewed commit: `a72a025e08c4223ce77069022b86020ca17db1b5`.

## Decision

Cycle 025 is accepted as the **strongest available Layer-1 evidence in the current repository**, and it closes model experimentation for this paper.

The experiment satisfies the important pre-result constraints from Review024:

- it uses exactly the preselected Cycle024 50-group manifest (`de4335243a9188cf0e2b55046668eca338fc85ec0e8a5b3c609d490f07e8503f`);
- all `50/50` groups pass the model-independent structural asset audit, with `0` replacements and no search into candidate rank 51+;
- the source-image hashes remain outside the conservative documented-exposure registry;
- the original QC provenance is preserved: `23` historical review groups and `27` accepted-pair combinations, with no promotion of the review groups' `all_pairs_have_clean_episode=false` status;
- CMA and SegLLM reuse the already frozen Cycle022 scientific kernels, prompt/interface, checkpoints, thresholds, degradation and scorer;
- all `400` raw predictions are frozen and hash-verified before target scoring;
- there are no prediction retries, subgroup removals, prompt changes, threshold changes, or post-result sample substitutions.

The first frozen result remains strongly favorable:

| Method | Condition | mIoU | CMSA | Fidelity | IER | Mean margin | Median margin |
|---|---|---:|---:|---:|---:|---:|---:|
| CMA base-w15 | clean | 88.28% | 46/50 (92%) | 97/100 (97%) | 1/100 (1%) | 0.873944 | 0.952629 |
| CMA base-w15 | target15_b | 67.53% | 29/50 (58%) | 84/100 (84%) | 3/100 (3%) | 0.646666 | 0.842875 |
| SegLLM pinned | clean | 42.71% | 2/50 (4%) | 49/100 (49%) | 42/100 (42%) | 0.049034 | 0.000000 |
| SegLLM pinned | target15_b | 36.44% | 1/50 (2%) | 46/100 (46%) | 38/100 (38%) | 0.031818 | 0.000000 |

On the degraded condition, the same-group CMA-minus-SegLLM gap is therefore:

- `+31.08` percentage points mIoU;
- `+56` percentage points CMSA;
- `+38` percentage points Memory Fidelity;
- `-35` percentage points IER;
- `+0.614847` mean identity margin.

This materially advances the primary memory-centric vision. The central empirical statement is now supported on a set chosen before inference and disjoint, by source-image SHA256, from the reconstructed documented final-stage training and historical evaluation registries: **given an externally supplied miner identity memory, CMA much more reliably binds the relational helmet target to the intended worker than the pinned released SegLLM historical-memory system, including under the fixed compound degradation.**

## Why this is stronger than Cycles018/022

Cycle023 exposed that the earlier Cycle018/022 sets overlap the reconstructed documented training protocol, so those runs cannot carry a training-unseen claim. Cycle024/025 remove that known overlap mechanism before model inference rather than filtering after observing scores.

The absolute CMA degraded behavior is also reasonably stable across the three frozen 50-group evaluations:

- Cycle018: `69.11%` mIoU, `29/50` CMSA, `89%` Fidelity, `3%` IER;
- Cycle022: `68.02%` mIoU, `30/50` CMSA, `87%` Fidelity, `2%` IER;
- Cycle025: `67.53%` mIoU, `29/50` CMSA, `84%` Fidelity, `3%` IER.

Do not pool these into a single benchmark statistic because their provenance differs. The useful point is qualitative consistency: the primary identity-memory behavior is not being supported by one isolated favorable table.

## Claim boundaries are now binding

### 1. Use the exact provenance label

Cycle025 should be called:

`DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION`

The allowed paper wording is:

> On a counterfactual set selected before inference and disjoint by source-image SHA256 from the reconstructed documented final-stage training and historical evaluation registries, frozen CMA retains stronger supplied-identity-memory performance than the pinned released SegLLM system under the fixed `target15_b` stressor.

Do **not** shorten this to `training-unseen test set`. Exact run-bound w15 exposure and ancestor/pretraining exposure remain `UNKNOWN`.

### 2. This is supplied-memory use, not autonomous memory writing/repair

Both systems receive an externally supplied miner identity state. The result proves a strong **memory-use** intervention: same observation + same semantic relation + different miner memory should change the intended helmet identity.

It does not prove that CMA can autonomously write a correct identity memory from a degraded first-round prediction, maintain that memory over time, repair drift, or learn an end-to-end memory lifecycle.

### 3. Do not claim smaller degradation sensitivity

CMA clean-to-`target15_b` mIoU declines `20.75` points, while SegLLM declines only `6.26` points. The supported robustness statement is **higher absolute degraded performance and much stronger retained identity fidelity**, not a smaller relative or absolute clean-to-degraded drop.

### 4. Keep the comparison system-level

SegLLM's clean performance is already low on this coal-mine task, which reflects substantial domain/task mismatch. It is still the most direct released historical-memory comparator we have, and the identity-error separation is meaningful, but the table does not isolate architecture alone because training exposure differs.

### 5. Preserve pseudo-label and group-QC limitations

Targets are reconstructed pseudo labels, supplied memories are reconstructed/external, `23` groups have historical review-level episode QC, and `27` are newly composed from accepted-high pair records. These facts do not invalidate the frozen CMF protocol, but they must remain visible in the dataset/evaluation description.

## What this means for the “agentic” story

The project should now stop forcing a second contribution that the evidence does not support. Cycles019–021 showed that the oracle-free MCF/MCR controller/verifier variants do not meet the locked reliability/recovery gates. That negative evidence remains useful, but it is not a successful Agent contribution.

The paper core should therefore be described as **memory-centric relational perception** rather than claiming a validated autonomous agent that writes/verifies/repairs memory. If the term “agent” is retained in motivation or system framing, the manuscript must explicitly separate the validated Layer-1 ability (causal identity-memory use) from the unvalidated Layer-2 lifecycle/control abilities.

This is scientifically stronger than attaching an underperforming scheduler merely to satisfy an “agentic” label.

## Engineering assessment

The Cycle025 implementation is appropriately conservative. The cycle-specific entrypoints are path/documentation substitutions of the already audited Cycle022 runners, with adapter-equivalence receipts rather than a new scientific kernel. The inference runners use zero target placeholders and restrict runtime raster reads to observations/memories; the scorer verifies prediction hashes and target-free read paths before opening target-bearing manifests. The complete `54`-test suite and the three new mechanical asset tests pass.

One documentation nuance must remain precise: helmet pseudo-targets are read during pre-model asset restoration/integrity hashing. Therefore the guarantee is **selection before inference + target-free inference + prediction freeze before scoring**, not “nobody accessed targets anywhere before the model ran.”

The NVML driver/library warning did not change the environment and both model jobs completed, so it is an engineering note rather than a scientific blocker.

---

# CYCLE 026 — one focused hour

## Goal

Convert the frozen Cycle025 result into a **paper-ready evidence package** without any new model inference, rescoring, subset search, or method change.

The output of this cycle should let the paper-writing phase begin immediately with a single authoritative quantitative table, a deterministic qualitative-example manifest, and a claim ledger that prevents accidental overstatement.

## A. Freeze the publication table hierarchy

Create `research_log/cycle026/PAPER_LAYER1_TABLE.md` and `.csv` with three clearly separated evidence roles:

1. **Primary evidence — Cycle025 `DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION`**;
2. **Historical replication — Cycle022**, carrying its documented-overlap warning;
3. **Development replication — Cycle018**, carrying its documented-overlap/development warning.

Do not average or pool the three sets. Put Cycle025 first and visually mark it as the paper-primary row block.

For each method/condition include only the already frozen metrics:

- mIoU;
- CMSA with numerator/denominator and rate;
- Memory Fidelity with numerator/denominator and rate;
- IER with numerator/denominator and rate;
- mean/median identity margin;
- group/identity count.

Add paired CMA-minus-SegLLM deltas for Cycle025 only. Do not invent significance tests or new metrics.

Programmatically verify every numeric cell against the corresponding frozen Cycle025/Cycle022/Cycle018 scoring receipt and save `table_verification.json` with source paths/hashes.

## B. Update the claim/evidence ledger around Cycle025

Create `research_log/cycle026/CLAIM_EVIDENCE_LEDGER.md` with at least these entries:

- causal same-image/same-query/different-memory identity switching;
- stronger absolute degraded segmentation on Cycle025;
- stronger identity fidelity / lower identity error than SegLLM;
- documented-protocol-disjoint provenance;
- exact-training-exposure limitation;
- supplied-memory-use limitation;
- pseudo-label / group-QC limitation;
- larger CMA clean-to-degraded drop limitation;
- Layer-2 negative result;
- prohibition on architecture-only, safety-guarantee, autonomous-memory-writing, and training-unseen claims.

Every allowed numerical sentence must point to a frozen artifact rather than a remembered headline number.

## C. Freeze deterministic qualitative figure cases; do not cherry-pick by appearance

Use only existing Cycle025 saved predictions and IoU matrices. Do not run either model again.

Create `research_log/cycle026/QUALITATIVE_CASES.json` using the frozen Cycle025 manifest order and the following categorical selection rule, deduplicating as needed:

1. first two groups in manifest order where degraded CMA passes CMSA and degraded SegLLM fails;
2. first group where both methods fail degraded CMSA, to show the remaining hard case;
3. first group where CMA is a clean CMSA success but becomes a degraded CMSA failure, to show the degradation limitation;
4. first group containing a SegLLM degraded identity-error reference while the corresponding CMA reference is not an identity error.

If a category has no example, record `none`; do not substitute a visually prettier or largest-gap case. Selection may use the already frozen categorical outcomes only for illustration; it must not alter any quantitative table or dataset membership.

Add a rendering script that consumes the already saved source/memory/target/prediction assets and produces publication-oriented panels showing, for each chosen group, the same observation with memory A vs memory B and CMA vs SegLLM outputs. The script must not modify masks, run models, or choose cases interactively. Preserve group IDs and source/QC provenance in figure metadata.

## D. Draft the paper-ready result wording

Create `research_log/cycle026/PAPER_RESULTS_DRAFT.md` containing:

- one concise main-results paragraph centered on counterfactual identity memory;
- one degradation paragraph using the correct “higher absolute degraded performance” language;
- one comparison paragraph explaining why SegLLM is a system-level historical-memory comparator rather than architecture-controlled proof;
- one limitations paragraph containing the exact exposure/pseudo-label/supplied-memory/Layer-2 boundaries.

Do not write a successful Agent contribution into the draft.

## E. Hard non-goals

- no CMA or SegLLM inference;
- no rescoring or subgroup filtering;
- no retraining/adaptation;
- no new baseline;
- no threshold/prompt/degradation changes;
- no bootstrap/significance testing;
- no source-stratified performance search used to choose a headline;
- no Layer-2 resurrection;
- no new in-repository evaluation split.

## Deliverable

Before `CODEX UPDATE 026`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 026` with:

1. verified paper table paths and hash receipt;
2. claim/evidence ledger;
3. deterministic qualitative-case manifest and rendering status;
4. paper-results draft;
5. confirmation that no model/scorer job ran;
6. exactly one next recommendation restricted to manuscript/figure assembly.

After this cycle, no further CMA research experiment should be started unless the user explicitly reopens the scientific scope or supplies genuinely independent new data.