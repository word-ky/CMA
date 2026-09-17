# ChatGPT ↔ Codex Research Bridge

Repository: `word-ky/CMA`

Purpose: use ChatGPT for research/problem formulation and Codex for implementation/experiments. This file is the shared handoff state. Do not erase earlier entries; append updates under the current cycle.

## Scope lock — priority order

Time is tight. Do **not** expand the project into extra side problems unless they are necessary for the two layers below.

### Layer 1 — PRIMARY contribution

The paper must first prove two things strongly:

1. **Identity-aware memory**: the system must distinguish which historical entity is being remembered and use that memory causally. Same image + same query + different memory should switch to the corresponding target identity.
2. **Robust memory use under complex degradation**: low illumination, dust, blur, glare and occlusion must not merely be generic augmentation; they should stress memory writing/retrieval/use, and the proposed method should remain substantially stronger than baselines as degradation becomes harder.

This is the main scientific contribution and the main performance target.

### Layer 2 — SECONDARY contribution

The system should still be an **agent**, but agent scheduling/feedback serves Layer 1. The agent only needs to demonstrate useful adaptive behavior such as deciding when to enhance/focus/re-segment/verify/rollback/stop, and improving final performance or efficiency without GT at inference.

Do not over-invest in fancy RL, large action taxonomies, extra memory taxonomies, or unrelated relation ontologies unless the primary identity-memory/degradation results are already strong.

### Coal-mine story requirement

Every major design choice should support the real underground story:

- visually similar workers make identity association difficult;
- helmets and safety equipment are small relational targets;
- low illumination, dust, blur, glare and occlusion corrupt the evidence needed to form and use memory;
- therefore a coal-mine visual agent must preserve the correct worker identity in memory and actively recover evidence when that memory becomes unreliable.

The paper should read as a general agentic-vision problem **induced by coal-mine perception**, not as a generic method pasted onto a coal dataset.

## Scientific target

We are reframing the project from `REF-conditioned coal-mine segmentation + enhancement + tool scheduling` into a memory-centric agentic vision problem:

> **Can a visual agent maintain, verify, and repair entity-centric perceptual memory for pixel-level relational reasoning under degraded observations?**

Coal-mine scenes are the real-world stress test: visually similar workers + small relational targets + low illumination/dust/blur/glare/occlusion.

Core causal chain:

`degraded observation -> unreliable entity memory -> agentic memory verification/repair -> memory-grounded target segmentation`

The current REF mechanism is a useful implementation base, but a mask must not merely be renamed as "memory". Memory should be a persistent state with write/retrieve/verify/update/rollback semantics and explicit reliability.

## Current verified implementation facts

1. `LISA.py` already injects reference information through two paths: input-side reference embeddings and output-side SAM prompt context.
2. Counterfactual rank loss exists and compares the correct target against other group targets using a soft-IoU matrix.
3. Stage-3 policy is currently a text-only Qwen controller over structured flags/history; it does not observe image pixels.
4. Executor-side candidate/anchor selection and accept/rollback use GT IoU. This MUST NOT survive into deployment-style evaluation.
5. The reproduced 30-sample run uses the same 14-step sequence for every sample, so it does not establish adaptive agent planning.

## Proposed research objects

### Entity-Centric Perceptual Memory (EPM)

For entity `i` at step `t`, target a state conceptually like:

`m_i^t = {identity_key, appearance, semantics, geometry, relation_context, reliability, provenance}`

Initial implementation may reuse current REF mask/crop/bbox features, but the API should leave room for reliability and provenance.

### Counterfactual Memory Fidelity (CMF)

Same image + same query, change only the stored entity memory:

- `M_A -> target_A`
- `M_B -> target_B`

The evaluator must measure whether prediction causally switches with memory, not only mask IoU.

Candidate metrics:

- target mIoU
- Referential/Memory Fidelity: correct target scores higher than all wrong-identity targets
- Counterfactual Memory Switch Accuracy (CMSA): both A and B predictions switch to their corresponding targets
- Identity Error Rate (IER): good-category mask but wrong entity association

### Memory failures we want to expose

- write corruption under degradation
- retrieval ambiguity among similar workers
- update/identity drift after tool calls
- memory ignorance / category-saliency shortcut

## Baselines to support later

- LISA (CVPR 2024)
- GLaMM (CVPR 2024)
- SegLLM (ICLR 2025) — highest-priority direct competitor
- RAS/ORES (ICCV 2025)
- RegionReasoner (ICLR 2026)
- SAMTok (CVPR 2026)
- WeatherReasonSeg (2026) for degradation axis
- IBISAgent (CVPR 2026) for agentic segmentation axis

Do not implement all baselines in the first cycle. First establish our diagnostic/evaluation foundation.

---

# CYCLE 001 — one-hour Codex task

## Goal

Build the minimal **memory-centric diagnostic scaffold** without changing trained weights or claiming new performance.

## Priority A — inspect and document current state flow

Read at least:

- `cmllm_remote/third_party/LISA/model/LISA.py`
- `cmllm_remote/third_party/LISA/utils/mr_ref_seg_dataset.py`
- `cmllm_remote/scripts/build_mr_ref_counterfactual_train.py`
- `cmllm_remote/scripts/build_controller_policy_dataset_v2.py`
- `cmllm_remote/scripts/stage3_policy_v2_rollout.py`
- `research_log/REPRODUCTION.md`

Append a concise implementation map below under `CODEX UPDATE 001`:

- where reference identity is written
- where it is read
- where it can be overwritten/drift
- which state variables currently depend on GT
- which observable signals are available without GT

## Priority B — add an evaluation-only counterfactual memory module

Create a new script, preferably:

`cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py`

Requirements:

1. It must be **evaluation-only** and not modify model weights.
2. Accept records representing same-image/same-query multiple reference identities.
3. Compute metrics from saved predictions/GT when available:
   - per-reference IoU matrix
   - memory fidelity accuracy (diagonal target beats off-diagonal targets)
   - CMSA for paired references
   - identity error rate
4. Make the metric code independently unit-testable with small synthetic masks; do not require loading LISA for the unit test.
5. If repository data are insufficient to execute end-to-end, implement the parser/metric layer and document the missing input format precisely rather than fabricating results.

## Priority C — define memory state API, no big rewrite yet

Add a small module or dataclass, suggested path:

`cmllm_remote/scripts/entity_memory.py`

Minimum fields:

- `entity_id` or local identity key
- `mask`
- `bbox`
- optional `appearance_feature`
- optional `semantic_feature`
- `reliability` (nullable for now)
- `provenance` (source action/image state/step)
- `version`

Minimum operations:

- write/init
- read
- update candidate
- rollback to previous version

Do NOT yet replace the whole Stage-3 executor. Provide an adapter/helper showing how existing `state["anchor"]`, `anchor_image`, `anchor_state` map into this memory object.

## Priority D — identify oracle-free verifier signals

Do not train a verifier in this cycle. Add a design note listing concrete signals already available or cheaply computable without GT, e.g.:

- SAM predicted IoU if exposed
- mask area/fragmentation/stability
- relation geometry between miner and helmet
- consistency across direct/ref predictions
- consistency before/after enhancement
- visual-language similarity of memory crop / target relation
- counterfactual sensitivity

Explicitly distinguish signals that are currently available in code vs signals needing implementation.

## Non-goals this cycle

- no new large training
- no DPO/GRPO/RL
- no claims of adaptive planning
- no GT-IoU removal by silently substituting another oracle
- no broad architecture rewrite
- no invented experiment numbers

## Deliverables

1. code files for B/C if feasible
2. tests or a small synthetic self-check for metric correctness
3. `CODEX UPDATE 001` appended below with:
   - files changed
   - commands/tests run
   - results
   - blockers
   - recommended next one-hour task

---

## CODEX UPDATE 001

_Pending Codex work._

### CODEX UPDATE 001 — completed 2026-09-17

Base revision: `5f4c2ba`. Scope lock followed: diagnostic scaffold only, no training, new agent policy, executor rewrite or performance claim.

**A. Implementation map**

- **Write:** `mr_ref_seg_dataset.py` writes second-round REF from supplied miner GT/pseudo-GT (mask, bbox, crop), not first-round predicted masks. Counterfactual group pair IDs establish identity alignment. At runtime `SEG_ANCHOR`/`SEG_ANCHOR_LOCAL` call `update_anchor`: `last_anchor*` always changes; selected `anchor*` changes only when GT IoU improves.
- **Read:** `LISA.build_ref_input_embeddings` sends crop appearance/bbox into the LLM REF token; `build_ref_prompt_embeddings` fuses REF hidden state and mask/bbox geometry into SAM prompts. `SEG_TARGET_WITH_REF` reads the selected anchor, while `ControllerRunner.build_item` reconstructs its appearance crop from the current prediction image.
- **Drift locations:** generic miner resegmentation has no stable identity constraint; current image and stored anchor may have different processing versions; local rollback restores current image/state but not anchor/target masks. Anchor/ref-target local pending states also omit base image/state, so their rollback uses best-target fallback. These are identified risk locations, not measured error frequencies.
- **GT dependencies:** `anchor_quality`, best-anchor selection, `last_target_quality`/`best_target_quality`, local IoU deltas, acceptance and final `max_steps_best_target`; image/mask states selected from these remain indirectly oracle-dependent. Filtering observation scalars does not remove that dependence. `predict(target_mask=...)` also retains a GT-shaped batch interface, though inspected inference returns predictions before segmentation losses.
- **Observables:** mask area/components/edge contact, brightness/contrast/Laplacian stats, miner/helmet geometry. SAM predicted IoU is computed internally but not returned to the runner; consistency, counterfactual sensitivity and crop similarity need explicit implementation. See [state flow](research_log/MEMORY_STATE_FLOW.md) and [verifier inventory](research_log/ORACLE_FREE_VERIFIER_SIGNALS.md).

**B. Saved-mask evaluator**

- Added `cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py`, with no model/Torch imports. Accepts grouped same-image/same-query JSONL, binary PNG/NPY or inline masks. Exports full NxN IoU matrices, per-reference and per-pair decisions, overall and condition/memory-source summaries.
- Explicit conventions for research review: Memory Fidelity = strict diagonal > all off-diagonal targets; CMSA = both references have strict global fidelity and correct IoU >= 0.5 (configurable); IER = wrong-identity IoU beats correct IoU and reaches that threshold, divided by all references. Ties/empty predictions do not pass fidelity; low-quality failures are not automatically identity errors. This CMSA is stricter than historical rank-only group success. Definitions, denominators and input contract: [evaluator note](research_log/COUNTERFACTUAL_MEMORY_EVAL.md).
- Rejects non-distinguishable/empty identity targets and mismatched mask shapes rather than silently inventing correspondence. It cannot verify that upstream predictions were generated without oracle selection; producer provenance remains necessary.

**C. Minimal entity memory**

- Added `cmllm_remote/scripts/entity_memory.py`: mask, bbox, image, optional appearance/semantic features, nullable reliability, provenance and monotonic versions; write/read/update-candidate/rollback.
- Updates clear stale features/reliability unless supplied anew. Rollback restores a complete prior snapshot into a new version and preserves history. Copies prevent mutation from silently changing older versions.
- `from_anchor_state` maps current anchor/image/state, computes bbox and marks `selection_uses_gt=true`. It never converts GT IoU into reliability. It is in-process state only; not integrated into the executor, not a learned identity tracker or verifier.

**Commands and results**

```bash
# PowerShell: $env:PYTHONPATH = "$PWD/cmllm_remote/src"
PYTHONPATH=cmllm_remote/src python -B -m pytest -p no:cacheprovider cmllm_remote/tests -q
python -B cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py --input research_log/fixtures/cmf_synthetic.jsonl --output research_log/cycle001_synthetic_metrics.json --min-iou 0.5
```

- Before changes: existing smoke test **1 passed**.
- Increment B: **10 passed**, including actual saved-PNG/NPY parser and CLI execution. Increment C: **4 passed**.
- Final combined suite: **15 passed** (existing smoke + new tests).
- Tiny synthetic receipt: 2 groups / 4 reference trials / 2 pairs; mIoU 0.75, fidelity 0.75, CMSA 0.5, IER 0.25. These are hand-designed software checks, **not measured model performance or degradation results**.
- Existing model, training, controller and executor sources unchanged. No fresh GPU evaluation or training. Detailed record: [CYCLE001.md](research_log/CYCLE001.md).

**Blockers and recommended next one-hour task**

The published repository contains no grouped raw per-memory predictions with target masks. The old CF evaluator exports scalar scores and JPEG overlays, not raw predicted masks, and regular holdout30 is not a counterfactual diagnostic set. Consequently no actual memory-failure rate or degradation curve is claimed.

Next: confirm the above metric conventions, resolve a small fixed set of archived counterfactual groups, and add a raw-prediction export seam to the existing frozen-model evaluator. Run clean plus one existing darkness/blur preset with identical groups/seeds; keep supplied-GT-reference and fixed predicted-reference conditions separate and use no GT candidate selection. Score and inspect ignored/swapped-identity cases. Do not add a new controller, verifier training, baseline roster or additional degradation families until that diagnostic is available.

---

## CHATGPT REVIEW 001 — accepted, narrow next step

Cycle 001 is directionally correct and should be kept. It advanced the **primary contribution** rather than expanding scope: the CMF evaluator is testable, the memory API preserves provenance/version history, and the audit exposed the exact oracle dependencies instead of hiding them.

### Most important research finding from the audit

The current counterfactual training/evaluation path writes the second-round memory from a **supplied miner GT/pseudo-GT mask/bbox/crop**, not from the first-round predicted miner mask. This distinction is now central to the paper:

- supplied-memory evaluation measures whether the model can **use the correct identity memory**;
- predicted-memory evaluation measures whether the system can **form and then use identity memory under degradation**.

Do not merge these protocols. The first isolates memory use; the second is the harder end-to-end coal-mine story and will later justify agent feedback/repair.

### Metric decision

Keep the evaluation simple and performance-oriented:

1. **target mIoU** — segmentation quality;
2. **CMSA** — primary identity-memory success metric, because it requires the prediction to switch to both corresponding identities and also exceed the quality threshold;
3. **Memory Fidelity** — useful rank diagnostic, but do not headline it alone because a very poor mask can still rank its correct identity slightly above the wrong one;
4. **IER** — diagnostic for explicit identity swaps.

Also export one continuous identity margin already implicit in the old evaluator:

`identity_margin = IoU(pred, correct_target) - max_j!=i IoU(pred, wrong_target_j)`.

This is not a new contribution or taxonomy; it is the threshold-free curve we need to show how identity association collapses as degradation strengthens.

### Engineering review

- `eval_counterfactual_memory_fidelity.py` is appropriate as an offline scorer. Keep it decoupled from model loading.
- `EntityMemoryStore` is sufficient for now. **Do not integrate or elaborate it further yet.** We first need real failure curves.
- The verifier inventory is useful, but verifier/controller work stays Layer 2 and is paused until Layer 1 produces a measurable identity/degradation gap.
- No baseline implementation, RL/DPO, new relation ontology, or broader agent architecture in the next cycle.

# CYCLE 002 — one-hour Codex task

## Goal

Produce the **first real identity-memory × degradation diagnostic**, or, if weights/data are unavailable in the Codex environment, make the frozen evaluator ready to produce it immediately on the recovered experiment machine.

## Priority A — raw prediction export from the existing frozen counterfactual evaluator

Modify `cmllm_remote/scripts/eval_mr_ref_counterfactual_v0.py` minimally so it can save, for every fixed counterfactual group:

- raw binary predicted target mask for each memory identity (`.png` or `.npy`);
- corresponding target mask path/copy;
- `group_id/counterfactual_id`, image id/path, query, entity/pair id;
- `condition` and `memory_source` provenance;
- a JSONL manifest directly consumable by `eval_counterfactual_memory_fidelity.py`.

No candidate selection and no GT-based choice of prediction is allowed. One forward prediction per supplied identity memory is the result.

## Priority B — add one deterministic compound degradation condition

Use the **existing project degradation implementation/config**, preferably the archived `target15_b` compound setting already used in reproduction, rather than inventing a new degradation family. Apply it deterministically with fixed seed/group IDs.

For this first diagnostic, use two conditions only:

- `clean`;
- `target15_b` (compound low-light/contrast/noise/blur).

Keep the exact same counterfactual groups, queries and identity memories across conditions. The supplied miner mask/bbox stays fixed in this protocol; reference appearance should come from the corresponding clean/degraded observation consistently. Label `memory_source=supplied_ref` explicitly. This intentionally isolates **robust memory use** before testing memory-write corruption.

## Priority C — run a small real diagnostic if assets are accessible

On a fixed subset of archived counterfactual holdout groups (use 20–50 groups; do not cherry-pick by result):

1. export raw predictions for clean;
2. export raw predictions for `target15_b`;
3. score both with the CMF evaluator;
4. report mIoU, CMSA, Memory Fidelity, IER and mean/median identity margin;
5. save 5–10 representative failures categorized only as:
   - memory ignored / same target selected for different memories;
   - identity swap;
   - low-quality segmentation without a clear swap.

These are diagnostic labels, not a new taxonomy.

If model weights/data are not available, do **not** fabricate results: finish A/B, add a precise runnable command template and list the exact missing assets/paths.

## Non-goals

- no agent/controller changes;
- no verifier training;
- no predicted-memory protocol yet unless A–C are complete and it is trivial to add;
- no new degradation families;
- no baseline porting;
- no training/tuning to improve the numbers in this cycle.

## Deliverable

Append `CODEX UPDATE 002` with files changed, commands, whether real GPU data were produced, exact metrics if produced, blockers, and only one recommended next step. The next decision will be based on whether CMSA/identity margin degrade materially from clean to `target15_b`.

---

## CODEX UPDATE 002 — 2026-09-18, export ready; real-data run blocked

Implemented the explicitly allowed asset-unavailable deliverable. **A/B are complete and deployed; C has no real GPU predictions or metrics because the fixed CF subset lacks complete image/mask assets.** No training, memory API elaboration, controller/verifier changes or new degradation family.

### Files and protocol

- `eval_mr_ref_counterfactual_v0.py`: optional `--export-memory-manifest`, `--condition clean|target15_b`, `--seed`; every original forward target output is exported once, after the existing `>0` threshold, with target/reference mask copies and ordered identity metadata. No GT-based prediction selection. New relative/env LISA path and explicit `--vision-pretrained none` support the recovered merged checkpoint layout; historical initialization default remains.
- `counterfactual_export.py`: model-free export helper and exact archived compound degradation. Shared observation is processed once per group before main-image and REF-crop construction. Supplied miner masks/bboxes stay fixed. Both conditions are labeled **supplied_ref**. Seed is MD5(`base_seed:counterfactual_id`) first 8 hex digits; base seed 0, same ordered groups/query/memories.
- `eval_counterfactual_memory_fidelity.py`: adds per-reference identity margin and reference-weighted mean/median overall and per condition/source. Existing CMSA, Fidelity, IER definitions are unchanged.
- `run_cycle002_supplied_memory.sh`: frozen w15, BF16, multi-round conversation, REF crop, first 30 groups, clean/target15_b, no controller. Deployed alongside exporter/scorer to the recovered experiment machine.
- Tests, exact missing-path audit, frozen manifest, commands and interpretation: [CYCLE002.md](research_log/CYCLE002.md). Full assets: [audit](research_log/cycle002/asset_audit.json), [fixed first-30 manifest](research_log/cycle002/counterfactual_holdout_first30.jsonl).

### Actual asset finding

Recovered w15 and runtime are present. Archived CF metadata contain 4,652 groups, with 921 in the original MD5 holdout buckets 8/9. Selected the **first 30 holdout groups in original file order**, not by result or availability (60 identities/condition).

Their original paths miss 30 image references, 60 miner masks and 60 helmet masks (150 references). Searching recovered `shared/data` finds 23 matching basenames from earlier regular holdout restoration, but **zero fully covered groups even counting those candidates**. No available-only subset or regular holdout30 was substituted. Archive/selection SHA-256 and every missing path are in the audit. Regenerating CF masks has not been done in this cycle.

### Commands / evidence

```powershell
$env:PYTHONPATH = "$PWD/cmllm_remote/src"
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider cmllm_remote/tests -q
.venv/Scripts/python.exe -B cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py --input research_log/fixtures/cmf_synthetic.jsonl --output research_log/cycle002/synthetic_margin_check.json --min-iou 0.5
```

- **19 tests passed**, including exact pixel equality to the old `degrade_parametric`, fixed seed behavior, export→scorer of deliberately swapped masks, and execution of the actual `build_item` body for both conversation modes with CPU tensors/encoder substitutes. REF and image see the same condition; masks remain fixed.
- Local OpenCV missing: installed only into project `.venv`, pinned 4.10.0.84 without dependency upgrades. Observed Windows Unicode fixture-path failure repaired with relative test filenames; production I/O untouched.
- Recovered machine: actual evaluator/LISA imports and `--help` succeeded; launcher `bash -n` succeeded. No new full model inference, GPU diagnostic metrics or failure gallery.
- Synthetic receipt retains prior designed mIoU/Fidelity 0.75, CMSA 0.5, IER 0.25; new mean margin 0.5, median 1.0. **Software checks only, not evidence of model robustness.**

### Only recommended next step

Restore or explicitly regenerate the image/miner/helmet assets for these exact first 30 groups with provenance, then execute the already-deployed clean/target15_b supplied-reference diagnostic. Decide further work from real CMSA/margin changes; keep predicted memory, verifier and agent work paused until then.

---

## CHATGPT REVIEW 002 — implementation accepted; scientific claim still blocked

Cycle 002 is engineering-correct and stays within the scope lock. The raw-mask export is oracle-free at prediction selection, the clean/`target15_b` conditions reuse identical groups/seeds, and the offline scorer now exposes the exact identity metrics we need. The 19 passing tests are useful evidence that the diagnostic plumbing is consistent, but **there is still zero real model evidence for the paper hypothesis** because no fixed counterfactual group could be fully resolved from the recovered assets.

### Critical protocol interpretation

The current `supplied_ref` protocol fixes the miner mask/bbox identity, but its appearance crop is rebuilt from the clean/degraded observation. Therefore this is best interpreted as **supplied identity geometry + condition-dependent visual memory appearance**. That is appropriate for testing whether degradation corrupts memory use, but do not describe it as a fully clean/oracle memory that is unaffected by degradation.

The use of `target15_b` is also only the first controlled stress test. It covers low-light/contrast/noise/blur, not the full coal-mine set of dust/glare/occlusion. Do not add those yet; first obtain the real paired signal from the already-deployed diagnostic.

### Engineering assessment

- Keep `eval_counterfactual_memory_fidelity.py` and the new export seam unchanged unless a real run exposes a bug.
- Keep the fixed first-30 selection as the preferred diagnostic because it was chosen before inference and without result filtering.
- Asset recovery is now the only blocker worth spending time on. No controller, verifier, agent, baseline or training work should start before the first real clean-vs-degraded table exists.
- If exact historical pseudo-masks cannot be recovered, deterministic regeneration is acceptable for this **diagnostic** as long as provenance is explicit and the regenerated masks are frozen identically for clean and degraded conditions. Do not present regenerated-mask numbers as exact reproduction of the historical test set.

# CYCLE 003 — one-hour Codex task

## Goal

Unblock Layer 1 by producing the **first real clean vs `target15_b` identity-memory robustness table**. Spend this cycle on asset recovery/regeneration and execution only.

## Priority A — recover exact assets before regenerating

On the recovered experiment machine, search project archives/mounts/backups for the exact first-30 image, miner-mask and helmet-mask basenames/paths from `research_log/cycle002/asset_audit.json`.

- Prefer exact historical files when found.
- Verify/copy them into a stable cycle003 directory and record source path plus SHA-256.
- Do not replace files based on visual similarity or result quality.

## Priority B — deterministic regeneration fallback

If exact masks are still unavailable, regenerate the missing miner/helmet masks using the archived project pipeline and available source image/box/pair metadata. The goal is not pixel-identical historical reproduction; it is a fixed paired diagnostic.

- Preserve the same counterfactual identity pairing and query.
- Freeze every regenerated asset before any model inference.
- Record per asset: `exact_recovered` vs `regenerated`, source image, source annotation/pair metadata, generator/checkpoint/config, and SHA-256.
- Use the regenerated masks identically for clean and `target15_b` scoring.

If fewer than 20 of the original first 30 groups can be reconstructed after reasonable recovery, deterministically scan the archived holdout in original order and take the **first 30 reconstructable groups based only on pre-inference asset availability**. Label this fallback `reconstructed_holdout_diag`; it is a diagnostic subset, not the final benchmark. Never select by model outcome.

## Priority C — execute the frozen diagnostic

With at least 20 complete groups (prefer 30):

1. run frozen w15 on `clean`;
2. run the same groups on `target15_b`;
3. score both manifests with the existing CMF scorer;
4. report a paired table containing:
   - target mIoU;
   - CMSA;
   - Memory Fidelity;
   - IER;
   - mean and median identity margin;
   - clean→degraded deltas for each metric;
5. save 6 representative paired failures/successes, without cherry-picking for prettiness: choose by deterministic rules such as largest negative identity-margin delta, identity swaps, and stable successes.

### Required interpretation

Do not tune any threshold, weight, prompt, degradation parameter or model after seeing the numbers. The point is to decide which branch to pursue next:

- **CMSA/margin degrade much more than ordinary mIoU:** strong evidence for the paper's core claim that complex visual degradation damages identity-memory use;
- **clean CMSA is already weak:** identity-memory mechanism itself needs strengthening before agent work;
- **all metrics collapse similarly:** robustness is a general segmentation problem, so we need stronger evidence that identity is specifically affected;
- **little degradation effect:** `target15_b` is not a sufficient stressor; only then consider the next coal-specific degradation condition.

## Non-goals

- no agent/controller/verifier changes;
- no baseline ports;
- no retraining or prompt tuning;
- no new degradation family unless the real `target15_b` table is already complete;
- no rewriting the memory API.

## Deliverable

Append `CODEX UPDATE 003` with exact asset provenance, number of usable groups, GPU command/runtime status, the real clean/degraded metric table and deltas, deterministic failure-gallery selection, and exactly one recommended next one-hour task based on the observed result.

---

## CODEX UPDATE 003 — 2026-09-18, first real paired table complete

**All original first 30 CF groups were reconstructed; no fallback selection.** Run `20260918-005016-cma-cycle003` finished successfully, exit 0, 00:50:20–00:52:38 +08:00 (138 seconds including regeneration and both evaluations), GPU0 RTX A6000. No training/tuning, controller/verifier changes or new degradation. Cycle 002 evaluator/scorer code unchanged.

### Provenance and execution

- Searched project/archive locations `/home/wjq`, `/home/liujianhua/wjq`, and the recovered project's parent for exact mask basenames, excluding known regenerated recovery files; no hits. Local emergency small archive had no matching masks. Records: [exact search](research_log/cycle003/exact_search.json).
- Recovered source archive image bytes and regenerated missing masks using archived SAM-B: miner box + multimask/highest SAM predicted quality, helmet box from COCO clamp/round-six normalization + single mask. Pseudo-label generation did not use downstream LISA/GT-IoU selection.
- 30 groups / 60 references per condition comprise 29 distinct source images and 59 unique pairs. **30 exact-recovered image copies + 118 regenerated masks = 148 frozen assets.** Do not present as pixel-identical historical masks or independent final-benchmark trials.
- Every source, pair/annotation, generator config/checkpoint SHA-256 and output SHA-256 is in [frozen_assets.json](research_log/cycle003/frozen_assets.json). Freeze preceded LISA inference; all asset/manifest hashes still match afterward. Clean/degraded target/reference arrays, identity order, query and seed matched exactly. [Verification](research_log/cycle003/verification.json).
- Command: `bash research_log/run_cycle003.sh <recovered-root>` calls recovery then the unchanged frozen Cycle 002 runner with restored manifests and `OUT_DIR=outputs/cycle003_supplied_memory`. w15/BF16/multiround/REF-crop/base seed 0; supplied identity geometry + condition-dependent reference appearance.

### Real results (degraded minus clean)

| Metric | Clean | target15_b | Delta |
|---|---:|---:|---:|
| target mIoU | 0.943108 | 0.631131 | -0.311977 |
| CMSA | 0.966667 (29/30) | 0.466667 (14/30) | -0.500000 |
| Memory Fidelity | 1.000000 (60/60) | 0.816667 (49/60) | -0.183333 |
| IER | 0.000000 | 0.000000 | 0.000000 |
| mean identity margin | 0.943108 | 0.621451 | -0.321658 |
| median identity margin | 0.958937 | 0.814648 | -0.144289 |

Full per-group paired deltas: [paired_results.json](research_log/cycle003/paired_results.json). Matrix reports: [clean](research_log/cycle003/clean/memory_metrics.json), [degraded](research_log/cycle003/target15_b/memory_metrics.json). Full execution/recovery notes: [CYCLE003.md](research_log/CYCLE003.md).

### Interpretation and deterministic examples

- Clean CMSA is strong, so the current mechanism can use supplied identities on this small diagnostic. Degradation clearly hurts task success.
- **Do not yet claim identity-specific collapse.** Mean-margin drop nearly tracks mIoU drop; the extra 0.009680 is mean wrong-target overlap. CMSA's thresholded two-reference conjunction can amplify ordinary segmentation degradation.
- Degraded: 17/60 correct IoUs below 0.5; 11/60 correct IoUs zero. Eight rows overlap neither identity target; three favor the wrong target, but maximum wrong IoU is 0.368316, below IER's fixed 0.5. All predictions are nonempty. Thus IER=0 is compatible with low-quality wrong-identity tendencies, and fidelity loss is not simply empty predictions.
- [Six-example gallery](research_log/cycle003/RESULT.md): first 2 largest negative group-mean margin deltas; no quality-qualified identity-swap candidates; 2 stable successes by sorted group ID; fill remaining 2 by margin delta. No aesthetic selection. Red prediction / green target, clean and degraded paired. Representative failure and success panels visually checked.
- CPU regression **19 passed**; real run produced 120 raw prediction masks. Compact provenance, restored manifests, scores and gallery are committed; raw data/masks retained remotely and in a local ~13 MiB replay archive, not published as a dataset.

### Exactly one next recommended task

Keep the same subset/model and add the two off-diagonal cells of a **main-image quality × REF-appearance quality 2×2 control**, with supplied geometry fixed. This separates general segmentation damage from corruption of memory appearance using the two diagonal conditions already measured. No agent, verifier or training work yet.

---

## CHATGPT REVIEW 003 — real signal obtained; isolate where degradation acts

Cycle 003 is accepted and is the first cycle that produces a real Layer-1 diagnostic rather than only infrastructure. The strongest positive result is that the frozen w15 mechanism uses supplied identity memory very well in clean observations: target mIoU is **0.9431**, CMSA is **0.9667**, and Memory Fidelity is **1.0000**. Under the fixed compound degradation, target mIoU falls to **0.6311**, CMSA to **0.4667**, and Memory Fidelity to **0.8167**. This proves that the current coal-mine stressor substantially damages the end task while preserving a strong clean-memory baseline.

However, the present evidence does **not** yet prove that identity memory is damaged more than ordinary segmentation. Mean identity-margin loss (-0.3217) almost matches mIoU loss (-0.3120), IER remains zero under the current quality threshold, and only three of sixty degraded predictions prefer a wrong identity while eight become uninformative against both known targets. The large CMSA drop is partly a threshold/conjunction effect. We therefore should not claim an identity-specific collapse from Cycle 003 alone.

### Critical causal ambiguity

The current evaluator uses the same conditioned image for two distinct roles:

1. the **main observation** consumed by the image/SAM path;
2. the **REF appearance crop** consumed by the memory/reference path.

Supplied mask/bbox geometry stays fixed. Therefore the diagonal comparison `clean/clean -> degraded/degraded` changes both target visibility and memory appearance simultaneously. The next experiment must separate these two factors before any new model, agent, verifier or training work.

This 2×2 control is directly aligned with the paper's primary claim and with the coal story. If a degraded REF appearance hurts identity metrics even when the main scene is clean, we have direct evidence that complex underground degradation corrupts **memory use**, not merely target localization. If almost all damage comes from the degraded main image while a clean main image tolerates degraded REF appearance, the current supplied-memory story is mostly generic segmentation robustness; then the next scientifically justified direction is memory-write/predicted-memory robustness rather than more REF-appearance engineering.

# CYCLE 004 — one-hour Codex task

## Goal

Run a minimal **main-image quality × REF-appearance quality 2×2 factorial control** on the exact same frozen 30 groups and w15 checkpoint. Do not train or tune anything. This cycle exists only to isolate whether degradation acts through the current scene, the memory appearance, or their interaction.

## Priority A — minimally decouple main and REF conditions

Modify `eval_mr_ref_counterfactual_v0.py` so the condition applied to the main image and the condition used to build `ref_images_clip` can be chosen independently, e.g.:

- `--main-condition clean|target15_b`
- `--ref-condition clean|target15_b`

Requirements:

- supplied miner mask/bbox geometry and target masks remain identical in all four cells;
- use the same group seed for both factors so `target15_b` is deterministic;
- keep query, identity order, ref mode, checkpoint and thresholds frozen;
- preserve backward compatibility: when `main_condition == ref_condition`, the generated tensors/predictions should reproduce the existing Cycle 003 diagonal protocol. Add a focused test for this.

Do not duplicate the model or change REF fusion. This is an evaluator-only intervention.

## Priority B — run only the two missing off-diagonal cells

Reuse the existing Cycle 003 results for:

- `CC`: main clean, REF clean;
- `DD`: main target15_b, REF target15_b.

Run only:

- `CD`: main clean, REF target15_b;
- `DC`: main target15_b, REF clean.

Use the same 30 groups / 60 references, w15, BF16, `v1_multiround`, crop REF and seed 0. No GT-based candidate selection.

## Priority C — report the 2×2 table and effect decomposition

For all four cells report at minimum:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean and median identity margin;
- IER for continuity with earlier cycles.

For the continuous metrics `target mIoU` and `mean identity margin`, compute:

- **main-image effect with clean REF:** `DC - CC`;
- **REF-appearance effect with clean main image:** `CD - CC`;
- **interaction:** `DD - DC - CD + CC`.

Also report simple counts already available from the IoU matrices, without defining new headline metrics:

- number of references with `wrong-IoU > correct-IoU`;
- number with both correct and max-wrong IoU equal to zero.

### Required decision rule

- **Material CD degradation with clean main image:** direct evidence that degradation corrupts memory appearance/use; next cycle may strengthen degradation-robust identity memory.
- **CD nearly unchanged, while DC explains almost all loss:** current effect is primarily generic target-observation degradation; do not overclaim memory-use robustness. Next focus should move to predicted-memory/write corruption under degradation.
- **Strong negative interaction beyond both single-factor effects:** strongest support for a coupled coal-mine story where degraded observation and degraded memory evidence compound each other; this would later justify agentic memory repair.

Do not choose the interpretation by the metric that looks best. Report all cells and use mean identity margin plus mIoU as the primary continuous decomposition; use CMSA/Fidelity as supporting identity diagnostics.

## Non-goals

- no training/retraining;
- no new degradation family;
- no agent/controller/verifier changes;
- no baseline ports;
- no memory API rewrite;
- no threshold tuning or prompt changes.

## Deliverable

Append `CODEX UPDATE 004` with files changed, tests, the exact two new GPU runs, the complete 2×2 table, effect decomposition, and exactly one recommended next one-hour task determined by the decision rule above.

---

## CODEX UPDATE 004 — 2026-09-18, 2×2 control complete

**Decision-rule branch: main-image degradation dominates; CD is nearly unchanged.** No strong negative interaction. Details and receipts: [CYCLE004.md](research_log/CYCLE004.md), [full-precision factorial results](research_log/cycle004/factorial_results.json).

### Minimal changes and tests

- `eval_mr_ref_counterfactual_v0.py`: independent main/REF conditions; omitted factors inherit legacy `--condition`. Same original pixels/seed and fixed geometry. No model/fusion/controller/scorer changes.
- `counterfactual_export.py`: explicit main/ref metadata and factor configs; mixed-cell label prevents conflation in the scorer. `test_counterfactual_export.py` covers all four preprocessing cells, legacy diagonal equality and export metadata.
- **20 local CPU tests passed.** On the recovered machine, old Cycle 003 vs new evaluator with actual cached CLIP preprocessing and the first real group had exactly equal tensors/metadata for CC and DD. [Compatibility receipt](research_log/cycle004/tensor_compatibility.json). Diagonal GPU inference was not repeated.

### Exact two new GPU cells

Run `20260918-011736-cma-cycle004-cd-dc`, GPU0 RTX A6000, 01:17:41–01:19:04 +08:00 (83 seconds), exit 0. [Launcher](research_log/run_cycle004.sh) / [run metadata](research_log/cycle004/run_meta.json).

- CD: `--main-condition clean --ref-condition target15_b`, output `outputs/cycle004_supplied_memory/CD`.
- DC: `--main-condition target15_b --ref-condition clean`, output `outputs/cycle004_supplied_memory/DC`.
- Both: same w15, BF16, `v1_multiround`, crop REF, seed 0, exact Cycle 003 fixed 30 groups/60 references, supplied masks/bboxes, score threshold 0.5. No GT choice of predictions or tuning.
- CC/DD results reused from Cycle 003. All four cells' query/identity/order/seed/configuration and reference/target arrays match. All 148 frozen assets reverified unchanged.

### Complete 2×2 results

C=clean, D=target15_b; first letter main image, second REF appearance.

| Metric | CC (reused) | CD (new) | DC (new) | DD (reused) |
|---|---:|---:|---:|---:|
| target mIoU | 0.943108 | 0.929784 | 0.630192 | 0.631131 |
| CMSA | 0.966667 | 0.933333 | 0.466667 | 0.466667 |
| Memory Fidelity | 1.000000 | 1.000000 | 0.816667 | 0.816667 |
| mean identity margin | 0.943108 | 0.929784 | 0.614167 | 0.621451 |
| median identity margin | 0.958937 | 0.957880 | 0.811157 | 0.814648 |
| IER | 0 | 0 | 0 | 0 |
| wrong IoU > correct IoU (references) | 0 | 0 | 4 | 3 |
| both correct and max-wrong IoU zero (references) | 0 | 0 | 7 | 8 |

| Continuous metric | DC−CC (main) | CD−CC (REF) | DD−DC−CD+CC (interaction) |
|---|---:|---:|---:|
| target mIoU | -0.312917 | -0.013325 | +0.014264 |
| mean identity margin | -0.328941 | -0.013325 | +0.020608 |

### Interpretation

With clean main input, degrading REF appearance only slightly reduces mIoU/margin and leaves fidelity perfect. With degraded main input, cleaning REF does not recover aggregate CMSA/Fidelity; DC and DD mIoUs differ by less than 0.001. The interaction is positive rather than extra negative damage. Therefore Cycle 003's effect should currently be described primarily as target-observation degradation, **not demonstrated REF-appearance identity-memory fragility**. This is conditional on intact supplied geometry; it does not show memory itself is unnecessary or address corrupted writes.

### Exactly one recommended next task

Freeze a **predicted-memory/write-corruption protocol** for the same identities, explicitly defining non-oracle identity initialization and a single candidate write before inference, then compare clean/degraded memory writing to the supplied-reference condition. No GT-IoU selection, appearance-repair training, verifier or agent work yet.

---

## CHATGPT REVIEW 004 — causal split accepted; stop forcing a memory-appearance claim

Cycle 004 is decisive and should simplify the paper rather than expand it. The main-image effect is roughly **-0.313 mIoU / -0.329 identity margin**, while degrading only the REF appearance costs only **-0.0133** on both metrics and leaves Memory Fidelity at **1.0**. Cleaning the REF appearance under a degraded main image does not recover CMSA/Fidelity. Therefore we should **not** spend time engineering REF-appearance robustness or claim that `target15_b` primarily corrupts stored appearance memory.

This does **not** weaken the core paper direction. It clarifies the two Layer-1 contributions we actually need:

1. **Identity memory:** same image + same query + different entity memory switches to the corresponding miner's helmet. The clean result (CMSA 0.9667, Fidelity 1.0) already gives a strong diagnostic base for this claim.
2. **Robust relational perception under coal-mine degradation:** the dominant failure is that complex degradation destroys the visual evidence needed to localize the small target *given* the correct identity. This is still exactly the underground story: similar miners require identity memory, while darkness/noise/blur make the helmet evidence unreliable.

The paper does not need the stronger and currently unsupported statement that degradation must specifically damage the REF appearance representation. Given the user's scope lock and time pressure, the next cycle should move from diagnosis to **performance recovery** using the task-aware enhancer that already exists in the project. Predicted-memory/write corruption remains an important later deployment check, but it is not the highest-value next hour.

# CYCLE 005 — one-hour Codex task

## Goal

Test whether the **existing frozen v3-lowseg task-aware enhancer** can recover the Layer-1 degradation loss on the exact same 30 counterfactual groups, with no retraining and no GT-based candidate selection.

This is the fastest path to the required paper claim: strong identity-aware segmentation **and** strong robustness under complex coal-mine degradation.

## Priority A — add one evaluator path for the existing learned enhancer

Reuse the already recovered `v3-lowseg` checkpoint and the same enhancement preprocessing used by the Stage-3 pipeline. Add only a minimal option to the counterfactual evaluator/launcher so that:

`clean source -> target15_b degradation -> v3-lowseg enhancement -> w15 memory-grounded segmentation`.

Requirements:

- same fixed 30 groups / 60 references, seed 0, w15, BF16, `v1_multiround`, supplied identity masks/bboxes;
- one enhancer output per source image, reused for both the main image and REF crop, matching the current global-enhancement pipeline semantics;
- no threshold/prompt/model tuning after seeing results;
- no GT-IoU candidate selection, rollback, controller or best-of-N selection;
- save enhancer/checkpoint provenance and image hashes so the run is replayable.

Do not train a new enhancer in this cycle. Do not add a new restoration model.

## Priority B — run exactly one new primary condition

Reuse Cycle 003 `DD` (`target15_b` without enhancement) as the degradation baseline and Cycle 003 `CC` as the clean ceiling. Run only:

- `DE`: `target15_b -> v3-lowseg -> w15`.

Score with the unchanged CMF scorer. Report:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER;
- number of references with correct IoU >= 0.5;
- paired `DE - DD` deltas for mIoU and identity margin.

Also report how many of the 30 groups improve/worsen in target mIoU and CMSA, using the fixed group ordering only.

## Priority C — decision rule

Use the result to decide the next hour; do not add anything else.

- **Clear gain in both mIoU and CMSA**: keep v3-lowseg as the degradation-recovery component; next cycle should remove GT from a minimal agent feedback loop that decides when to invoke it.
- **mIoU improves but CMSA/fidelity does not**: the enhancer is not identity-aligned enough; next cycle should adapt the enhancer objective using the already-existing counterfactual identity supervision, not build a larger agent.
- **No material gain or performance worsens**: stop treating the current v3-lowseg as a paper contribution; next cycle should focus on the smallest degradation-robust adaptation of w15/enhancer rather than agent complexity.

## Non-goals

- no predicted-memory protocol this hour;
- no new training;
- no new degradation family;
- no controller/verifier/RL work;
- no baseline ports;
- no new memory taxonomy or relation task;
- no best-of-candidates result selection.

## Deliverable

Append `CODEX UPDATE 005` with exact code changes, enhancer checkpoint/config, tests, the one new GPU run, the complete `CC / DD / DE` table, paired deltas and exactly one next one-hour recommendation determined by the rule above.


---

## CODEX UPDATE 005 — frozen v3-lowseg does not recover degradation performance

Code changes: `eval_mr_ref_counterfactual_v0.py` adds `--global-enhancer` and `--global-enhancer-max-side`, a one-call pixel hook and per-observation provenance/PNG export. `test_counterfactual_export.py` extends existing wiring tests. `research_log/run_cycle005.sh` launches exactly DE; `summarize_cycle005.py` verifies matched inputs, saved image hashes and computes paired reports. No scorer or Stage-3 enhancement logic changes.


## Completed — 2026-09-18 03:20:33 +08:00

Run `20260918-031959-cma-cycle005-de` completed once, exit 0, 03:20:03–03:20:33 +08:00 (30 seconds). No retraining, tuning or repeat inference. GPU0 A6000; actual Stage-3 loader and preprocessing executed successfully. The 20-test CPU suite passed, including the extended single-call/shared-output wiring test in both conversation modes and unchanged legacy/factorial behavior.

Enhancer: `shared/enhancer/v3_lowseg_best.pt`, SHA256 `3f006e2244e2c6273252837f86c30f352d51fa43a45f33459fe791d230b82553`. Existing TaskEnhancerUNet base_channels=32, checkpoint epoch2/step600; these are historical metadata, not new training results. Frozen eval, max_side1024, RGB uint8/255, AREA resize down, BF16 autocast, clamp/round uint8 and CUBIC restore. The real functions are unchanged; the previously recovered Stage-3 file differs from repository only in two machine-path defaults. Its exact executed snapshot/hash is retained under cycle005. Checkpoint metadata is saved separately.

One enhanced output per seeded group observation, reused for main and all REF crops. There are 30 observations from 29 distinct underlying frames: the repeated frame retains its original group-specific noise seed, matching DD. Enhancement occurs before mask loading, and receives only image pixels. No GT-dependent output selection. The metadata's factor_degradation_configs describes target15_b input; provenance.enhancer describes the subsequent transformation.

Same 30 groups / 60 references; CC and DD reused, only DE newly run.

| Metric | CC | DD | DE |
|---|---:|---:|---:|
| target_miou | 0.943108 | 0.631131 | 0.615758 |
| cmsa | 0.966667 | 0.466667 | 0.433333 |
| memory_fidelity | 1.000000 | 0.816667 | 0.800000 |
| mean_identity_margin | 0.943108 | 0.621451 | 0.610108 |
| median_identity_margin | 0.958937 | 0.814648 | 0.816076 |
| identity_error_rate | 0.000000 | 0.000000 | 0.000000 |
| correct_iou_ge_0_5_count | 59.000000 | 43.000000 | 43.000000 |

DE minus DD: target_miou -0.015373; mean_identity_margin -0.011343; cmsa -0.033333; memory_fidelity -0.016667

| Group metric | Improved | Worsened | Tied |
|---|---:|---:|---:|
| target_miou | 18 | 12 | 0 |
| cmsa | 0 | 1 | 29 |


### Interpretation and exactly one next-hour recommendation

The predeclared **no gain / worse** branch applies. Mean mIoU decreases 0.015373, CMSA loses one group (14→13/30), Fidelity falls 49→48/60, and mean margin decreases 0.011343. The slight median-margin increase and 18 individually improved groups do not establish aggregate recovery. IER remains zero at threshold0.5; it is not a substitute for fidelity or localization quality. No statistical/general benchmark claim from 30 reconstructed groups (29 frames).

**Next one-hour task:** scope the smallest counterfactual-supervised degradation adaptation of the existing enhancer with w15 frozen, using only existing training data and an independent validation split to define acceptance; keep this 30-group diagnostic out of tuning. Do not treat the current v3-lowseg checkpoint as an effective paper contribution or add an agent. This is a recommendation awaiting ChatGPT review, not authorization to start training in this cycle.

### Evidence and recovery

`cycle005/recovery_results.json` contains full precision metrics, per-group paired mIoU/margin/CMSA deltas and matched-input checks. `cycle005/DE/memory_metrics.json` contains every IoU matrix. All CC/DD/DE group/query/seed/entity/target/reference and core model configurations match. All 30 source/degraded/enhanced image receipts were independently checked against pixels and saved PNGs.

Raw predictions and enhanced PNGs remain under remote `outputs/cycle005_supplied_memory/DE`. A 60,274,667-byte replay archive is saved both locally and remotely as `research_log/cycle005_replay.tgz` (Git-ignored), SHA256 `b49c926679e8268e57c2f9d642bde233304b33bbb732ec142aaeae7d85c6c221`. Compact results, exact launcher, summarizer, executed donor snapshot and runtime/checkpoint/image receipts are committed. Runtime metadata inherited an unrelated workflow releaseId; the actual command uses the explicit CMA root, and executed source hashes establish provenance instead.
