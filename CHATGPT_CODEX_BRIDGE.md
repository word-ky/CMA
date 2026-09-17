# ChatGPT ↔ Codex Research Bridge

Repository: `word-ky/CMA`

Purpose: use ChatGPT for research/problem formulation and Codex for implementation/experiments. This file is the shared handoff state. Do not erase earlier entries; append updates under the current cycle.

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
