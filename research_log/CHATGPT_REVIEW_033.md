# CHATGPT REVIEW 033 — runtime/gradient audit accepted; next audit memory-interface parity before attributing the baseline gap

Reviewed commit: `e226556e2e34d3b06349d9cfeed80198fda7318c`.

## Decision

Cycle033 is accepted as a useful **zero-model method-teaching audit**. It obeyed the current user instruction to pause paper writing and performance exploration. No model weights/images were loaded, no forward/training/scoring occurred, no production source or checkpoint changed, and no Layer-2 controller was revived.

The cycle closes the last runtime-layout uncertainty from Review032 and materially sharpens the memory-centric Layer-1 causal story. It does not create new performance evidence, and it should not be treated as an ablation.

## 1. Runtime token-layout uncertainty is now closed

The recovered A6000 environment confirms the actual frozen vision/tokenizer contract rather than relying on model-name convention:

- CLIP image size 224 and patch size 14 -> 256 visual patches;
- the 255 expansion offset is therefore correct for the one-image frozen layout;
- the actual added-token IDs are REF=32001 and SEG=32000;
- a representative group has true expanded REF at 327, output `ref_hidden_fcs` selection at 326, and auxiliary REF reconstruction at 327;
- the later second-round SEG-associated selection is at 357, after the true REF slot at 327;
- all 50 frozen A/B rows have identical real token sequences.

This confirms the earlier causal correction. The output `ref_hidden_fcs` branch does not read the injected A/B REF identity in the current forward pass. The later relational SEG state can read it because REF occurs earlier in the causal sequence. The 255 offset itself is not the bug; the content selection remains one token before REF/SEG by the repository's shifted-mask convention.

Do not hot-fix frozen w15. The measured Cycle025 system is the system that used this exact behavior.

## 2. The supplied-vs-learned separation is now technically clear

Cycle033 correctly distinguishes three levels that had previously been mixed together:

1. **Externally supplied identity state**: same-condition masked appearance crop, miner mask, and miner bbox.
2. **Learned use of that state**: crop+bbox injection at the true REF slot can influence later SEG semantics; mask+bbox are transformed into an explicit geometry context reaching the SAM prompt; trainable projections/decoder parameters learn how to use these signals.
3. **Measured evidence**: Cycle025 validates the whole supplied-memory bundle and whole system, not the isolated causal contribution of appearance, mask geometry, bbox, REF projection, rank loss, or any other individual component.

The trainability audit is appropriately conservative. It recovers the documented source recipe but explicitly leaves the exact historical per-parameter optimizer/requires-grad snapshot UNKNOWN. Likewise, the gradient graph is graph connectivity only, not measured gradient magnitude or component importance.

A useful teaching formulation is therefore:

> **CMA is given an already localized miner state; it learns to use that state to select the miner-associated helmet under the relational query.**

That is stronger and more accurate than saying the current evidence proves autonomous identity recognition or persistent agent memory.

## 3. Critical new implication: the CMA-vs-SegLLM gap is system-level, not a channel-matched memory ablation

After Cycle033 made the exact CMA input bundle explicit, the next scientific risk is comparator input semantics.

The pinned SegLLM port uses the same underlying supplied miner annotation source, but its native memory interface is not the same as CMA's internal interface. `memory_state.py` uses the supplied miner mask to produce a masked appearance crop and uses the supplied bbox to produce a native bbox encoding. `run_smoke.py` then verifies that **appearance + bbox** are injected into SegLLM's native memory replacement path. It does not inject a separate full miner-mask geometry tensor into the relational output path.

CMA, by contrast, uses the miner mask in at least two identity-specific ways: it helps define the appearance crop and it is downsampled into the explicit 16x16 output geometry context, together with bbox, before the SAM prompt. Therefore:

- both systems are grounded in the same underlying supplied identity/localization annotations;
- their method-native representations and direct spatial channels are different;
- the comparison is valid as a **system-level comparator under corresponding supplied identity information**;
- it is **not** a controlled experiment proving that CMA's learned semantic memory mechanism alone explains the +31.08 pp mIoU / +56 pp CMSA gap;
- the current evidence also cannot tell us how much of CMA's switchability comes from explicit mask geometry versus appearance/REF semantics.

This is not a reason to discard the result. It is the correct boundary around what the result proves.

## 4. What Cycle033 advances in the memory-centric agentic vision plan

It advances Layer 1 by replacing vague "memory" language with an auditable information-flow model:

- **what is stored/supplied** is now explicit;
- **where identity enters** is now explicit;
- **what can learn** is separated from what is frozen;
- **which losses can pressure identity distinction** is separated from actual measured contribution.

It does not advance Layer 2. There is still no validated autonomous memory writer/retriever/verifier/controller. The current validated object is externally supplied, localized entity memory.

The right immediate priority is therefore not a new agent module and not another performance run. It is to make sure we understand the **information contract** of CMA and the baseline before we decide what future experiment would genuinely isolate memory reasoning.

---

# CYCLE 034 — one focused hour: memory-interface parity and information-budget audit

## Goal

Teach exactly what information CMA and pinned SegLLM receive from the same miner annotation source, how each method transforms it, and what the current system-level comparison can and cannot attribute. No model calls and no paper writing.

## A. Build a raw-input parity table

Create `research_log/cycle034/MEMORY_INTERFACE_PARITY.md`.

For CMA base-w15 and pinned SegLLM, list side-by-side:

- current RGB image;
- relational text/query;
- supplied miner mask;
- supplied miner bbox;
- masked appearance crop;
- any full-frame/downsampled mask geometry exposed downstream;
- bbox coordinate encoding;
- special-token/history state;
- helmet target availability at inference.

For every row distinguish:

1. **same raw source information**;
2. **derived representation differs**;
3. **one system has an additional direct path**.

Do not call two interfaces "identical" merely because they originate from the same mask/bbox files.

## B. Trace the two native memory paths from source files

Create `research_log/cycle034/CMA_VS_SEGLLM_CAUSAL_MAP.md` with source anchors.

For CMA, trace:

`miner mask+bbox -> masked crop / REF input -> later SEG`

and

`miner mask+bbox -> 16x16 geometry context -> SAM prompt`.

For SegLLM, trace:

`miner mask+bbox -> native masked crop + bbox encoding -> native history replacement -> final relational decode`.

Explicitly state whether the full miner-mask spatial field itself reaches the relational decoder after crop construction. If the answer is uncertain from the vendored wrapper/upstream receipt, mark UNKNOWN and identify the exact source needed; do not infer.

## C. Verify Cycle025/SegLLM inference-time target exclusion and source parity

Without running either model, audit the frozen Cycle025 CMA manifests and pinned SegLLM preparation/runtime receipts to answer:

- Are the same underlying current image and miner identity annotations used for paired A/B trials?
- Does either method read the helmet target before prediction freeze?
- Which miner-derived artifacts are condition-dependent under `target15_b`?
- Which geometry remains fixed across clean/degraded conditions?

Create `research_log/cycle034/INPUT_CONTRACT_RECEIPT.md` with file/hash/source evidence where already available. No rescoring and no new data selection.

## D. Define raw-information parity vs representation parity

Add a concise teaching note `research_log/cycle034/PARITY_TEACHING_NOTE.md` explaining:

- **raw-information parity**: both methods originate from the same externally supplied miner localization/identity annotations;
- **representation parity**: whether the same derived tensors/direct channels are exposed internally;
- **system-level comparison**: compares complete native systems, not isolated components;
- **component attribution**: requires a matched ablation and is not established by Cycle025.

Use one simple thought experiment with three hypothetical future variants only—`appearance-only`, `bbox-only`, `mask-geometry-only`—to explain what each would isolate. Do **not** run them.

## E. End with one decision memo, not an experiment

Create `research_log/cycle034/NEXT_SCIENTIFIC_DECISION.md` answering:

> If the user later chooses to resume scientific development, what is the single most informative matched experiment for separating identity semantics from supplied localization?

The memo should rank no methods and run nothing. Prefer a minimal matched component ablation design using the already frozen protocol, but keep it as a proposal only until the user explicitly asks to resume experiments.

## Hard non-goals

- no model weights loaded;
- no inference/training/scoring;
- no new baseline/checkpoint/data split;
- no production-source patch;
- no paper/Related Work/venue work;
- no Layer-2 agent/controller work;
- no claim that explicit geometry explains the measured gap without an ablation.

## Deliverable

Before `CODEX UPDATE 034`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 034` with the parity table, two causal maps, input-contract receipt, teaching note, and exactly one future scientific decision recommendation. The update must explicitly confirm zero model/scorer calls and zero scientific-result changes.
