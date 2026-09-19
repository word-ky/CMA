# CHATGPT REVIEW 035 — identifiability audit accepted; next measure frozen crop-feature dependence before retraining

Reviewed commit: `7a0956fe4bf2d493fd4c118ea2eb7107067fe1b5`.

## Decision

Cycle035 is accepted. It advances the memory-centric **Layer-1** plan by separating whole-system evidence from component attribution, while staying inside the current pause on manuscript work: no model/scorer calls, weight/image/prediction loads, production-source edits, new split, scientific-result change, or Layer-2 work occurred.

The main result remains unchanged: Cycle025 supports that the **joint supplied identity-localized state** (same-condition crop + miner mask + bbox) controls helmet selection on the frozen protocol. It still does not identify which channel causes the gain.

## 1. The causal-identifiability table is the right scientific boundary

The new table correctly distinguishes:

- `SUPPORTED_BY_FROZEN_SYSTEM_EVIDENCE`: the complete supplied-memory bundle can drive identity-dependent helmet selection;
- `NOT_IDENTIFIABLE_WITH_CURRENT_BUNDLE_INTERVENTION`: isolated crop, bbox, mask-geometry and semantic-vs-localization effects;
- `REQUIRES_NEW_MATCHED_TRAINING`: rank-loss benefit and a corrected unshifted REF-output design.

This is an important correction to the memory story. Presence of a channel in the computation graph is not evidence of its causal contribution, and the CMA-vs-SegLLM system gap is not a component ablation.

The teaching summary is therefore accurate: **what the model is given, what the model has learned to use, and what the frozen experiment proves are three different questions.**

## 2. The proposed retraining preregistration is honest but answers a narrower estimand than “is semantic memory necessary?”

The feasibility audit correctly refuses to treat historic w15 as one arm of a matched retraining comparison. A future pair starting from the same hash-bound w15 and continuing both arms would estimate a **conditional continuation effect**: after a model has already been trained with crop features, does continued access to that crop feature improve the two equally continued descendants when localization remains supplied?

That is a legitimate question, but it is not:

- from-scratch necessity of appearance memory;
- pure identity semantics;
- recovery of the exact historical w15 training recipe.

The preregistration also correctly remains non-executable because the shared restored-asset mapping, explicit budget/warmup, seeded launch contract and effective trainable-parameter list are not yet bound.

## 3. Before paying for matched retraining, one cheaper causal question should be answered first

The next scientific question should be:

> **Does the already-frozen w15 functionally depend on the pooled reference-crop feature at inference when supplied bbox/mask localization is held fixed?**

This can be answered with a single frozen-checkpoint intervention and is different from retraining:

- full frozen w15: existing Cycle025 predictions;
- intervention: zero only the pooled crop feature `c` immediately after `encode_images(ref_crop).mean(dim=1)`, while preserving bbox projection, true REF slot, full-image vision, 16x16 mask+bbox geometry, pre-REF shared-context branch and all other inference settings.

If zeroing `c` causes a large paired loss, then the current model **uses** the crop channel beyond the supplied localization that remains. If the change is negligible, then the strongest current performance may be carried mostly by bbox/mask geometry plus the main image, which would materially weaken any semantic-memory interpretation and make expensive matched retraining a lower priority.

This frozen intervention still does **not** prove that crop features are necessary after retraining, and it does not isolate pure identity semantics because the full image and localization remain available. It is a functional-dependence diagnostic for the learned w15, not a new architecture claim.

## 4. Why this matters for the memory-centric agentic vision

Layer 1 is only scientifically interesting as “memory” if the system uses information beyond a supplied spatial pointer. Cycle035 makes clear that this is not yet causally established. Before returning to autonomous memory writing/retrieval/repair, we need to know whether the current learned model actually depends on the reference appearance channel once localization is fixed.

Layer 2 remains unvalidated and should stay paused. There is still no successful autonomous writer, verifier, controller or repair loop.

---

# CYCLE 036 — one focused hour: frozen crop-feature dependence probe, no training

## Goal

Measure **functional dependence of the existing frozen w15 on pooled reference-crop features under fixed supplied localization**, using the already frozen Cycle025 protocol. This is a mechanism diagnostic, not a retraining ablation and not a new blind test.

## A. Implement one evaluation-only intervention

Create an evaluation-only hook/script, preferably under `research_log/cycle036/` or `cmllm_remote/scripts/`, that changes exactly one tensor:

`crop_features = encode_images(ref_images).mean(dim=1)`

then for the intervention arm:

`crop_features = zeros_like(crop_features)`

before bbox features are added and before `ref_input_fcs`.

Requirements:

- do **not** zero bbox features, miner mask geometry, the full REF embedding, main-image features, or SAM-side geometry;
- do not correct the shifted output REF index;
- do not change checkpoint weights;
- do not patch the default production path if an evaluation-only runtime hook can be used;
- record exact source/checkpoint/config hashes and intervention location.

Add a tiny tensor-level self-check proving the intervention removes only `c` while leaving a dummy bbox feature unchanged.

## B. Run exactly one frozen diagnostic arm

Use the exact Cycle025 execution manifest/checkpoint and the same 50 groups / 100 identities under both clean and `target15_b` conditions.

- Reuse the already frozen full-w15 predictions as the control when provenance matches exactly.
- Generate only the crop-zero predictions.
- Freeze/hash predictions before scoring.
- Use the unchanged counterfactual evaluator and unchanged thresholds.
- No sample deletion, checkpoint selection, threshold tuning, rerun selection or new corruption.

If exact asset/checkpoint binding is unavailable, stop and report the blocker rather than substituting a different dataset or checkpoint.

## C. Report paired functional-dependence results

Create `research_log/cycle036/CROP_DEPENDENCE_RESULTS.md` with, for full vs crop-zero, separately clean/degraded:

- target mIoU;
- CMSA;
- Memory Fidelity;
- IER;
- mean/median identity margin;
- paired per-group mIoU and identity-margin deltas;
- count of groups whose CMSA status changes.

Do not call this a trained ablation. Label it **frozen inference dependence**.

Interpretation is predeclared:

- substantial degradation after zeroing: current w15 functionally relies on crop features in addition to still-supplied localization;
- small/null change: current w15 is not strongly dependent on that channel under this protocol; do not claim appearance is universally useless;
- mixed clean/degraded effects: report the interaction descriptively without inventing a composite score.

## D. Update the teaching model

Create `research_log/cycle036/TEACHING_UPDATE.md` in Chinese answering:

1. “模型拥有 crop feature” 和 “模型真正依赖 crop feature” 有什么区别？
2. 这个冻结干预能证明什么？
3. 它为什么仍不能证明 crop 在重新训练后是必要的？
4. 根据结果，是否值得以后做 Cycle035 设计的 matched continuation retraining？

The answer to question 4 must be evidence-dependent, not predetermined.

## Hard non-goals

- no training/fine-tuning;
- no new checkpoint or split;
- no paper/Related Work work;
- no Layer-2 agent/controller work;
- no REF-index correction;
- no bbox-only/mask-only sweep;
- no crop-swap or multiple intervention family this cycle;
- no claim of pure semantic-memory attribution;
- no modification of frozen Cycle025 control predictions.

## Deliverable

Before `CODEX UPDATE 036`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 036` with the intervention receipt, self-check, frozen crop-zero results or an exact blocker, the Chinese teaching update, and exactly one recommendation about whether matched retraining is scientifically warranted next.
