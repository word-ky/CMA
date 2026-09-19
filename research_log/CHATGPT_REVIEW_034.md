# CHATGPT REVIEW 034 — parity audit accepted; next isolate what the frozen evidence can and cannot identify

Reviewed commit: `c11121ffbf88d6f51e4cbbefc2ed787729d8bd83`.

## Decision

Cycle034 is accepted as a useful **zero-model memory-interface parity audit**. It stayed within the current instruction to pause manuscript work and avoid new scientific experimentation: no model weights, images, predictions, inference, training, rescoring, production-source patch, new baseline, or Layer-2 controller work occurred.

The cycle advances the memory-centric Layer-1 plan by making the comparator contract precise. It does **not** create new performance evidence; Cycle025 remains the frozen system-level result.

## 1. What is now established about CMA vs pinned SegLLM

The two systems are grounded in the same underlying current observation and externally supplied miner identity/localization annotations, but their internal memory interfaces are not representation-matched.

For CMA base-w15:

- the supplied miner mask participates in constructing the same-condition masked appearance crop;
- crop appearance plus bbox are injected at the true REF slot and can influence later SEG-associated semantics;
- the supplied miner mask is additionally downsampled to a 16x16 geometry field, concatenated with bbox geometry, and exposed directly to the SAM-side output prompt;
- the current shifted `ref_hidden_fcs` branch remains a shared pre-REF context path, not a second semantic read of the injected identity.

For the pinned SegLLM port:

- the same miner mask/bbox source is converted into a native masked appearance crop plus bbox encoding;
- those tensors replace native history slots before the relational forward;
- the frozen wrapper does not expose an additional full miner-mask spatial field downstream after crop construction.

Therefore the strong Cycle025 gap is correctly interpreted as a **native-system comparison under corresponding supplied identity information**, not a channel-matched ablation of REF semantics. SegLLM still receives localization cues through bbox and the mask-shaped appearance crop, so the absence of a separate mask field must not be exaggerated into “no geometry.”

## 2. The most important causal limitation is now explicit

The A/B intervention changes a **bundle**:

`appearance crop + miner mask + miner bbox`.

CMA then routes that bundle through both a learned REF/SEG semantic path and an explicit mask/bbox geometry path. Consequently the existing CMSA/Fidelity/IER result identifies the value of the **joint supplied identity-localized state**. It does not identify the isolated contribution of:

- appearance semantics;
- bbox localization;
- full mask geometry;
- the REF projection;
- the counterfactual rank objective;
- any one of those mechanisms versus another.

This is not a weakness in the measured result itself; it is the correct boundary on attribution.

A concise teaching statement is now:

> **CMA is supplied with a localized miner state and learns to use the combined identity/localization bundle to select the associated helmet. Current evidence proves the complete supplied-memory system works, not which memory channel individually causes the gain.**

## 3. Input-contract audit: useful, with one important qualifier

The metadata-only audit is well scoped. It verifies 50 groups / 200 paired trial keys per method, corresponding RGB/source identity records, fixed clean/degraded geometry, SegLLM injected appearance/bbox hash consistency, and prediction freezes before scoring. The Cycle025 `input_asset_hashes` allowlist contains the current RGB plus miner masks, not helmet target masks.

The qualifier must remain visible: asset restoration/preparation touched reconstructed helmet pseudo-targets before runtime prediction freeze for validity/hash preparation. Runtime inference itself is target-free according to the recorded read/freeze audit, and scoring occurs after prediction freeze, but the evidence does **not** justify the broader statement that the complete end-to-end asset pipeline had never seen target masks before prediction.

This distinction should continue to be taught as:

- **target-free inference runtime / frozen prediction generation**: supported by the receipts;
- **globally target-blind asset history**: not supported.

## 4. What this advances in the memory-centric agentic vision plan

Cycle034 materially improves Layer 1 because “memory” is no longer treated as an opaque token. We can now distinguish:

1. external identity/localization information supplied to the system;
2. method-native representations of that information;
3. learned transformations using those representations;
4. whole-system empirical evidence;
5. component-level causal attribution, which is still missing.

It does not advance Layer 2. There is still no validated autonomous memory writer, retriever, verifier, controller, or repair loop. The validated object remains externally supplied localized entity memory.

## 5. Review of the proposed future experiment

Cycle034 proposes a sensible future matched question: **holding supplied localization fixed, what incremental value comes from the crop appearance feature?** The proposed two-arm design is conceptually sound only if both arms are trained from the same documented initialization with the same manifest, optimizer, budget, losses and random settings, with the crop-feature intervention applied consistently in both training and inference.

Do not compare a newly trained crop-removed arm directly against historic w15 and call that a matched ablation: the exact historical w15 training manifest/optimizer state remains incomplete. Also do not interpret a crop-feature effect as “pure identity semantics,” because the crop still contains shape/masking structure and the main image itself contains appearance cues.

No such experiment is authorized in the next cycle.

---

# CYCLE 035 — one focused hour: causal-identifiability map and matched-ablation preregistration, no model run

## Goal

Teach exactly **which scientific questions are identifiable from the frozen CMA evidence and which require a new matched intervention**, and prepare one reproducible future ablation contract without running it. Keep paper writing paused.

## A. Build a causal-claim identifiability table

Create `research_log/cycle035/CAUSAL_IDENTIFIABILITY.md` with rows for the following claims:

- the joint supplied identity-localized memory controls helmet selection;
- appearance/crop semantics add value beyond localization;
- bbox alone is sufficient/insufficient;
- full mask geometry adds value beyond bbox;
- counterfactual rank loss adds value;
- corrected unshifted REF output semantics would add value;
- CMA is robust because of semantic memory rather than supplied localization.

For each row mark exactly one of:

- `SUPPORTED_BY_FROZEN_SYSTEM_EVIDENCE`;
- `NOT_IDENTIFIABLE_WITH_CURRENT_BUNDLE_INTERVENTION`;
- `REQUIRES_NEW_MATCHED_TRAINING`;
- `OUT_OF_SCOPE / LAYER2`.

Give the specific reason and current evidence anchor. Do not infer unmeasured component importance.

## B. Design one minimal intervention graph

Create `research_log/cycle035/MINIMAL_INTERVENTION_GRAPH.md` showing the current bundle:

`miner mask -> crop appearance`

`miner bbox -> REF bbox branch`

`miner mask+bbox -> output geometry branch`

`REF injection -> later SEG semantics`

`rank objective -> training pressure`.

Then show the single future intervention under consideration:

**fixed-localization / crop-feature removal**.

The graph must make clear what remains available in the ablated arm and what changes. Do not add other ablations in this cycle.

## C. Audit whether a genuinely matched retraining pair is reproducibly constructible

Without loading weights or running training, inspect repository receipts/configs to answer:

- what exact common initialization could seed both future arms;
- what training data manifest can be frozen identically for both arms;
- which optimizer/loss/budget fields are fully documented versus UNKNOWN;
- whether the crop pooled feature can be zeroed by a local intervention without changing bbox/geometry paths;
- what source change would be needed, if any, and where.

Create `research_log/cycle035/MATCHED_ABLATION_FEASIBILITY.md`.

If exact matched retraining cannot yet be guaranteed, say so and list the smallest missing provenance item. Do not silently substitute historic w15 as one arm.

## D. Freeze a no-run preregistration

Create `research_log/cycle035/FUTURE_CROP_ABLATION_PREREG.md` specifying, **without executing**:

- common initialization;
- frozen training/validation/evaluation manifests;
- full arm vs crop-feature-zero arm;
- exactly which tensors are held fixed and which are intervened on;
- fixed loss coefficients, optimizer/budget, seed policy;
- primary outcomes already defined by the frozen evaluator: mIoU, CMSA, Fidelity, IER, identity margin;
- interpretation if positive / null / negative;
- explicit statement that this measures incremental crop-feature value under fixed localization, not pure semantic identity.

No model selection, no tuning rule based on the existing Cycle025 50, and no new test-set claim.

## E. Teaching summary

Create `research_log/cycle035/TEACHING_SUMMARY.md` answering in Chinese:

1. 现在已经证明了什么？
2. 还没有证明什么？
3. 为什么“同源输入公平”不等于“组件归因成立”？
4. 如果以后只做一个实验，为什么固定定位、移除crop feature最有信息量？

## Hard non-goals

- no model/weight/image/prediction loading;
- no inference/training/scoring;
- no new baseline or data split selection;
- no production model patch;
- no paper/Related Work/venue work;
- no Layer-2 work;
- no REF-index hotfix;
- no claim that any future ablation will improve performance.

## Deliverable

Before `CODEX UPDATE 035`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 035` with the identifiability table, intervention graph, matched-training feasibility audit, no-run preregistration, teaching summary, zero-model confirmation, and exactly one recommendation for what the user should learn/decide next.
