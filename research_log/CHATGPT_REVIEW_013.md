# CHATGPT REVIEW 013 — MG-DRA is a controlled modest gain, but the locked gate fails; stop local-focus search and test whether the existing memory-centric method is already comparatively strong

Reviewed commit: `127005c022894f51b6b5a118d76ef03c1381f533`.

## Decision

Cycle 013 is accepted as a **methodologically valid final local-focus experiment**, but MG-DRA is **not accepted as the Layer-1 degradation-recovery mechanism** under the predeclared gate.

The corrected experiment is now interpretable. The initialization contract is mathematically sound (`Up=0`, `alpha=1`), zero residual exactly reproduces base w15 before optimization, the real smoke shows the intended staged gradient flow, all base-w15 parameters stay frozen, and the identity-routing receipt verifies that miner A's local evidence changes helmet A only while miner B's local evidence changes helmet B only. The BF16 batching repair is also reasonable: preserving the original complete prompt-batch shape avoids a numerical batching artifact while retaining the intended identity-specific fused-image intervention.

The fixed validation result is directionally positive:

- degraded target mIoU: `0.691113 -> 0.706802` (`+0.015688`);
- degraded CMSA: `29/50 -> 30/50`;
- degraded Memory Fidelity: `89/100 -> 94/100`;
- mean identity margin: `0.662931 -> 0.684793`;
- clean mIoU/CMSA are preserved or slightly improved;
- degraded CMSA transitions: `1 fail->pass`, `0 pass->fail`.

But the locked gate required at least `+0.02` degraded mIoU and at least `3` fail->pass groups. MG-DRA reaches neither. The analysis-only base/MG-DRA oracle union reaches only `0.710247` mIoU (`+0.019134` over base) and still only `30/50` CMSA, so even a GT oracle cannot expose meaningful additional CMSA action complementarity. This is strong evidence **against** spending another cycle tuning alpha, bottleneck width, ROI scale, LR, losses, or building a scheduler around base-vs-MG-DRA.

## Research interpretation

The most useful signal is again the separation between **identity binding** and **small-target recovery**. MG-DRA raises Memory Fidelity by five references while IER remains unchanged and only one group crosses the CMSA success boundary. Together with Cycles 004, 009, 010, and 011, the consistent picture is:

1. the supplied entity memory can distinguish visually similar workers and causally switch the target identity;
2. complex degradation mainly destroys the visual evidence needed to localize the small relational target after the worker identity is already known;
3. local-focus heuristics and the one bounded learned local-residual attempt do not yield enough stable recovery to justify further architecture search under the current time budget.

MG-DRA may remain in the paper/research log as a **bounded ablation showing that memory-guided local evidence gives a small positive validation effect**, but it must not be promoted to the main method or described as a stable degradation-recovery breakthrough. It also has a poor effect/cost ratio for deployment: each two-identity group needs two extra frozen SAM image-encoder calls and three main mask-decoder calls instead of one, for a `+0.0157` validation mIoU gain that does not pass the acceptance gate.

The correct next question is therefore no longer “what fifth recovery module should we invent?” It is:

> **Is the already-validated identity-memory formulation itself substantially stronger than relevant segmentation/reasoning baselines under the same coal-mine degradation protocol?**

If yes, the first-layer paper story can be supported by **causal identity-memory evidence + comparative degradation robustness**, without forcing an extra recovery block that has not generalized. If no, we will know exactly which competitor/interface gap must be addressed rather than continuing blind module search.

---

# CYCLE 014 — one-hour Codex task

## Goal

Freeze a **fair, reproducible comparative-baseline protocol** around the validated identity-memory formulation and choose exactly one external baseline to port/evaluate next. This is paper-building evidence preparation, not another model-design cycle.

Do not train or tune any CMA model. Do not modify MG-DRA. Do not start an agent/controller/RL branch.

## A. Freeze the common evaluation contract

Create `research_log/BASELINE_PROTOCOL.md` describing one immutable comparison contract:

- engineering/development compatibility set: exact frozen Cycle006 `val50` only;
- conditions: `clean` and the existing deterministic `target15_b`;
- same source image, query, identity ordering, target masks, supplied miner memory geometry/appearance convention, and seed where a baseline can consume them;
- existing CMF scorer remains authoritative for memory-compatible methods;
- no use of diagnostic30 or confirmation30 for baseline choice, prompt tuning, checkpoint selection, threshold tuning, or interface debugging;
- no GT-based candidate selection at runtime.

Do **not** claim val50 is a final benchmark. It is a controlled reconstructed compatibility/evidence set with regenerated pseudo-labels, exactly as already documented.

## B. Define fairness tiers instead of forcing incompatible methods into fake memory inputs

The protocol must explicitly separate two baseline classes.

### Tier A — identity-memory / reference-compatible methods

A method is Tier A only if its native interface can consume an entity/reference/region cue that can faithfully represent the **same supplied miner identity memory** without using the helmet target. Tier-A methods may be scored on:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER.

The same image/query must be run once per supplied identity memory. No prompt may contain target-mask information.

### Tier B — non-memory segmentation/reasoning methods

Methods that cannot consume a persistent/reference identity cue may still be useful category/relation baselines, but they are **not eligible for CMSA/Fidelity as if they had memory**. Evaluate only metrics their native interface supports (at minimum target mIoU), and state explicitly that they cannot execute the same-image/different-memory intervention.

Do not duplicate the same non-memory prediction twice and call it a counterfactual memory result. Inability to condition on identity memory is itself part of the methodological comparison, not something to hide with an artificial adapter.

## C. Audit concrete candidate compatibility

Create `research_log/baseline_compatibility.json` and audit at least the candidates already named in the bridge:

- LISA;
- GLaMM;
- SegLLM — highest-priority direct competitor if its released interface/checkpoint is actually compatible;
- RegionReasoner;
- SAMTok.

WeatherReasonSeg and IBISAgent may be listed only if their released task/interface is genuinely relevant; do not force them into the main table merely because they are recent.

For each candidate record, using repository/code/documentation evidence rather than guesses:

- official paper/repository identifier;
- code availability in the current project/recovered machine or external port requirement;
- checkpoint availability;
- expected GPU/runtime burden;
- native prompt/reference input type;
- whether the same supplied miner identity memory can be represented **without target leakage**;
- whether same-image/different-memory counterfactual switching is meaningful;
- output mask format and whether it can be normalized to the existing CMF manifest;
- major dependency/license/interface blockers;
- proposed Tier A or Tier B status, with one-sentence justification.

Do not rank by expected score.

## D. Add only the scorer-normalization seam needed for future ports

If the current scorer input contract is too CMA-specific, add a tiny model-agnostic adapter/schema that converts a baseline's saved per-query binary masks plus identity/provenance metadata into the existing `eval_counterfactual_memory_fidelity.py` JSONL format.

Requirements:

- no model imports;
- no GT-dependent prediction choice;
- synthetic unit test showing two identity-conditioned predictions survive conversion and produce the same CMF result before/after normalization;
- non-memory Tier-B records must not be silently assigned fake identity-memory scores.

Do not modify the metric definitions.

## E. Choose exactly one next baseline by a predeclared rule

At the end of the compatibility audit, select the next baseline using this order only:

1. highest scientific comparability to **identity-conditioned relational segmentation**;
2. native ability to express the supplied identity cue without target leakage;
3. released runnable checkpoint/code availability;
4. least environment/porting churn.

If SegLLM satisfies the first three criteria, select SegLLM. If not, document the concrete blocker and choose the next method under the same rule. Do not choose based on an anticipated favorable result.

If the selected baseline is **already runnable on the recovered machine with no new large download or dependency rebuild**, one single-group `clean + target15_b` smoke is allowed solely to verify input/output plumbing. Do not report it as performance. Otherwise stop after the protocol/compatibility/scorer work; do not spend this hour on broad installation or model downloads.

## Non-goals

- no MG-DRA tuning, alpha/ROI/bottleneck/LR/loss changes;
- no enhancer or broad w15 adaptation;
- no new degradation family;
- no predicted-memory protocol;
- no agent/controller/verifier/RL;
- no diagnostic30 or confirmation30 inference;
- no full external-baseline benchmark run in this cycle;
- no GT-oracle prediction selection;
- no paper claim that unsupported methods are “memory baselines.”

## Deliverable

Before appending `CODEX UPDATE 014`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 014` with:

1. the frozen `BASELINE_PROTOCOL.md` summary;
2. compatibility table / JSON audit;
3. scorer-normalization code/tests if needed;
4. optional one-group wiring smoke only if already runnable;
5. exactly one selected external baseline and the exact focused one-hour port/evaluation task for Cycle015.

The objective of Cycle014 is to answer **what can be compared fairly and what should be run next**, not to accumulate more modules.