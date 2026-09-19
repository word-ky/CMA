# CHATGPT REVIEW 038 — crop dependence is real; reopen Layer-2 only as memory-action control

Cycle038 is accepted as a meaningful Layer-1 mechanism result. The unchanged frozen w15 crop-zero intervention finally isolates a component-level dependency that the earlier bundle-level counterfactual tests could not identify.

## What is now supported

With main image, miner mask/bbox localization, checkpoint, split and scorer fixed, zeroing only the pooled reference-crop feature causes a very large collapse:

- clean: mIoU 88.28% -> 39.99%, CMSA 92% -> 4%, Fidelity 97% -> 48%, IER 1% -> 39%;
- target15_b: mIoU 67.53% -> 34.28%, CMSA 58% -> 0%, Fidelity 84% -> 47%, IER 3% -> 37%;
- zero-minus-full mIoU: -48.29 pp clean and -33.25 pp degraded;
- CMSA groups lost/gained: 44/0 clean and 29/0 degraded.

This is strong evidence that the already-trained frozen w15 **functionally depends on the crop-memory channel beyond the retained supplied mask/bbox geometry**. The model is therefore not merely using the miner localization branch while ignoring the crop feature.

This materially strengthens the memory-centric model story and makes a system-level agent that acts on memory scientifically better motivated than before.

## Important causal boundary

Do not overclaim this as pure appearance semantics or from-scratch necessity. The masked crop can carry texture, color, silhouette and residual spatial structure; replacing a trained feature by zero is also an out-of-distribution intervention. Cycle038 therefore establishes functional use of the visual crop-memory channel, not the isolated causal gain of semantic identity appearance, and not what a model retrained without crop would do.

The Cycle035 matched-continuation experiment is now lower priority. It would answer a narrower conditional retraining question, whereas the current project direction needs a distinct system-level contribution.

## Why the agentic direction should now change

The earlier Layer-2 verifier/scheduler line failed because it tried to use an oracle-free output score as a general segmentation-confidence gate. The calibrated MCR signal was informative for identity ranking but was not reliable enough as a stand-alone accept/abstain verifier. Do **not** revive that thresholding pipeline, Qwen policy, RL, or another generic output verifier.

The new system-level contribution should instead make the **memory state itself the object of action**.

The existing enhancement and rollback logic is useful, but its semantics must be changed:

- enhancement/focus/re-observation should create a **candidate memory version** rather than blindly replacing the current image/state;
- miner re-segmentation should be interpreted as a **geometry refresh** of entity memory;
- REF segmentation is a **memory read/use** action;
- rollback should restore the previous **memory version** (appearance + geometry + provenance), not merely restore a previous image or target mask;
- stop means the current memory version is sufficiently reliable for the requested relational perception.

This yields the desired loop:

`observe -> read memory -> act on evidence -> write candidate memory -> use CMA -> assess -> commit or rollback memory -> stop`

That is qualitatively different from the retired Layer-2 verifier. It is a memory-state transition system.

## Critical engineering caution about enhancement

Do not assume the old enhancer is automatically a useful agent action. Earlier whole-image enhancement experiments had little or negative standalone gain, so an agent cannot create performance from an action with no headroom. The correct next step is therefore an **action-headroom audit plus memory-state wiring**, not immediately training a new controller.

# CYCLE 039 — one-hour Codex task

## Goal

Determine whether the existing Stage-3 actions provide enough complementary upside to justify a memory-centric agent, and convert their semantics from image-state operations into explicit entity-memory state transitions. This cycle is **no training and no learned policy**.

## A. Build an agentic-memory action map

Create `research_log/cycle039/AGENTIC_MEMORY_ACTION_MAP.md` mapping the existing actions to memory semantics, using exact source anchors from `stage3_rule_controller_v3_seg_local_enhance.py`, `stage3_policy_v2_rollout.py`, and `entity_memory.py`.

At minimum map:

- direct/ref segmentation -> READ/USE_MEMORY;
- global or local enhancement -> REOBSERVE / REPAIR_EVIDENCE;
- miner segmentation -> REFRESH_GEOMETRY;
- background suppression / local focus -> TARGETED_REOBSERVATION;
- candidate anchor/ref update -> WRITE_CANDIDATE_MEMORY;
- rollback -> ROLLBACK_MEMORY;
- final accept/stop -> COMMIT_MEMORY / STOP.

For each action, state exactly which old decision fields depend on GT IoU and therefore cannot survive into an agent policy.

## B. Add a minimal memory-action state adapter

Add one small module, preferably `cmllm_remote/scripts/memory_action_state.py`, wrapping the existing `EntityMemoryStore` rather than rewriting Stage-3.

Required operations:

1. initialize/read current memory version;
2. create a candidate version with changed image/appearance and/or geometry plus provenance;
3. commit candidate;
4. rollback to a previous committed memory version;
5. expose an audit trace: entity id, version, source action, image state, changed channels, committed/rolled-back status.

No GT quality field may be accepted by this API. Unit-test candidate -> commit and candidate -> rollback behavior.

## C. Quantify existing action headroom from already saved outputs

Use only already archived predictions/traces from the earlier Stage-3/enhancement cycles when possible. Do not generate a new action sweep.

For each recoverable action family, report offline diagnostic oracle ceiling only:

- direct/base;
- global enhancement;
- local target enhancement;
- miner/anchor refresh;
- ref/focus path;
- rollback/final selected state.

Report how often each action improves/worsens target mIoU and, where paired identity predictions exist, CMSA/Fidelity. The oracle ceiling is **diagnostic only** and must never be used as the policy.

Predeclare an actionability gate for continuing to an actual agent pilot: at least one non-base action family must show either >= +2.0 pp oracle-ceiling target mIoU over the frozen base on degraded cases or recover >= 5 additional CMSA groups on a 50-group paired set. If available archives cannot support the metric, report that honestly and do not fabricate it.

## D. Recommendation for Cycle040

Select exactly one candidate recovery action only if the headroom gate passes. Explain how that action would create a candidate memory version and how an oracle-free **relative** signal could compare candidate vs current memory without reusing the failed MCR absolute threshold as a general verifier.

If no action passes the gate, do not build a controller. State that the current action set lacks enough recovery headroom and recommend improving the recovery action itself before policy learning.

## Hard non-goals

- no matched-continuation retraining;
- no new enhancer training;
- no Qwen/RL/DPO/GRPO;
- no new verifier threshold sweep;
- no GT-based online accept/rollback;
- no manuscript writing;
- no REF-index hotfix;
- no new dataset/baseline.

## Deliverable

Append `CODEX UPDATE 039` to `CHATGPT_CODEX_BRIDGE.md` with the action map, memory-state API/tests, headroom table, the pass/fail decision on the actionability gate, and exactly one recommended next action. Because this review file was created separately due the connector's no-safe-append limitation on the 500KB bridge, mirror this REVIEW038 verbatim into the canonical bridge immediately before `CODEX UPDATE 039`.
