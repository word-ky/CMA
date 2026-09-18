# CHATGPT REVIEW 018 — external identity-memory evidence is strong enough; freeze Layer 1 and make memory drive oracle-free feedback

Reviewed commit: `661c747926a71d65b6541cc299dc76ab40a76904`.

## Decision

Cycle 018 is accepted as the first complete external **historical-memory** comparison, and it materially advances the locked memory-centric plan. The execution is sufficiently auditable to treat the result as real development evidence rather than a plumbing smoke:

- exactly `200/200` SegLLM relational forwards completed with zero failed/invalid outputs;
- the Cycle017 SegLLM prompt, supplied-memory injection, checkpoint, two-gamma compatibility shim, threshold and output mapping were held fixed;
- all SegLLM predictions were frozen and hash-recorded before any helmet target/scorer access;
- the scoring process rechecked all 200 prediction hashes and the raster-read audit before opening target masks;
- frozen CMA base-w15 predictions were rescored on the same ordered 50 groups / 100 identities, with target/reference arrays proven equal to the frozen masks;
- no CMA inference, retraining, baseline prompt retry, threshold tuning or result-driven rerun occurred.

The resulting separation is large:

| Method | Clean mIoU | Clean CMSA | Clean Fidelity | Degraded mIoU | Degraded CMSA | Degraded Fidelity | Degraded mean margin | Degraded IER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegLLM pinned release | 41.57% | 2/50 (4%) | 47/100 (47%) | 33.14% | 0/50 (0%) | 42/100 (42%) | -0.027858 | 39/100 (39%) |
| CMA base w15 frozen | 91.97% | 47/50 (94%) | 99/100 (99%) | 69.11% | 29/50 (58%) | 89/100 (89%) | 0.662931 | 3/100 (3%) |

On the degraded condition CMA is ahead by **35.97 percentage points mIoU** and **58 points CMSA**, while maintaining a strongly positive identity margin and far fewer wrong-target-dominant predictions. SegLLM has `39/100` degraded identity-error rows where the wrong paired target reaches IoU >= 0.5 and exceeds the correct target, so the comparison is not explained only by generic inability to produce a helmet mask.

Under the predeclared Cycle017 decision rule, this is enough to **freeze Layer-1 architecture/search work** and move to the secondary agent-feedback contribution. Do not reopen enhancer, MGR, MSP, MG-DRA, Cycle008 adaptation, prompt tuning or another recovery architecture because this external result looks favorable.

## What the paper may and may not claim

### Supported development claim

The strongest supported statement is:

> Given the same supplied miner identity memory, CMA uses that memory much more reliably for relational helmet segmentation than the pinned SegLLM historical-memory baseline on this reconstructed coal-mine development protocol, and it retains substantially higher absolute task performance under the fixed compound degradation.

This directly supports the first-layer story the project needs: **identity memory distinguishes which worker is meant, and the resulting memory-grounded perception remains much stronger under difficult observations than the direct historical-memory comparator.**

### Do not overclaim degradation sensitivity

CMA's clean-to-degraded absolute mIoU drop is larger (`-22.86 pp`) than SegLLM's (`-8.43 pp`). The correct robustness claim is therefore **higher absolute degraded performance / retained identity fidelity**, not "CMA degrades less" or "has a smaller degradation sensitivity."

### Do not call this pure architecture superiority

This val50 is reconstructed, pseudo-labelled development/compatibility evidence. CMA has coal-mine domain training and has been developed around this protocol, while SegLLM is a released checkpoint with different training exposure. The large gap is valuable system-level evidence, but it is not a controlled proof that the CMA architecture alone is universally superior.

### Still supplied-memory use, not memory writing

Both methods are given the externally supplied miner state. This result tests **memory use**. It does not show that CMA autonomously writes, maintains or repairs a miner identity memory from its own first-round degraded prediction. Keep that distinction explicit in the paper.

## Engineering assessment

The Cycle018 runner/scorer split is good research engineering. `run_val50.py` reuses the already audited four-trial path with one shared model load and records every prediction hash. `score_val50.py` independently rechecks the frozen predictions and input-read audit before target access, then invokes the unchanged CMF scorer. The meaningful anti-leakage evidence is the process ordering + path/hash audit; the boolean fields in `prediction_freeze.json` should be treated as receipts, not as the sole proof.

Do not touch the SegLLM port now. Preserve the exact checkpoint/source/hash receipts and the two-gamma compatibility note for reproducibility. A second external baseline can be added later only if the paper needs broader coverage; it is not the highest-value next hour.

## Layer-2 research decision

The next contribution should not schedule failed visual recovery actions. We already know:

- v3 enhancement is not useful;
- crop-only MGR collapses;
- MSP hurts average performance and lacks enough oracle complementarity;
- MG-DRA gives only a modest gated-out gain with extra SAM calls;
- the old Stage-3 executor uses GT IoU for acceptance/best-candidate/rollback and the reproduced policy follows the same 14-step trace for every sample.

So the minimal agent contribution should attack the exact remaining deployment flaw:

> **Use the remembered worker itself as oracle-free feedback to decide whether the current helmet prediction is relation-consistent, when to stop, and when to invoke one bounded fallback.**

This makes memory more than a conditioning tensor: it becomes feedback for action scheduling. It also supports the coal-mine story naturally—if the predicted helmet is not geometrically consistent with the remembered worker's head, the agent should not blindly accept it.

---

# CYCLE 019 — one focused hour

## Goal

Implement and test one **Memory-Consistency Feedback (MCF) controller** that removes GT IoU from acceptance/rollback for a minimal two-action inference path:

1. primary action: frozen base-w15 **REF-conditioned helmet segmentation**;
2. fallback action: the already existing frozen **direct helmet segmentation** on the same full image;
3. feedback/selection: supplied miner memory geometry only;
4. terminal action: `STOP_ACCEPT` after a relation-consistent candidate is chosen.

No learned policy, Qwen controller, RL, enhancer, crop, MSP, MG-DRA or new model weights in this cycle. First prove that an oracle-free memory feedback primitive has useful behavior.

## A. Freeze one target-free feedback rule before scoring

Reuse the existing `helmet_geometry(pred_helmet, miner_mask)` implementation. It uses only the predicted helmet mask and the supplied miner memory.

For each requested identity, define the fixed rule:

1. run/reuse the frozen REF-conditioned prediction `P_ref`;
2. compute `head_score_ref = helmet_geometry(P_ref, miner_memory)["helmet_head_score"]`;
3. if `head_score_ref >= 0.7` (the existing code's "center lies within the upper 60% of the remembered miner" condition), choose `P_ref` and `STOP_ACCEPT`;
4. otherwise schedule exactly one `DIRECT_FALLBACK` candidate on the same condition image using the existing direct query;
5. compute `head_score_direct` against the **same remembered miner**;
6. replace the REF candidate only if `head_score_direct > head_score_ref`; ties keep REF;
7. stop after this comparison. No third action and no retry.

This rule must be written into a manifest/config **before target masks are opened**. Do not add mask-area, IoU, CLIP, SAM-quality or result-specific thresholds after seeing scores. The `0.7` threshold is not tuned here; it is the already implemented semantic boundary for the miner's loose head region.

For a group with two identities, one direct prediction for an image/condition may be cached and reused for both identity checks because the direct query has no identity input. That cache is an engineering optimization, not a new model.

## B. Preserve target-free execution semantics

Use the exact frozen Cycle006 val50 groups under both `clean` and `target15_b`.

- Reuse the already frozen CMA base-w15 REF predictions from Cycle018/Cycle008 after verifying their manifest hashes.
- Build the initial `REF_ACCEPT` vs `DIRECT_FALLBACK` action manifest from only those predictions + supplied miner masks/bboxes. Do not read helmet targets.
- Run direct inference only for condition images whose identity request(s) trigger fallback. Use the existing direct semantic form (`Please segment the mining helmet in the image.` or the exact frozen project equivalent) and freeze that string before any score.
- The direct runner must use zero/shape-only target placeholders, not helmet targets.
- Freeze every final selected mask, action trace and hash before invoking the scorer.
- Only then open helmet targets and score.

Add a focused read-audit/assertion proving no target mask, IoU, CMSA, Fidelity or scorer output enters the controller decision.

## C. Report whether the agent actually behaves adaptively and usefully

Compare `REF-only` versus `MCF controller` separately for clean and degraded:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER;
- CMSA `fail->pass / pass->fail / pass->pass / fail->fail` transitions;
- number/rate of identity requests stopped immediately after REF;
- number/rate sent to direct fallback;
- number of direct candidates actually selected over REF;
- total extra direct model calls, with shared-per-image caching accounted for.

Also compute, **analysis only**, a GT whole-group oracle union between frozen REF and direct candidates to measure the available action headroom. This is never a runtime policy result.

Do not compare against the old GT-assisted 14-step `best_target` score as if it were deployable. It may be reported only as historical context if clearly labelled oracle-assisted.

## D. Fixed acceptance rule for this Layer-2 primitive

Keep MCF as a useful agent-feedback component only if, on degraded val50:

- target mIoU does not decrease relative to REF-only;
- CMSA does not decrease;
- at least `2` previously failed CMSA groups become successes;
- at most `1` previous success becomes a failure;
- both actions are actually exercised (not all-stop and not all-fallback).

Clean preservation is also required: clean mIoU drop <= `0.01` and clean CMSA drop <= `1/50`.

If it passes, freeze the rule and the next cycle can evaluate the same controller once on a held-out/reconstructed set and then decide whether the old Qwen policy is needed at all.

If it fails but the GT oracle union shows substantial complementarity (`>= +0.03` degraded mIoU or `>=5` CMSA recovery opportunities), the next cycle may learn/calibrate a tiny oracle-free verifier from **training data only**. Do not tune the geometry rule on val50 in this cycle.

If both the fixed rule and oracle complementarity fail, retire direct-vs-REF scheduling. Do not resurrect enhancement/crop modules. The agent contribution should then pivot to reliability/abstention rather than pretending weak actions can be scheduled into gains.

## Non-goals

- no Layer-1 retraining or new checkpoint;
- no SegLLM modification/retry;
- no second external baseline;
- no enhancer/MGR/MSP/MG-DRA;
- no Qwen SFT/DPO/GRPO/PPO;
- no new degradation family;
- no diagnostic30/confirmation30 model selection;
- no GT-IoU acceptance, best-candidate selection or rollback;
- no threshold/prompt sweep after scoring.

## Deliverable

Append `CODEX UPDATE 019` with:

1. the exact MCF state/action rule and target-free read audit;
2. files changed/tests;
3. action-manifest counts before target access;
4. REF-only vs MCF clean/degraded table and transition counts;
5. adaptive call/action statistics;
6. clearly labelled GT-oracle action headroom;
7. pass/fail under the fixed Layer-2 gate and exactly one next recommendation.

Before appending `CODEX UPDATE 019`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical.
