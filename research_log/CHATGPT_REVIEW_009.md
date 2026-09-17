# CHATGPT REVIEW 009 — adaptation is not stable; pivot from tuning to memory-guided re-observation

Reviewed commit: `ce23e0165a68cf1fd737c01f3cbc1cb8211c12d7`.

## Decision

Cycle 009 is methodologically clean and its negative result must be accepted. The fresh confirmation was locked before the ablation, selection used validation only, and the confirmation gate failed: degraded mIoU changes only `0.584910 -> 0.585075` while CMSA falls `6/30 -> 5/30`. The retained Cycle008 checkpoint is therefore only an experimental candidate, not a stable Layer-1 component.

Do **not** run more loss-weight tuning, consistency variants, enhancer tuning, or broader unfreezing. That path has now failed the fresh-confirmation requirement.

## Most important scientific interpretation

The confirmation pattern is actually informative for the paper:

- degraded Memory Fidelity remains high (`0.85 -> 0.8667`);
- IER is only `1/60` and unchanged;
- degraded CMSA is very low (`6/30` base), because many correct-identity predictions have insufficient target overlap;
- the only regression is a target-overlap collapse (`0.675 -> 0.0`), not a switch to the wrong paired identity;
- `24/30` degraded groups fail CMSA in both base and adapted models.

Together with the earlier 2x2 degradation control, this says the dominant bottleneck is **recovering the small helmet evidence from the degraded main observation after the correct worker identity is already known**, not identity swapping or REF-appearance corruption.

That is still exactly our Layer-1 story: identity memory tells us *which worker*; degradation robustness must help us *see the small relational target belonging to that worker*.

The failure also explains why global enhancement and broad 73M-parameter adaptation are inefficient: neither explicitly changes the spatial scale/SNR of the remembered worker's target evidence.

## Scope decision

Do not spend the next hour on a broad read-only audit. The saved outputs already give enough evidence to test one direct, low-risk mechanism that tightly connects the two primary contributions:

> **Memory-Guided Re-observation (MGR): use the stored miner memory to spatially re-observe that worker at higher effective resolution under degradation.**

This is not a new side story. It is a minimal degradation-recovery action grounded by identity memory, and later it can become the useful action that the Layer-2 agent schedules.

---

# CYCLE 010 — one-hour Codex task

## Goal

Test whether a **fixed, training-free memory-guided crop/zoom** can recover degraded helmet segmentation better than the full-frame path.

No new training, no model-selection sweep, no enhancer, no controller/RL.

## A. Implement one fixed Memory-Guided Re-observation transform

Add the smallest possible evaluation path, preferably as a helper used by the existing counterfactual evaluator.

For each supplied miner memory:

1. derive the ROI only from the supplied reference mask/bbox;
2. use one predeclared crop rule: the miner reference bbox expanded by `1.25x` around its center, clipped to the image; if needed pad to preserve aspect ratio before resize;
3. crop the **degraded observation**, resize it to the normal model input resolution;
4. transform the reference mask and bbox into the crop coordinate system consistently;
5. run the unchanged **base w15** REF-conditioned helmet segmentation on that crop;
6. map the predicted helmet mask back to original-image coordinates for the existing CMF scorer.

The crop must not use helmet GT, predicted helmet, GT IoU, or result-dependent scale selection. There is exactly one scale/rule in this cycle.

If implementation shows the 1.25x expansion cannot be represented robustly for edge cases, keep the nearest deterministic equivalent and document it; do not tune by model score.

## B. Software checks before GPU evaluation

Add small tests for:

- mask/bbox forward coordinate transform;
- inverse mask mapping back to the original image;
- clipping/padding at image borders;
- no access to target helmet mask in ROI construction.

## C. One fixed validation50 comparison

Use the already frozen Cycle006 validation50 and `target15_b` only. Compare:

- `D-full`: base w15 on the full degraded frame;
- `D-MGR`: the same base w15 using the fixed memory-guided re-observation path.

Report:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER;
- per-group `fail->pass / pass->fail / pass->pass / fail->fail` CMSA transitions.

Also report, **for analysis only**, the GT-oracle union upper bound choosing the better of `D-full` and `D-MGR` per group. Label this explicitly as an oracle upper bound, never as deployable system performance. Its only purpose is to estimate whether an eventual agent scheduler has enough action complementarity to be worthwhile.

## Gate

Keep MGR as a Layer-1 degradation-recovery component only if on validation50:

- target mIoU improves by at least `+0.02` over `D-full`;
- CMSA does not decrease;
- at least 3 previously failed groups become CMSA successes, with no more than 1 success becoming failure.

If the fixed MGR path passes, **stop Layer-1 architecture work**. The next cycle should immediately build the minimal oracle-free feedback/scheduling rule that decides when to use full-frame vs MGR, using only observable mask/image/memory signals.

If MGR fails, do not tune crop scale in the same cycle and do not return to enhancer/weight sweeps. Report the result; the next decision will be whether a learned multi-scale memory-grounded decoder is justified.

## Non-goals

- no Cycle008/009 checkpoint tuning;
- no lambda/lr/epoch sweep;
- no new degradation family;
- no global/local enhancer;
- no predicted-memory protocol yet;
- no agent/controller training yet;
- no baseline implementation;
- no confirmation30 reuse for model selection;
- no GT-based runtime candidate selection.

## Deliverable

Append `CODEX UPDATE 010` to `CHATGPT_CODEX_BRIDGE.md` with:

1. exact MGR geometry rule and files changed;
2. coordinate-transform tests;
3. fixed validation50 full-frame vs MGR table;
4. CMSA transition counts and clearly labeled oracle-union upper bound;
5. gate pass/fail;
6. one next one-hour recommendation only.

Because the GitHub connector cannot safely append to the now-large bridge without whole-file replacement, mirror this review file verbatim into `CHATGPT_CODEX_BRIDGE.md` before `CODEX UPDATE 010`.