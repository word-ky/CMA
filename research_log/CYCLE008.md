# Cycle008 — direct w15 degradation adaptation

2026-09-18 05:37 +08:00: synced review41b614c and mirrored research_log/CHATGPT_REVIEW_007.md verbatim into bridge as requested. Cycle007 enhancer line remains closed. Frozen Cycle006300/50 inputs reused. No prior Cycle008 outputs/process; no cached no-enhancer validation50 C/D outputs. GPU0 free50.6GB. Baseline22 tests pass.

Contract: only existing mask_decoder/text_hidden_fcs/ref_hidden_fcs/ref_visual_fcs/ref_input_bbox_fcs/ref_input_fcs/ref_embedding_scale trainable; LLM/CLIP/SAM image encoder frozen. Clean and deterministic target15_b paired observations, same group query/identity/geometry. Total0.5*(segC+segD)+0.5*(rankC+rankD)+0.25*BCE(D,sigmoid(C.detach())); rank margin0.05. One300-step epoch, AdamW5e-6/weight_decay1e-4, clip1.0, seed20260528. Native-resolution evaluator build_item and preprocessing reused; no enhancer or training resize.

Planned single fixed-batch backward, then full epoch. Trainable subset uses FP32 master parameters under BF16 autocast; frozen base remains BF16. Sequential C backward then D backward gives the exact declared gradient with clean stop-gradient for consistency; optimizer steps only after both. Final checkpoint stores only declared trainable tensors over immutable w15, avoiding duplicate frozen weights. Validation gate: degraded mIoU+0.02/CMSA non-decrease, clean mIoU loss<=0.01/CMSA loss<=0.02. Diagnostic30 C/D pair only once if pass; no holdout inference otherwise.

05:39 +08:00: no-enhancer base C/D run20260918-053830-cma-cycle008-base launched. Trainer subset/loss/detached-teacher tests added; full CPU suite24 passed. Native-resolution inputs maintained.

05:41 +08:00: base run completed05:39:43 exit0. Fixed paired-backward smoke launched20260918-054116-cma-cycle008-smoke; no optimizer step. Inspect smoke before full training.

05:42 +08:00: smoke passed05:41:21–05:41:32 exit0. 73,442,533 trainable scalars/141 tensors; all7 requested module families receive finite nonzero gradients. Frozen parameters gradient-free. Six IoU-quality-head tensors within mask_decoder have no gradient because requested mask losses do not use predicted quality; retained in declared subset, no quality loss added. No optimizer steps in smoke.

05:43 +08:00: full single training/gate run20260918-054248-cma-cycle008-full launched. Preserve run ID; no restart or parameter changes based on outcomes.

25 CPU tests pass, including sequential paired-backward equality to the complete declared objective. Exact141 parameter names/shapes/counts and7 family gradient norms saved in cycle008. FP32 subset checkpoint is loaded onto the original BF16 w15 evaluator for deployment evaluation; base C/D were produced by unchanged d68d50d evaluator before adding the optional checkpoint flag. Base clean mIoU0.9197182919; degraded0.6911134047. No threshold/config changes after baseline observation.


**Completed05:53:25 +08:00. Validation PASS; diagnostic CMSA gain not reproduced.**


## CODEX UPDATE 008 — direct adaptation passes validation; diagnostic identity gain remains unproven

Completed the single task from `41b614c`. `research_log/CHATGPT_REVIEW_007.md` was mirrored verbatim into this bridge before this update. No enhancer, agent or new split; reused exact Cycle006 train300/val50 with reconstructed masks and recorded image-byte isolation. The no-enhancer baseline was freshly established, not replaced by enhancer results.

### Implementation, parameter scope and gradient verification

`train_w15_degradation_counterfactual.py` reuses the normal evaluator build_item, native image/mask/bbox geometry, `v1_multiround` query and prediction ordering, existing BCE+Dice and counterfactual soft-IoU rank helpers. Both clean and source-resolution deterministic target15_b (group seed0) are constructed for each group. There is no384 training downscale in this cycle. SAM longest-side1024 and CLIP processing are the normal evaluator paths.

Only the following existing surfaces are trainable: **73,442,533 scalars across141 tensors**. Exact names/shapes/dtypes are in `cycle008/trainable_parameters.json` (also the smoke/train directories).

| Surface under model | Parameters | Smoke gradient norm |
|---|---:|---:|
| ref_embedding_scale | 1 | 0.046159968 |
| visual_model.mask_decoder | 4,058,340 | 2.273122468 |
| text_hidden_fcs | 17,830,144 | 0.640192569 |
| ref_hidden_fcs | 17,830,144 | 0.003499307 |
| ref_visual_fcs | 132,608 | 0.000114993 |
| ref_input_bbox_fcs | 16,801,792 | 0.243747744 |
| ref_input_fcs | 16,789,504 | 0.814293301 |

LLM backbone, CLIP, SAM image encoder, prompt encoder and all other parameters stay frozen. All7 requested families receive finite, nonzero gradients on the fixed smoke group `cf_d0fa96077bf85e0a`; every frozen parameter remains gradient-free. Six tensors in mask_decoder's IoU-quality prediction head have no gradient because these mask losses do not use its quality output. They remain in the declared subset; no extra quality objective was introduced. Smoke made zero optimizer steps.

Trainable master parameters are FP32 under BF16 autocast; frozen base stays BF16. REF-input gradients traverse the frozen LLM to the trainable projections; no backbone weights are optimized. Clean0.5*(BCE+Dice+rank) backward runs first, then degraded0.5*(BCE+Dice+rank)+0.25*BCE(degraded,sigmoid(clean.detach())) backward, then one optimizer step. A numerical unit test confirms equality to the declared joint gradient. Rank margin0.05; clean prediction detached only for consistency. AdamW5e-6, weight_decay1e-4, clip1.0, seed20260528, one epoch/300 groups; no loss/learning-rate changes or checkpoint selection.

Final checkpoint stores only the declared trainable tensors over immutable base w15. The evaluator's optional `--adaptation-checkpoint` loads that subset into normal BF16 w15; default no-adaptation behavior/scorer is unchanged. **25 CPU tests passed**, real paired-gradient smoke passed, each prescribed group used exactly once, and base/adapted validation and diagnostic identities/queries/seeds/targets/reference masks match.

### Actual runs

- No-enhancer base C/D validation: `20260918-053830-cma-cycle008-base`,05:38:33–05:39:43 +08:00,70s,exit0. Used the unchanged d68d50d evaluator before adding the optional checkpoint loader.
- Fixed paired-gradient smoke: `20260918-054116-cma-cycle008-smoke`,05:41:21–05:41:32 +08:00,11s,exit0,zero optimizer steps.
- Sole train + adapted validation + conditional diagnostic: `20260918-054248-cma-cycle008-full`,05:42:53–05:53:25 +08:00,632s,exit0. Exact command: `bash <root>/research_log/run_cycle008_full.sh <root>`; subordinate trainer/evaluator commands are in the committed launchers.
- Final adaptation checkpoint SHA256: `0541449d9c6d06618de8b3f17a4289cb495e336b6600b35dd7d51a3ad48110d1`,293,810,853 bytes; local/remote copies verified. No duplicate frozen model is required; load over the recorded w15 base.

### Fixed validation50: no enhancer in any cell

| Metric | Base C | Adapted C | Base D | Adapted D |
|---|---:|---:|---:|---:|
| target_miou | 0.919718 | 0.920641 | 0.691113 | 0.713416 |
| cmsa | 0.940000 | 0.960000 | 0.580000 | 0.620000 |
| memory_fidelity | 0.990000 | 1.000000 | 0.890000 | 0.930000 |
| mean_identity_margin | 0.919718 | 0.920641 | 0.662931 | 0.698181 |
| median_identity_margin | 0.963190 | 0.961073 | 0.847463 | 0.856061 |
| identity_error_rate | 0.000000 | 0.000000 | 0.030000 | 0.020000 |

**Gate PASS, all four criteria true:** degraded mIoU+0.022302>=0.02; degraded CMSA+0.04 (29→31/50); clean mIoU+0.000922; clean CMSA+0.02 (47→48/50). Degraded Fidelity89→93/100 and mean margin+0.035250. This is the predeclared point-estimate gate on one fixed split, not a statistical significance or broad benchmark claim.

### Single diagnostic30 evaluation after PASS

Only the final passing checkpoint was run, once per clean/degraded condition; both base cells were reused from Cycle003. No diagnostic inference occurred before the validation gate. No candidate was selected using diagnostic results.

| Metric | Base C (reused) | Adapted C | Base DD (reused) | Adapted DD |
|---|---:|---:|---:|---:|
| target_miou | 0.943108 | 0.938923 | 0.631131 | 0.639207 |
| cmsa | 0.966667 | 0.966667 | 0.466667 | 0.433333 |
| memory_fidelity | 1.000000 | 1.000000 | 0.816667 | 0.816667 |
| mean_identity_margin | 0.943108 | 0.938923 | 0.621451 | 0.639008 |
| median_identity_margin | 0.958937 | 0.952980 | 0.814648 | 0.829782 |
| identity_error_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

The diagnostic outcome is mixed: degraded mIoU+0.008076 and mean margin+0.017557, but CMSA14→13/30 and Fidelity unchanged49/60. Clean mIoU−0.004186, with CMSA29/30 and Fidelity60/60 unchanged. Thus clean identity preservation remains good, but the validation identity gain does **not** yet establish a stable diagnostic CMSA improvement. Do not present the passing validation gate as a general identity-robustness breakthrough. This bounded result does not reopen enhancer or agent work.

### Exactly one next one-hour recommendation

Run one **validation-only consistency ablation**: remove L_cons while holding the same initialization,300/50 split, trainable subset, paired observations, optimizer and epoch fixed. Compare to this candidate on the same C/D validation set to isolate whether consistency adds benefit beyond supervised clean/degraded adaptation. Do not use diagnostic30 for further candidate selection. Await the research owner's next explicit task before that run.

Replay archive: `research_log/cycle008_replay.tgz`,268,521,278 bytes,SHA256 `42f6fddc70725d4cdd1f5b07181b5db1aaa8a061ece0fcd0d4396b0c30274b73`; local and remote verified. Includes final subset weights and raw predictions; Git-ignored. Compact evidence is published.
