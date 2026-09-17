# Cycle007 — revised proximal teacher, one adaptation

2026-09-18 04:51 +08:00: synced ChatGPT review006 / Cycle007 at87523a4. Explicitly replaces missing historical teacher with frozen v3, weight0.25. Frozen Cycle006 train300/val50 retained, no asset regeneration or split change. Other weights/lr/gate as declared. GPU0 free50.6GB, no existing v4 run.

Reuse v3 architecture, loader, restoration/direct BCE+Dice helpers, differentiable SAM preprocessing; evaluator build_item for grouped multiround REF identity queries; existing LISA soft_iou_matrix for margin0.05 rank. New trainer holds all w15 parameters frozen and shares one differentiable SAM embedding across REF/direct forwards. One enhancer output per original-resolution deterministic degradation, AREA resized to384 for training. Original miner boxes scaled to384 with nearest resized masks. REF/CLIP pixels detach/round; main SAM resize remains differentiable. Historical direct losses are averaged over group identities; restoration weighted masks are per-identity then averaged. Validation stays existing native-resolution max_side1024 Stage3 pipeline. This training/evaluation resize difference is recorded, not tuned.

22 CPU tests passed including rank ordering/backward and previous data/evaluator wiring. Next: fixed first-group identity-only backward smoke (no optimizer), then one full300-step epoch only if finite non-zero gradient. Evaluate oldv3/newv4 on fixed50 with unchanged scorer; diagnostic30 only after mIoU gain>=0.02 and CMSA non-decrease.

04:55 +08:00: launched fixed-batch gradient smoke 20260918-045454-cma-cycle007-smoke. No optimizer steps. Check its log before training.

04:56 +08:00: smoke 20260918-045454-cma-cycle007-smoke completed04:54:58–04:55:09 exit0. Identity-only enhancer gradient norm14.412564277648926 finite; w15 frozen/no gradients; REF detached; main SAM differentiable; zero optimizer steps. Full single training+validation/gate run launched: 20260918-045620-cma-cycle007-full. Inspect this run before any resumption; do not restart based on losses.


## Gradient and reuse details

Student/teacher share the original v3 checkpoint hash; teacher has requires_grad=False and no_grad forward. W15 has requires_grad=False globally, eval mode, no optimizer parameters. Main SAM embeddings use the v3 differentiable_get_visual_embs override and are computed once per group, then reused for REF and two direct-task forward passes. The frozen image encoder still differentiates with respect to its input. CLIP/REF crops are produced through detached rounded RGB bytes, so the observed gradient is through main SAM only. Identity-only smoke used unweighted REF BCE+Dice + rank (restoration/direct terms excluded); the fixed first group's rank hinge was zero, while BCE+Dice supplied a finite nonzero gradient. This proves the combined path, not that the rank hinge was active on that first group; CPU rank test separately exercises an active wrong-identity hinge and backward.

Full run begins with fresh v3 student initialization; smoke does not alter saved weights. Single final-epoch checkpoint is the only candidate. Training log records every group/step and loss. Evaluation label flag defaults to historical v3lowseg but allows v4 provenance without changing pixels/model/scorer. No source/mask selection uses model outcomes.


**Completed05:02:35 +08:00; validation gate FAIL; no diagnostic30 run.**


## CODEX UPDATE 007 — one adaptation completed; validation gate FAIL, enhancer line stops

Completed the single revised experiment from `87523a4`: student and frozen proximal teacher both start from v3; teacher weight0.25. The exact Cycle006 image-disjoint300/50 split and regenerated masks were reused without new selection or recovery. This is a new reconstructed adaptation experiment, not historical v3 reproduction.

### Implementation and fixed recipe

`train_task_enhancer_v4_counterfactual.py` reuses the v3 UNet/loader, restoration/direct BCE+Dice helpers and differentiable SAM preprocessing; the evaluator's actual grouped `v1_multiround` build_item; and LISA's actual soft_iou_matrix. One deterministic source-resolution target15_b observation (group_seed, seed0) is AREA resized to384, enhanced once and shared across identities. Miner bboxes scale with the image; masks use nearest resize. REF crop/CLIP bytes detach and round; main SAM input remains differentiable through bilinear resize to1024. Training384 versus native-resolution/max_side1024 evaluation is recorded explicitly; no resolution sweep.

All w15 parameters stay frozen/eval. One differentiable SAM embedding is shared across the REF and two historical direct-task forwards. REF target loss is BCE+Dice averaged over identities. Rank is mean relu(0.05 + max wrong soft-IoU − correct soft-IoU), with float32 loss arithmetic. Direct miner/helmet losses average the historical per-identity targets; restoration masks are per-identity and averaged. The teacher runs frozen/no_grad. No REF-image gradient is claimed.

One epoch,300 AdamW steps, lr5e-6, weight_decay1e-4, clip_norm1.0, seed20260528; weights: align1.0, miner0.005, helmet0.01, residual0.08, teacher0.25, color0.08, TV0.005, REF target0.02, CF rank0.5; rank margin0.05. Single final `last.pt`, no intermediate selection. Config and every group/step/loss are preserved under `research_log/cycle007/train/`. The only evaluator behavior extension is an explicit enhancer label; default v3 behavior and CMF scorer are unchanged.

### Gradient smoke and tests

Fixed first group `cf_d0fa96077bf85e0a`; only unweighted REF BCE+Dice + CF rank backward, no optimizer created/step. REF loss0.7943140864, rank hinge0 on this group, enhancer gradient norm **14.4125642776**, every gradient finite. w15 parameters frozen with no gradients; REF pixels detached; main SAM tensor differentiable. The combined-path smoke does not claim an active rank hinge on this first group: the CPU wrong-identity test exercises active rank/backward, and full training has positive rank losses (mean0.04109034).

**22 CPU tests passed**, including inherited data/evaluator tests and active rank ordering/gradient test. Real GPU smoke exited0; full training/evaluation exited0; both v3/v4 checkpoints loaded and produced all50×2 validation outputs. Collection verified exactly300 unique prescribed training groups, each used once, and matched validation group/query/seed/identity/reference/target inputs.

### Actual runs and provenance

- Smoke: `20260918-045454-cma-cycle007-smoke`, 04:54:58–04:55:09 +08:00,11s, zero optimizer steps.
- Sole training + old-v3/new-v4 validation + gate: `20260918-045620-cma-cycle007-full`, 04:56:24–05:02:35 +08:00,371s, exit0. Command: `bash <root>/research_log/run_cycle007_full.sh <root>`. It calls the exact trainer flags in `run_cycle007_train.sh`, then two unchanged CMF evaluations and the predeclared gate.
- Initial/teacher v3 SHA256: `3f006e2244e2c6273252837f86c30f352d51fa43a45f33459fe791d230b82553`.
- Final v4 SHA256: `b22b85e601729a7b65ad975cfd735eb03cc34f25c7384a7cb9c011cdc9b90f8f`; local copy matches remote. Source hashes match deployed trainer/evaluator. Compact receipts and full per-group score matrices are committed; weights/images stay in project-local and remote replay artifacts.

### Fixed validation result (50 groups /100 references)

| Metric | Old v3 | New v4 | v4−v3 |
|---|---:|---:|---:|
| target_miou | 0.685059 | 0.678440 | -0.006620 |
| cmsa | 0.560000 | 0.500000 | -0.060000 |
| memory_fidelity | 0.900000 | 0.890000 | -0.010000 |
| mean_identity_margin | 0.648614 | 0.638023 | -0.010592 |
| median_identity_margin | 0.860473 | 0.871516 | +0.011043 |
| identity_error_rate | 0.040000 | 0.040000 | +0.000000 |

**Gate FAIL:** required mIoU gain>=0.02 AND CMSA non-decrease; observed mIoU delta−0.006620 and CMSA delta−0.06. CMSA28→25/50, Fidelity90→89/100. IER remains4/100 and median margin rises slightly; neither overrides the predeclared failure. No diagnostic30 directory or inference was created; diagnostic30 runs=0. No repeat, sweep, alternate teacher or enhancer restart was performed. Stop this enhancer route as instructed; the negative result applies to this fixed bounded attempt and is not a universal impossibility claim.

### Exactly one next one-hour recommendation

Move to a bounded **direct w15 adaptation with clean/degraded counterfactual mixing**, using the existing frozen train/validation split and a predeclared trainable-parameter/mixing recipe. Compare to unadapted w15 on that validation set; keep the enhancer and agent out. Await the research owner's exact next contract before starting this new training stage.

Replay persistence: `research_log/cycle007_replay.tgz`,183,320,739 bytes, SHA256 `887b181849ff65c5845ef0ed2a50ff8a4c768db525948ace082c65fc5bc4f68e`; verified matching local/remote copies. Contains checkpoint, train logs, validation raw masks and lossless enhanced images. Git-ignored; compact receipts/matrices/manifests are published.
