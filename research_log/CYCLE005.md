# Cycle 005 — frozen enhancer recovery

2026-09-18: resumed from CHATGPT REVIEW 004 / Cycle 005 at 5529d26. No existing Cycle 005 run or local edits. Issue #1 remains coordination context; bridge is the current task.

Baseline: 20 CPU tests passed. Reuse Cycle 003 CC/DD, exact first30 groups, seed0, w15 BF16, v1_multiround and supplied geometry. GPU0 free 50.6GB, no project evaluator process.

One bounded implementation: optional evaluator enhancement hook calling existing Stage-3 load_task_enhancer_model/run_task_enhancer, frozen v3_lowseg_best.pt, max_side1024. One call per seeded observation before reading masks, shared main/REF output. Existing scorer unchanged. Save lossless enhanced observations, checkpoint/source/input/output hashes. Tests cover single call, shared pixels and unchanged geometry plus previous paths; real integration is the one DE run. No training or GT selection.

Status: implementation in progress; no GPU run launched yet.

2026-09-18 03:20 +08:00: full CPU suite 20 passed after integration; git diff --check passed. Started exactly one DE run: 20260918-031959-cma-cycle005-de. Remote outputs outputs/cycle005_supplied_memory/DE. Check this run before any restart.


## Completed — 2026-09-18 03:20:33 +08:00

Run `20260918-031959-cma-cycle005-de` completed once, exit 0, 03:20:03–03:20:33 +08:00 (30 seconds). No retraining, tuning or repeat inference. GPU0 A6000; actual Stage-3 loader and preprocessing executed successfully. The 20-test CPU suite passed, including the extended single-call/shared-output wiring test in both conversation modes and unchanged legacy/factorial behavior.

Enhancer: `shared/enhancer/v3_lowseg_best.pt`, SHA256 `3f006e2244e2c6273252837f86c30f352d51fa43a45f33459fe791d230b82553`. Existing TaskEnhancerUNet base_channels=32, checkpoint epoch2/step600; these are historical metadata, not new training results. Frozen eval, max_side1024, RGB uint8/255, AREA resize down, BF16 autocast, clamp/round uint8 and CUBIC restore. The real functions are unchanged; the previously recovered Stage-3 file differs from repository only in two machine-path defaults. Its exact executed snapshot/hash is retained under cycle005. Checkpoint metadata is saved separately.

One enhanced output per seeded group observation, reused for main and all REF crops. There are 30 observations from 29 distinct underlying frames: the repeated frame retains its original group-specific noise seed, matching DD. Enhancement occurs before mask loading, and receives only image pixels. No GT-dependent output selection. The metadata's factor_degradation_configs describes target15_b input; provenance.enhancer describes the subsequent transformation.

Same 30 groups / 60 references; CC and DD reused, only DE newly run.

| Metric | CC | DD | DE |
|---|---:|---:|---:|
| target_miou | 0.943108 | 0.631131 | 0.615758 |
| cmsa | 0.966667 | 0.466667 | 0.433333 |
| memory_fidelity | 1.000000 | 0.816667 | 0.800000 |
| mean_identity_margin | 0.943108 | 0.621451 | 0.610108 |
| median_identity_margin | 0.958937 | 0.814648 | 0.816076 |
| identity_error_rate | 0.000000 | 0.000000 | 0.000000 |
| correct_iou_ge_0_5_count | 59.000000 | 43.000000 | 43.000000 |

DE minus DD: target_miou -0.015373; mean_identity_margin -0.011343; cmsa -0.033333; memory_fidelity -0.016667

| Group metric | Improved | Worsened | Tied |
|---|---:|---:|---:|
| target_miou | 18 | 12 | 0 |
| cmsa | 0 | 1 | 29 |


### Interpretation and exactly one next-hour recommendation

The predeclared **no gain / worse** branch applies. Mean mIoU decreases 0.015373, CMSA loses one group (14→13/30), Fidelity falls 49→48/60, and mean margin decreases 0.011343. The slight median-margin increase and 18 individually improved groups do not establish aggregate recovery. IER remains zero at threshold0.5; it is not a substitute for fidelity or localization quality. No statistical/general benchmark claim from 30 reconstructed groups (29 frames).

**Next one-hour task:** scope the smallest counterfactual-supervised degradation adaptation of the existing enhancer with w15 frozen, using only existing training data and an independent validation split to define acceptance; keep this 30-group diagnostic out of tuning. Do not treat the current v3-lowseg checkpoint as an effective paper contribution or add an agent. This is a recommendation awaiting ChatGPT review, not authorization to start training in this cycle.

### Evidence and recovery

`cycle005/recovery_results.json` contains full precision metrics, per-group paired mIoU/margin/CMSA deltas and matched-input checks. `cycle005/DE/memory_metrics.json` contains every IoU matrix. All CC/DD/DE group/query/seed/entity/target/reference and core model configurations match. All 30 source/degraded/enhanced image receipts were independently checked against pixels and saved PNGs.

Raw predictions and enhanced PNGs remain under remote `outputs/cycle005_supplied_memory/DE`. A 60,274,667-byte replay archive is saved both locally and remotely as `research_log/cycle005_replay.tgz` (Git-ignored), SHA256 `b49c926679e8268e57c2f9d642bde233304b33bbb732ec142aaeae7d85c6c221`. Compact results, exact launcher, summarizer, executed donor snapshot and runtime/checkpoint/image receipts are committed. Runtime metadata inherited an unrelated workflow releaseId; the actual command uses the explicit CMA root, and executed source hashes establish provenance instead.
