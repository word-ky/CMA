# Cycle 003 — completed real paired diagnostic

**Completed:** run `20260918-005016-cma-cycle003`, exit 0; 2026-09-18 00:50:20–00:52:38 +08:00, 138 seconds including recovery and both conditions. The initial launch notes below are history, not a pending run.

- 2026-09-18T00:51:11.8669655+08:00: Read CHATGPT REVIEW 002 / CYCLE 003 at c6f9f66. Scope: exact asset search, deterministic SAM-B fallback, frozen clean vs target15_b evaluation only.
- Run ID: 20260918-005016-cma-cycle003 (tmux). Remote root: /home/wenchang/asdasdsad/wjq/coalminellm_recovery_20260917.
- Sequence: recover_cycle003.py freezes first30 images/masks/manifests before run_cycle002_supplied_memory.sh executes w15 on clean then target15_b.
- Both A6000 GPUs showed approximately 50.6 GB free via torch.cuda.mem_get_info. Selected GPU0. No weights, thresholds, prompts or degradation parameters changed.
- Do not launch this run again while active. Next: inspect log, collect frozen provenance and real score reports, then select deterministic six-case paired gallery.

## Recovery and frozen protocol

Exact mask basename searches under `/home/wjq`, `/home/liujianhua/wjq` and the recovered project's parent returned no matches outside the known regenerated recovery directory; see `cycle003/exact_search.json`. Local emergency small archive also contained no matching PNGs. The available source image ZIP and SAM-B checkpoint supported deterministic reconstruction, so the original first 30 groups were retained; **no fallback or availability/result filtering**.

- 30 groups, 60 identity trials per condition; 29 distinct source image members and 59 unique identity pairs. Do not treat all trials/frames as independent samples.
- 30 image copies exactly recovered from archive member bytes; 118 unique miner/helmet mask assets regenerated with archived SAM-B recipe. Total 148 frozen assets. SAM predicted quality chooses the miner pseudo-mask; no GT/LISA target score is used for this generation choice.
- `cycle003/frozen_assets.json` records every source, pair/annotation metadata, generator/config, checkpoint SHA-256 and output SHA-256. All 148 assets and restored manifests were rechecked after inference; hashes match and freeze receipt predates the first prediction (`cycle003/verification.json`).
- Both conditions use identical masks/bboxes, target masks, queries, ordered IDs and seeds; paired comparisons verified reference and target arrays exactly equal. Appearance crops come from the respective clean/degraded observation. No weights, prompts, thresholds or degradation parameters were tuned.
- Frozen w15, BF16, `v1_multiround`, crop reference, seed 0; unchanged Cycle 002 runner. This is supplied identity geometry with condition-dependent appearance, not predicted memory, GT-free memory writing or exact historical pseudo-label reproduction.

## Results

| Metric | Clean | target15_b | Delta (degraded − clean) |
|---|---:|---:|---:|
| Target mIoU | 0.943108 | 0.631131 | -0.311977 |
| CMSA | 0.966667 (29/30) | 0.466667 (14/30) | -0.500000 |
| Memory Fidelity | 1.000000 (60/60) | 0.816667 (49/60) | -0.183333 |
| IER | 0.000000 | 0.000000 | 0.000000 |
| Mean identity margin | 0.943108 | 0.621451 | -0.321658 |
| Median identity margin | 0.958937 | 0.814648 | -0.144289 |

Full matrix/per-reference scores: `cycle003/{clean,target15_b}/memory_metrics.json`. Paired group deltas: `cycle003/paired_results.json`. No invented or synthetic model metrics in this table.

Degraded failures: 17/60 correct IoUs below 0.5; 11/60 zero correct IoU; 8 of these have zero IoU with both known targets, and 3 have wrong-target IoU greater than correct IoU. All predictions are nonempty. Maximum wrong-target IoU is 0.368316, below the fixed IER threshold, so IER=0 does not mean no association drift. Mean wrong IoU is 0.009680.

Clean identity use is strong on this diagnostic; degradation substantially reduces quality-qualified pair success. However, mean-margin loss (0.321658) is close to mIoU loss (0.311977), with only 0.009680 extra due to wrong-target overlap. CMSA is a thresholded conjunction of two predictions and amplifies localization failures. **Do not infer identity-specific damage solely from its larger numerical drop.** Existing evidence predominantly shows segmentation/localization deterioration, with three low-quality wrong-identity tendencies; causal damage to memory appearance remains unisolated.

## Gallery and validation

[Six paired examples](cycle003/RESULT.md): 2 largest negative group-mean margin deltas; no quality-qualified identity-error cases available; 2 stable successes by group ID; fill 2 remaining slots by margin delta. No aesthetic selection. Overlay red = prediction, green = corresponding target. Largest-drop and stable-success panels visually inspected. The low-quality label does not exclude wrong-target overlap below 0.5.

- Source recovery and two real GPU evaluations completed successfully; 120 raw prediction files.
- Paired identity/query/seed and target/reference consistency checks passed; frozen hashes passed.
- Existing CPU suite rerun: **19 passed**. No core evaluator changes this cycle.
- Post-run verification initially used the system's older Python (missing `hashlib.file_digest`); reran with the project's Python 3.12 environment. No model rerun or data change.
- Raw masks and complete restored data remain on the server. A ~13 MiB compressed replay bundle was fetched to local `research_log/cycle003_replay_assets.tgz` (ignored by Git). Git contains compact reports, provenance/manifests and gallery; published manifests are audit references, not self-contained inputs without that asset bundle.

## Exactly one next recommendation

Use this same frozen subset/model to add the two off-diagonal conditions in a **main-image quality × REF-appearance quality 2×2 control**. Keep supplied geometry fixed. Together with the two existing conditions, this separates general image segmentation degradation from degradation of the memory appearance evidence. No agent/verifier/training work yet.
