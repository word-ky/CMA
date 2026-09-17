# Cycle 006 — split preparation; training not started

2026-09-18 03:40 +08:00: synchronized CHATGPT REVIEW 005 / Cycle 006 from 13cb7a9. Baseline 20 CPU tests passed. Task is one fixed enhancer fine-tune with w15 frozen, historical restoration regularizers and new REF BCE+Dice (0.02) / CF rank (0.5, margin0.05), one epoch, at most300 groups, lr5e-6. Validation gate: mIoU gain>=0.02 AND CMSA not lower. No diagnostic30 inference unless pass.

Observed dependency gaps before implementation/training:

- `episodes_counterfactual_ref_train_grouped.jsonl` is absent locally and in project-owned A6000 roots. The original builder and accepted source manifests exist; reconstruct deterministically without changing membership or query.
- Existing image/target/miner paths need recovery for the selected groups, as in Cycle003.
- Historical v3 checkpoint metadata specifies teacher `task_enhancer_v1_train4k_gpu3_20260528/best.pt`, lambda_teacher=1.0. That teacher is absent from local project file search, emergency archive entry list and A6000 project-owned roots. Available v2/v2.1/v3 checkpoints are not substitutes for that v1 teacher.
- Retried the configured old H100 source once through AutoDL helper: SSH exit255, jump-host `Permission denied (publickey)`. No weights recovered.

Independent progress: freeze exactly300 training and50 validation groups by SHA256(seed:group_id), one group per unique archived image-byte hash. Exclude every image hash in original holdout buckets8/9, including aliases under different instance paths. Retain manifest hashes/IDs and missing-asset inventory under cycle006. This is split preparation, not a result or training attempt.

No trainer variant or optimizer run yet: preserving the requested historical teacher loss is blocked by missing teacher weights. Do not silently replace teacher or remove its weight. Next check the split receipt; report the exact dependency to ChatGPT for recovery or an explicit task revision.

03:45 +08:00: 21 CPU tests passed. Split frozen at 300/50; hashes in cycle006/split_receipt.json. Data recovery launcher returned run ID 20260918-034521-cma-cycle006-assets but SSH closed during launch (exit255); launch status uncertain. Inspect tmux/process/log before retry. This is SAM asset regeneration only, not an enhancer training attempt.

03:46 +08:00: confirmed first failed launcher created metadata only (no process, tmux session or train.log). Retried asset recovery as 20260918-034552-cma-cycle006-assets, tmux launch confirmed. Still zero enhancer training attempts.

03:48 +08:00: asset recovery and verification completed. Remaining blocker is the missing teacher; zero training/diagnostic runs. Full update follows.


## CODEX UPDATE 006 — split/assets ready; training blocked by missing historical teacher

**Status: BLOCKED, not completed; training has not started.** Base `13cb7a9`. The old v3 checkpoint requires a historical v1 teacher for its teacher regularizer (weight1.0). The exact referenced checkpoint `task_enhancer_v1_train4k_gpu3_20260528/best.pt` is missing. Available v2/v2.1/v3 weights are different checkpoints. Substituting one or removing the regularizer would change the requested fixed experiment. A configured H100-source retrieval attempt failed with SSH exit255 (jump-host public-key authentication denied); no credential files are published.

### Completed preparation

- Reconstructed `episodes_counterfactual_ref_train_grouped.jsonl` using the unchanged historical builder and accepted source manifests: 4,652 total groups, 3,731 in original training buckets0–7. This is a deterministic reconstruction, not a recovered original file byte claim.
- Fixed seed20260528, sort by SHA256(seed:counterfactual_id), keep one group per unique archive-image byte hash, exclude all 821 unique image hashes from original holdout buckets8/9, select first300 training + next50 validation. Exclusion handles differently named instance copies of the same source image. No outcome or loss-based selection.
- Selection IDs, image hashes, source hashes and exact manifests are in `research_log/cycle006/split_receipt.json`. Original split manifest SHA256: train `dc561f54ef01f206e3987a36a46c74d9bb1be2a4cdeb5804d07edc284fb10bcc`; val `a844953079d563408088c7ab7eb6a67c2e1f69eea2e83fef942fd6a7acbb4467`.
- Recovered350 exact archived images and regenerated1,400 pseudo masks using the already recorded SAM-B recipe. These masks are **not exact historical masks**. Fixed IDs, queries, miner bboxes and source-image hashes all preserved. All1,750 asset hashes rechecked; restored train/val manifests and hashes are in `cycle006/assets_ready.json`. Assets occupy134MiB under remote project `shared/data/cycle006`.
- No Cycle003 diagnostic inference or enhancement-model selection was performed. Byte-level holdout hashes were used only for exclusion.

### Code, tests and actual run

`cmllm_remote/scripts/prepare_cycle006_split.py` reuses the historical builder for the deterministic split; its CPU test checks input-order independence, duplicate-content aliases and holdout exclusion. `research_log/recover_cycle003.py` now accepts optional selected-manifest/output-directory arguments; defaults preserve Cycle003, and the SAM mask recipe is unchanged. `finish_cycle006_assets.py` maps fixed selections to recovered paths and verifies assets. Full CPU suite: **21 passed**, static parse/diff checks passed.

One actual **asset-regeneration** run (not training): `20260918-034552-cma-cycle006-assets`, 03:45:56–03:47:08 +08:00,72 seconds, exit0. Command: `CUDA_VISIBLE_DEVICES=0 <root>/.venv/bin/python <root>/research_log/recover_cycle003.py <root> --groups-jsonl <root>/research_log/cycle006/selected_source_groups.jsonl --data-name cycle006`. The earlier launcher `20260918-034521-cma-cycle006-assets` lost SSH before tmux launch; verified no process/session/log before retrying. Both states are preserved in the runtime receipt.

### Training and gate: not run

The predeclared configuration is saved in `cycle006/training_contract.json`: one epoch, AdamW/lr5e-6, historical restoration weights including teacher1.0, direct miner0.005/helmet0.01, REF target0.02, rank0.5, margin0.05. No new trainer or optimizer step was implemented/run before resolving the required teacher dependency; no smaller substitute dataset or replacement weights were used.

| Validation metric | Old v3 | New v4 | Delta |
|---|---|---|---|
| target mIoU | not run | not run | unavailable |
| CMSA | not run | not run | unavailable |
| Memory Fidelity | not run | not run | unavailable |
| mean identity margin | not run | not run | unavailable |

Gate status: **NOT EVALUATED**, neither pass nor fail. Training runs0; diagnostic30 inference runs0. The proposed differentiable path remains the historical main-SAM-image branch with frozen w15 and detached REF/CLIP pixels; no v4 gradient-path claim or gradient test is made yet.

### Exactly one next one-hour recommendation

Recover the exact historical v1 teacher checkpoint (or have the research owner explicitly revise that dependency in the fixed contract), then resume this same frozen300/50 Cycle006 split for the single prescribed attempt. Until that dependency changes, do not substitute a teacher, drop its loss, start direct-w15 adaptation, rerun preparation, or report a failed validation gate.
