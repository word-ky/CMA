# Frozen inference dependence: crop features under fixed localization

Status: COMPLETE. REVIEW037 / Cycle038 executed the unchanged Cycle036 diagnostic once after the two-snapshot capacity gate passed. Same frozen w15, same Cycle025 50 groups / 100 identities, clean + target15_b, BF16, seed0; only pooled crop features zeroed before bbox addition. No training, new split, control rerun or default production change.

| Condition | Arm | mIoU | CMSA | Fidelity | IER | Mean margin | Median margin |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | full | 88.28% | 92% | 97% | 1% | 0.873944 | 0.952629 |
| clean | zero | 39.99% | 4% | 48% | 39% | 0.043327 | 0.000000 |
| target15_b | full | 67.53% | 58% | 84% | 3% | 0.646666 | 0.842875 |
| target15_b | zero | 34.28% | 0% | 47% | 37% | 0.005095 | 0.000000 |

Zero minus full; rates below are percentage-point deltas, margins unscaled.

| Condition | mIoU pp | CMSA pp | Fidelity pp | IER pp | Mean margin | Median margin | CMSA lost / gained / changed |
|---|---:|---:|---:|---:|---:|---:|---|
| clean | -48.29 | -88 | -49 | 38 | -0.830617 | -0.952629 | 44 / 0 / 44 |
| target15_b | -33.25 | -58 | -37 | 34 | -0.641571 | -0.842875 | 29 / 0 / 29 |

All 100 paired group-condition rows, including per-identity margin deltas, are in paired_results.json and PAIRED_GROUP_DELTAS.csv. Clean CMSA falls46/50 ->2/50 (44lost,0gained); degraded29/50 ->0/50 (29lost,0gained). No group deletion or alternative run selection.

Interpretation: clear degradation in both conditions supports functional dependence of this already-trained w15 on pooled crop features even with miner bbox/mask localization and main-image evidence retained. This is a frozen input intervention, not a trained component ablation. Masked crop includes shape/localization structure; zeroing can also cause distribution shift. The effect is not pure semantic identity attribution, from-scratch necessity, proof of matched-retraining benefit, or agentic memory robustness. The existing set is known and small, targets are reconstructed pseudo labels and exact historic training exposure remains unknown.

Execution: run20260920-003129-cma-cycle038-crop-zero,00:31:34–00:32:55+08,exit0. Two gate snapshots58seconds apart showed48525MiB free on both cards and no active compute processes. PhysicalGPU1 used. Remote self-check passed.100 model group-forwards produced200 new identity masks; runtime hook recorded100 pooled crop interventions. Prediction freeze timestamp1789835569.89132 precedes scoring1789835571.1611362. All200 fetched raw-mask hashes and original full metric hashes verified locally. Runtime raster allowlist, zero-target hook, same-condition pixel check and exact source/checkpoint binding passed. Scorer unchanged; two condition reports generated only after freeze.

Raw predictions remain at outputs/cycle036/crop_zero on A6000 and local outputs/cycle038_replay/outputs/cycle036/crop_zero; replay archive outputs/cycle038_results.tgz. Compact receipts are copied into this cycle directory without modifying historical Cycle036 failure records. Existing54-test pass applies to unchanged implementation; no redundant model/test rerun.
