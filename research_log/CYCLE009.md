# Cycle009 — one consistency ablation and a locked fresh confirmation

2026-09-18 06:14 +08:00: synced61a24c2; mirrored CHATGPT_REVIEW_008.md verbatim into bridge. Exactly one additional no-consistency training attempt, same Cycle008 contract except lambda_cons=0. Old diagnostic30 closed for selection/inference. Confirmation chosen by original holdout order after first30, image-byte disjoint from train300/val50/diagnostic30. Commit selection before training; SAM pseudo-label recovery is permitted asset preparation, never w15 inference/score-based selection.

Validation-only choice: require clean mIoU>=base−0.01 and clean CMSA>=base−1 group; highest degraded CMSA wins, then mIoU, but if equal CMSA and abs(mIoU difference)<0.002 keep Cycle008. Only the selected checkpoint and base may run fresh confirmation C/D. Acceptance: degraded mIoU+0.01/CMSA non-decrease, clean mIoU loss<=0.01/CMSA loss<=one group. No tuning on confirmation.

Confirmation30 locked: b390cf63beb328cbcabc84a0b70f00af7609ca288e09ff03bda9036d56b0c8b9. Asset recovery20260918-061525-cma-cycle009-assets completed06:15:47 exit0;30 exact source images+120 regenerated masks verified against150 hashes and pre-training lock. Zero confirmation w15 inference.26 CPU tests pass. Commit selection before smoke/full candidate training.
