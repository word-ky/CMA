# Cycle038 — frozen crop dependence measured

REVIEW037/e0ee5a8 and Issue1 comment5743409418 completed. Two A6000 snapshots58seconds apart each showed48525MiB free, meeting24576MiB gate. Ran unchanged Cycle036 script exactly once:20260920-003129-cma-cycle038-crop-zero,exit0. No training, control rerun, code change, new split or Layer2/manuscript work.

Full -> crop-zero: clean mIoU88.28% ->39.99%, CMSA92% ->4%, Fidelity97% ->48%, IER1% ->39%; degraded mIoU67.53% ->34.28%, CMSA58% ->0%, Fidelity84% ->47%, IER3% ->37%. Zero-minus-full mIoU is-48.29points clean/-33.25degraded. CMSA lost/gained/changed counts44/0/44 and29/0/29. Mean/median margins and all100paired group-condition deltas are in CROP_DEPENDENCE_RESULTS.md, paired_results.json and PAIRED_GROUP_DELTAS.csv.

All200new masks frozen before scoring;100hook calls, runtime read/zero-target/pixel/source/checkpoint checks passed. Downloaded raw-mask hashes and unchanged Cycle025 control metric hashes verified locally. Existing tiny tensor checks passed remotely again; unchanged implementation already has54passing tests. Raw archive outputs/cycle038_results.tgz; compact receipts and Chinese TEACHING_UPDATE.md retained under research_log/cycle038. Historical failed-cycle files remain unchanged locally.

Scientific conclusion: current frozen w15 strongly depends on pooled crop features beyond retained supplied localization on this protocol. This is functional input dependence, not matched retraining, pure semantic attribution or from-scratch necessity. Distribution shift from zeroing, crop shape information, known pseudo-labelled subset and unknown historical exposure remain limitations.

Exactly one next recommendation: use the now-observed large crop-zero effect to decide in review whether the cost of the Cycle035 matched-continuation comparison is warranted; no training is started or presumed authorized.
