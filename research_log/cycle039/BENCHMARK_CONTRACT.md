# CMA_BENCH_LARGE_V1 — frozen execution contract

352 images/groups, 704 supplied identity memories, two observations per group: clean and target15_b. Each method produces 1,408 masks (352 × 2 identities × 2 conditions). All 352 selected groups passed restoration checks; no replacement occurred. This cycle generated zero task-model predictions.

CMA_BENCH_LARGE_V1.json binds selection, assets, protocol, execution inputs, QC and scorer hashes. protocol_receipt.json records the completed freeze at Unix 1789838250.4036596, before new task-model inference. execution_manifest.json embeds every group's prepared observation RGB/file hashes and identity-mask hashes. inference_specs.json carries the target-free raw-information inputs. Source/pair manifests and frozen_assets.json bind miner/helmet IDs, boxes, source annotations and pseudo-mask restoration receipts.

For each condition, A and B use the identical observation and semantic query: “Based on the miner mask from the previous round, segment only the mining helmet worn by that miner.” Only supplied miner identity changes. Native prompts may express the same relation in each model's supported syntax; record that adaptation. Image bytes, supplied miner masks/boxes, conditions and group order remain fixed. Internal model representations need not match, and comparisons must acknowledge that difference.

The unchanged target15_b implementation applies gamma 2.15, contrast 0.655, scale 0.405, noise sigma 23.5 and blur 0.88. Seed is 0; group seed is the integer from the first eight hex digits of MD5(`0:` + group ID). The frozen exporter hash binds operation order, numeric casting and kernels. This one compound stressor does not establish separate dust/glare/occlusion robustness.

Original segmentation masks were lost; fixed SAM vit_b restoration regenerated 1,408 masks for 704 miner/helmet pairs using recorded boxes. These are pseudo-labels, not human ground truth. Miner identity localization is externally supplied. Preparation uses helmet annotations/masks for validity and hashing; task inference must receive no helmet targets or target-derived proposal selection. Predictions freeze before the unchanged offline CMF scorer reads targets. This is not a globally target-blind preparation pipeline.

Report clean/degraded mIoU, CMSA, Memory Fidelity, IER, mean/median identity margin and paired deltas, together with source/QC limitations. Use the existing scorer definitions and defaults; do not substitute paper-specific baseline scores. CMF requires two actual identity-conditioned predictions. A baseline without identity input must be reported separately with CMF marked N/A.

Report model load, proposal/preprocessing time, model time, wall time, calls and peak GPU allocation when available; do not hide RAS proposal cost. Environment and checkpoint revisions accompany results. Current READY means an existing implementation, not completed evaluation on this new set.

Assets remain at A6000 `/home/wenchang/asdasdsad/wjq/coalminellm_recovery_20260917/shared/data/cycle039`; prepared RGBs are under its `research_log/cycle039/prepared`. Local receipt archive is `outputs/cycle039_replay/cycle039_asset_receipts.tgz`. Weights and PNGs are not committed to Git.
