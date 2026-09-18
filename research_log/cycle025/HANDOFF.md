# Cycle025 — last frozen model evaluation

Status: **DOCUMENTED_PROTOCOL_DISJOINT_CONFIRMATION**. Review024/598bd3c completed. The first result is retained with no performance gate, tuning branch, replacement or subgroup rerun. Model experimentation for this paper stops after this cycle; Layer2 remains retired.

## Pre-model asset integrity

Used exactly the Cycle024 manifest SHA256 `de4335243a9188cf0e2b55046668eca338fc85ec0e8a5b3c609d490f07e8503f` and registry SHA256 `837088966a5ef3fc6f24515318b4e4f0743a9c76e54a08fa1fcd8cbc17768aa7`. Original order preserved. **50/50 groups valid, 0 invalid, 0 replacements**; all100identities passed mechanical metadata and mask checks. Evaluated source counts remain23review groups and27accepted-pair combinations. The review groups retain their original `all_pairs_have_clean_episode=false`; no episode-QC status was promoted or rewritten.

Pre-restoration checks verified exact source bytes, absence from the exclusion registry, exact accepted-high pair records, distinct miner/helmet identities, source image/annotation dimensions and in-bounds boxes. Helmet annotation boxes were checked against stored metadata with0.002pixel serialization tolerance, not a quality threshold. The unchanged Cycle022-compatible SAM-B recipe restored50images and200nonempty image-aligned pseudo masks. Source annotation IDs, prompt boxes and original pair paths are retained in frozen_assets.json. The final complete valid/invalid audit was written and hashed before either CMA or SegLLM loaded; its hash is linked through selection_receipt.json into protocol_receipt.json. No quality-score sweep, morphology filter or visual ranking was used.

Asset preparation necessarily reads helmet pseudo-targets for mechanical validity and hashing. The guarantee is prediction-before-scoring isolation with target-free inference, not that targets were never accessed in preparation.

## Frozen methods and actual prediction isolation

Reused the already reproduced Cycle022 SAM recovery, CMA base-w15 loader/build_item/predict_item, SegLLM native interface and unchanged CMF scorer. Four cycle-specific entrypoints are exact copies after cycle022→cycle025 path/documentation substitution; adapter_equivalence.json records normalized source hashes and inverse-substitution equality. No scientific kernel changed; all frozen kernel hashes equal Cycle022. No extra baseline/model smoke was run.

CMA uses the same base-w15 model-file hashes, BF16,512tokens,v1_multiround,REF crop,seed0 and group-seed degradation convention. Actual CMA condition pixels were asserted identical to the prepared SegLLM pixels. Within each group the observation/query stay fixed while supplied miner identity changes. Every CMA forward asserts inference=True and zero target placeholders; raster reads are restricted to source/memory/prepared observations.

SegLLM retains source HEAD `4593a069f09628ce3a5b46e657f5417fefd7be46`, checkpoint revision `095e0637fcba015a02c0686f67b848c79c3cc80b`, the pinned three weight-shard hashes, two-gamma compatibility path, native `[REF:1]` prompt/history index0, and unchanged mask/box encoding, threshold and resolution mapping. Tracked source diff is empty. Source/weight audit was executed before inference in this cycle (its reused receipt scope text says post-run, but its timestamp and ordered run log establish actual pre-run execution).

CMA produced100group-condition forwards /200identity masks; SegLLM produced200native forwards. Both complete prediction manifests froze before scorer target access. The scorer verified prediction hashes, identity order and target-free runtime raster paths first, then opened target-bearing manifests and invoked the unchanged CMF scorer once per method/condition. Independent local replay verified all400raw prediction hashes and the full audit→protocol→both model starts→both prediction freezes→scoring time chain. No failed-looking group was rerun or removed.

## First frozen results

| Method | Condition | mIoU | CMSA | Fidelity | IER | Mean margin | Median margin | Empty masks | Groups / identities |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CMA base-w15 | clean | 88.28% | 46/50 (92%) | 97/100 (97%) | 1/100 (1%) | 0.873944 | 0.952629 | 0/100 | 50 / 100 |
| CMA base-w15 | target15_b | 67.53% | 29/50 (58%) | 84/100 (84%) | 3/100 (3%) | 0.646666 | 0.842875 | 0/100 | 50 / 100 |
| SegLLM pinned | clean | 42.71% | 2/50 (4%) | 49/100 (49%) | 42/100 (42%) | 0.049034 | 0.000000 | 0/100 | 50 / 100 |
| SegLLM pinned | target15_b | 36.44% | 1/50 (2%) | 46/100 (46%) | 38/100 (38%) | 0.031818 | 0.000000 | 0/100 | 50 / 100 |

Paired CMA minus SegLLM on identical groups (rates in percentage points):

| Comparison | mIoU pp | CMSA pp | Fidelity pp | IER pp | Mean-margin delta | Median-margin delta | Empty-mask delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| cma_clean minus segllm_clean | +45.57 | +88.00 | +48.00 | -41.00 | +0.824910 | +0.952629 | +0 |
| cma_target15_b minus segllm_target15_b | +31.08 | +56.00 | +38.00 | -35.00 | +0.614847 | +0.842875 | +0 |

Paired degraded minus clean, with no source-stratified performance selection:

| Comparison | mIoU pp | CMSA pp | Fidelity pp | IER pp | Mean-margin delta | Median-margin delta | Empty-mask delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| cma_target15_b minus cma_clean | -20.75 | -34.00 | -13.00 | +2.00 | -0.227278 | -0.109754 | +0 |
| segllm_target15_b minus segllm_clean | -6.26 | -2.00 | -3.00 | -4.00 | -0.017215 | +0.000000 | +0 |

## Interpretation and fixed claim boundary

On this preselected documented-protocol-disjoint set, CMA retains a substantial system-level advantage under the fixed compound stressor: **+31.08points mIoU,+56points CMSA,+38points Fidelity and−35points IER** relative to pinned SegLLM. Its degraded CMSA is29/50, leaving21groups that fail the strict two-identity criterion. CMA's clean-to-degraded mIoU decline is20.75points, versus6.26for SegLLM; the evidence supports higher absolute degraded performance, not smaller degradation sensitivity.

Allowed wording: **On a counterfactual set selected before inference and disjoint by source-image SHA256 from the reconstructed documented final-stage training and historical evaluation registries, frozen CMA retains stronger supplied-identity-memory performance than the pinned released SegLLM system under the fixed target15_b stressor.**

Exact run-bound w15 and ancestor/pretraining exposure remain UNKNOWN. This is not exact training-unseen generalization, a controlled architecture-only superiority result, autonomous memory writing/repair, or broad corruption/safety robustness. Targets are reconstructed pseudo labels; identity memories are externally supplied; the review/constructed-group QC provenance remains a limitation. Old Cycle018/022 results and their overlap disclosures are unchanged and are not pooled with this set.

## Execution, tests and artifacts

Run `20260919-015702-cma-cycle025-final-confirmation` started2026-09-19 01:57:07+08 and completed02:06:47+08,exit0. No prediction retries. NVML reported a driver/library mismatch, but direct CUDA tensor allocation on A6000 succeeded and both inference jobs completed; no driver/environment modification was made. Initial full CPU test collection lacked the existing src package on PYTHONPATH; setting it resolved collection without a code change.15focused existing tests passed16.71s,3new mechanical asset tests passed, and the complete54-test cmllm_remote suite passed26.12s. Syntax checks and path-only adapter equivalence passed. No further training/checkpoint roundtrip applies to this frozen evaluation.

The archive outputs/cycle025_results.tgz contains raw predictions and compact run/evaluation receipts; source images and restored target/memory masks remain on A6000 shared/data/cycle025. Input images, model weights and prepared PNGs are not added to Git. The raw prediction replay is under local outputs/cycle025_replay. All per-group IoU matrices and paired deltas remain in scoring/comparison.json and the four CMF reports. Runtime durations below are metadata, not a controlled speed comparison.

CMA load-to-freeze 95.03s; SegLLM 342.43s. Archive 1,378,601bytes, SHA256 `1188f4d69b9fdd61e4c7617a6c68f2694ee49447aadbef389f138fe188848386`. Asset audit SHA256 `5d62ca1e684f2d2ada9879e1b5635717f66b445abb31f57e9209a7a94de3e3eb`; protocol SHA256 `b1b234244a18a245a93ab50b95fb45a1e04d5b2cd4161c6b4740f4d15a8dce7a`.

Exactly one next recommendation: produce the paper tables, figures and text around this frozen documented-protocol-disjoint confirmation, retaining the exposure and pseudo-label/QC limitations. No further model experiment or in-repository subset search is authorized by this cycle.
