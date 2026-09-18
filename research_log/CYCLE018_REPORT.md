# Cycle018 — first frozen SegLLM versus CMA val50 result

Implements REVIEW017 (`35ea314`). **200/200 SegLLM relational forwards completed, zero failed/invalid outputs, no reruns or tuning. All predictions were frozen before target access.** Frozen CMA base-w15 predictions were rescored on the exact same 50 groups, identity ordering, conditions and target masks. No CMA inference or training was performed.

## Comparison

Percentages below are on the reconstructed development/compatibility val50 set. Fidelity is reference-weighted; CMSA is pair-weighted. IER is the degraded identity-error rate.

| Method | Clean mIoU | Clean CMSA | Clean Fidelity | Degraded mIoU | Degraded CMSA | Degraded Fidelity | Degraded mean margin | Degraded IER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegLLM, pinned released checkpoint | 41.57% | 2/50 (4%) | 47/100 (47%) | 33.14% | 0/50 (0%) | 42/100 (42%) | -0.027858 | 39/100 (39%) |
| CMA base w15, frozen | 91.97% | 47/50 (94%) | 99/100 (99%) | 69.11% | 29/50 (58%) | 89/100 (89%) | 0.662931 | 3/100 (3%) |

Additional required metrics:

| Method / condition | Mean margin | Median margin | IER | Empty masks | Invalid outputs |
|---|---:|---:|---:|---:|---:|
| SegLLM clean | 0.047717 | 0.000000 | 41/100 | 0/100 | 0 |
| SegLLM target15_b | -0.027858 | 0.000000 | 39/100 | 0/100 | 0 |
| CMA clean | 0.919718 | 0.963190 | 0/100 | 0/100 | 0 |
| CMA target15_b | 0.662931 | 0.847463 | 3/100 | 0/100 | 0 |

Paired degraded-minus-clean changes, with rate changes in percentage points:

| Method | mIoU Δ | CMSA Δ | Fidelity Δ | IER Δ | Mean margin Δ | Median margin Δ |
|---|---:|---:|---:|---:|---:|---:|
| SegLLM | -8.43 pp | -4 pp | -5 pp | -2 pp | -0.075575 | 0.000000 |
| CMA base w15 | -22.86 pp | -36 pp | -10 pp | +3 pp | -0.256787 | -0.115727 |

All per-group IoU matrices, strict Fidelity/IER decisions and paired per-identity deltas are retained in `cycle018/scoring/`. The unchanged scorer uses correct IoU strictly greater than wrong IoU; ties are not fidelity successes. CMSA additionally requires both correct IoUs ≥0.5; IER requires wrong IoU > correct IoU and wrong IoU ≥0.5.

## Bounded interpretation and next decision

On this set, CMA exceeds SegLLM's degraded mIoU by35.97 percentage points and CMSA by58 points, with a much larger positive identity margin. SegLLM's39 degraded identity errors each have wrong-target IoU ≥0.5 exceeding correct-target IoU, so the gap cannot be described solely as generic inability to segment any helmet. Separately,24/100 degraded SegLLM predictions overlap neither paired target at IoU≥0.5, versus19/100 CMA predictions; these are localization/visibility failure indicators, not a visual diagnosis of the underlying cause. Empty masks explain none of the difference. SegLLM's IER decrease41→39 under degradation is not an improvement in correct-target performance: mIoU, Fidelity and margin all worsen.

**Layer-1 external development evidence is strong enough to freeze and move next to minimal oracle-free agent feedback**, under the review's decision rule. This is a next-task recommendation, not an agent implementation started in this cycle.

The claim is specifically **supplied-memory use**: the correct miner state is provided externally. This does not test end-to-end memory writing or autonomous maintenance. CMA has domain adaptation and prior selection on this reconstructed validation set; SegLLM is a released checkpoint with different training exposure. Targets are regenerated SAM pseudo-labels. These results do not establish pure architectural superiority or untouched final-set generalization. Also, CMA has a larger clean-to-degraded absolute drop; the evidence supports higher degraded absolute performance, not a smaller degradation sensitivity. No prompt, threshold, checkpoint or scorer was tuned to this result.

## Protocol and prediction freeze

- `cycle018/protocol_receipt.json` checks the exact Cycle014 manifest/scorer/degradation hashes and ordered50 groups/100 identities against both Cycle008 CMA prediction manifests. `inference_specs.json` contains only image, miner geometry, fixed query/seed and input hashes, with no helmet fields. All150 source image/miner-mask assets were hash-checked before preparation. `execution_manifest.json` records the100 lossless condition images' pixel/file hashes and supplied-memory provenance.
- The full runner adds only an outer50-group loop, loading the model once. `run_smoke.main` accepts optional CLI arguments and a preloaded model bundle and returns its receipt; its actual per-group preprocessing, prompt/history, native inference, selection, assertions and output mapping are unchanged. The two-gamma helper, model/dependency versions, HIPIE, native thresholds and checkpoint are unchanged. The weight-fidelity receipt passed for the actual full run.
- Every group retains Cycle017 index0, actual injected tensor hashes, identical A/B current-image/text, ordered final relational output selection and one unpadding/resize mapping. The native dummy inference target placeholders contain no helmet annotations and do not select predictions.
- `prediction_freeze.json` records all200 prediction paths/hashes and freeze time. In a separate scoring process, all200 hashes and group/identity order were rechecked before targets were opened. The combined raster-read audit contains only prepared images and supplied miner masks. `prediction_freeze_audit.json` records freeze time preceding scorer start.
- After freeze, every helmet target hash was checked against Cycle006 assets. Both conditions' saved CMA target and reference arrays were proven exactly equal to the corresponding frozen helmet and miner masks. CMA manifest hashes matched the predeclared Cycle008 exports; its prediction hashes are in `cma_reuse_provenance.json`. All CMA headline metrics recompute to the previously frozen values.
- `score_val50.py` only serializes saved masks to the existing schema, audits provenance, invokes the unchanged CMF evaluator and summarizes paired deltas/empty counts. No model is loaded during scoring and no score feeds back to inference.

## Runtime, artifacts and validation

Run `20260918-174555-cma-cycle018-val50` completed17:48:49 +08, exit0. Actual process load-to-freeze time166.43s; per-trial native inference plus postprocessing/export mean0.4086s, median0.4011s, range0.3828–0.7757s. Maximum allocated GPU memory16,499,173,376 bytes on A6000 GPU0. These are engineering metadata including resident allocations and warm-process effects, not controlled model-efficiency comparisons.

The replay archive with all200 masks, detailed per-group runtime receipts, full freeze record and scoring outputs is retained locally at `outputs/cycle018_results.tgz` and on A6000 at `research_log/cycle018_results.tgz`:899,708 bytes, SHA256 `6e9dc1d0de4c8bfa37cb9ca124210c98470dd45fca84deb08c2664d3adbf51b8`. All200 fetched prediction hashes matched the frozen record. Compact manifests, runtime/weight receipts, all per-group metrics and provenance are published in `research_log/cycle018/`; original input images and model weights are not published.

Focused validation:15 existing CMF/export tests passed in11.34s; new/changed scripts parse; actual200-trial assertions and post-freeze target/reference equivalence checks passed. No prediction failed, no output was omitted, no successful trial was rerun, no fallback configuration was used. Full command and code hashes are recorded in the runtime/protocol artifacts.

The first frozen group’s four NPY outputs also match the preexisting Cycle017 file hashes exactly. This checks the outer-loop refactor against saved evidence without adding another inference run.
