# Cycle022 — fresh confirmation supports freezing Layer1; Layer2 stays retired

Implements REVIEW021 (`45ec6f4`). **The first frozen result confirms a large Layer1 advantage on50 new image-disjoint holdout groups.** Under target15_b, CMA base-w15 achieves68.02%mIoU,30/50CMSA,87/100Fidelity and2/100IER; pinned SegLLM achieves27.52%mIoU,1/50CMSA,38/100Fidelity and30/100IER. No model/prompt/threshold/degradation changes, subgroup reruns or Layer2 operations were performed.

## Prior-use registry and frozen selection

Before selecting or running a model, scanned750 prior research_log/output JSON, JSONL, Markdown and text/log artifacts on A6000. Resolved group IDs, identity pair IDs, image filenames and explicit hashes to original mining_helmet archive-image bytes. Every excluded image SHA256 retains source-file provenance; every scanned file retains its own hash. Conservative mentions, including skipped IDs, are excluded rather than used to rescue sample count. Registry contains414 unique previously used image hashes.

Checked the original counterfactual holdout pool (unchanged split_bucket>=8):921groups,762valid unused unique-image candidates after excluding60prior-used group entries and99duplicate-byte candidate entries. Ordered by sha256("CYCLE022:"+counterfactual_id), then ID; took first50. Selection was frozen before restoration/inference with SHA256 `4180324b305f72001a4793abc0b3f24b085531c4f4aa7c6dc5aa483962c70db5`. No difficulty, model output or outcome filtering. Two distinct miner/helmet identity metadata entries are retained in original pair order; all restored masks passed nonempty/shape checks before inference.

Local verification confirms train300, val50, Cycle003 diagnostic and Cycle009 old confirmation image hashes are all in the registry, and the50selected image hashes are unique with zero registry overlap. An additional scan of782local prior artifact files found no selected group/pair/hash/image-filename matches. This additional crosscheck did not change selection. Full receipts are used_image_registry.json, selection_receipt.json, local_audit.json and local_prior_material_crosscheck.json.

Restored50exact archived source images and200miner/helmet masks using the existing deterministic SAM-B recipe. All masks are explicitly reconstructed pseudo labels. Restored asset hashes, group/identity ordering, source image hashes, memory hashes and target hashes were frozen before model inference. No masks were hand-fixed. Freshness is with respect to the recorded Cycles001–021 iteration history; this is not an independent audit of every historical checkpoint pretraining example.

## Frozen methods and prediction-before-score audit

CMA reused the exact Cycle021 model initialization and build_item/predict_item path: unadapted base-w15, BF16,512tokens,v1_multiround,REF crop,seed0, unchanged per-group degradation seed and recipe. The loader block was extracted unchanged into load_base_w15; byte-for-byte block equality is recorded. All checkpoint file hashes matched Cycle021. Every forward asserts inference=True with all-zero target labels, and actual raster reads are restricted to source images and supplied memories.100batched group-condition forwards produced200identity predictions. No MCR score was computed.

SegLLM reused the unchanged50-group native runner, pinned code `4593a069f09628ce3a5b46e657f5417fefd7be46`, checkpoint revision `095e0637fcba015a02c0686f67b848c79c3cc80b`, existing two-gamma shim, native prompt `Segment the mining helmet worn by instance 1.[REF:1]`, historical index0, BOX/MASK encoding, native selection/threshold, final relational output and original-resolution mapping. All port file hashes match Cycle018; the gamma/tgt-embedding fidelity receipt passed.200native forwards, no retries. Model-specific native prompt syntax is unchanged; the semantic task is the same as CMA's frozen relational query.

Both methods share exact image pixels for each condition, identical supplied miner masks/bboxes and identity order. The CMA runner explicitly compared every actual condition image against the prepared SegLLM pixels. Within each A/B pair, the shared query/image stay fixed while memory identity changes.

All400raw predictions and hashes froze before target scoring. The scoring process verified both complete manifests, identity order and target-free raster audits before opening target-bearing manifests. It then verified target asset hashes and invoked the unchanged CMF scorer once per method/condition. Local replay independently matched all400prediction hashes. Asset preparation necessarily read targets for validity/hash freezing before inference; inference never read them, and metric scoring waited for both complete prediction freezes.

## Confirmation results

| Method | Condition | mIoU | CMSA | Memory Fidelity | IER | Mean margin | Median margin | Empty masks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CMA base-w15 |clean|90.87%|47/50 (94%)|98/100 (98%)|2/100 (2%)|0.892993|0.956758|0/100|
| CMA base-w15 |target15_b|68.02%|30/50 (60%)|87/100 (87%)|2/100 (2%)|0.657803|0.860244|0/100|
| SegLLM |clean|37.29%|3/50 (6%)|43/100 (43%)|38/100 (38%)|0.041568|0.000000|0/100|
| SegLLM |target15_b|27.52%|1/50 (2%)|38/100 (38%)|30/100 (30%)|0.016394|0.000000|0/100|

Paired CMA minus SegLLM on the same groups; rates in percentage points:

| Condition | mIoU delta | CMSA delta | Fidelity delta | IER delta | Mean-margin delta | Median-margin delta |
|---|---:|---:|---:|---:|---:|---:|
| clean |+53.57|+88|+55|-36|+0.851425|+0.956758|
| target15_b |+40.49|+58|+49|-28|+0.641409|+0.860244|

Paired target15_b minus clean:

| Method | mIoU delta | CMSA delta | Fidelity delta | IER delta | Mean-margin delta | Median-margin delta |
|---|---:|---:|---:|---:|---:|---:|
| CMA |−22.85 pp|−34 pp|−11 pp|0 pp|−0.235190|−0.096514|
| SegLLM |−9.77 pp|−4 pp|−5 pp|−8 pp|−0.025174|0|

Complete per-group IoU matrices and paired identity/group deltas are in scoring/comparison.json and the four unchanged CMF reports; comparison.csv provides the main numbers. No poor subgroup was rerun or removed.

## Interpretation and exactly one next recommendation

The large degraded advantage seen on val50 reproduces on this fresh subset:40.49points mIoU and58points CMSA, with49points higher Fidelity and28points lower IER. This supports the frozen supplied-entity-memory-use comparison under the fixed compound low-light/noise/blur stressor. It does not establish end-to-end memory writing, an adaptive agent, arbitrary underground-corruption generalization or pure architectural superiority: CMA and SegLLM have different training exposure, and targets are reconstructed pseudo labels. CMA's larger clean-to-degraded drop also means the result supports higher absolute degraded performance, not a smaller degradation sensitivity.

**Exactly one next recommendation: freeze method development and move to paper evidence tables/figures and bounded claim wording.** Layer2 stays retired regardless of this positive confirmation. No additional experiment or paper-writing cycle was started without review.

## Runtime, failures and artifacts

Run `20260918-213643-cma-cycle022-confirmation` executed21:39:19–21:45:17+08,exit0. CMA load-to-freeze53.14s; SegLLM219.93s. These timings are run metadata, not controlled efficiency comparisons. No inference failures or successful-prediction retries.54CPUtests passed22.15s, including16focused tests11.14s; exact loader-block equality, unchanged SegLLM port hashes and all real runtime assertions passed.

Intermittent SSH closes/timeouts interrupted launches and transfers. The earlier asset launch IDs213020/213059 were checked absent, with no process or run directory. The final existing213643 ID was started once through a single AutoDL helper connection, retaining run logs; it continued detached through later connection failures. No experiment was restarted because of lost connectivity.

Post-run SegLLM source/weight audit verified all3shards against pinned LFS SHA256 and the exact Git HEAD, with no tracked source modifications. An initial raw-source hash assertion differed because the transferred checkout uses CRLF; all5checked files match pinned source hashes after LF normalization. Both raw and normalized hashes are retained. No source/model change was needed. Existing untracked HIPIE/ancillary dependencies remain the prior documented setup.

The replay archive is1,344,694bytes, SHA256 `c137791efbe7436361660b217cdf4e0992263188eea9569a676352d003b5f688`, retained locally in outputs/cycle022_results.tgz and on A6000.400raw masks are under local outputs/cycle022_replay and remote outputs/cycle022. Source/restored assets remain remote shared/data/cycle022. Compact registry, selection, frozen protocol, model/code/weight receipts, predictions, scoring audit, metrics, logs and handoff are under research_log/cycle022. Input images/model weights are not added to Git.
