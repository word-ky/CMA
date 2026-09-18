# Cycle019 — fixed oracle-free MCF is adaptive but provides no gain

Implements REVIEW018 (`f033726`). **The fixed Layer-2 gate fails: zero CMSA recoveries.** 200 frozen REF predictions were reused;11 direct forwards were executed only for predeclared fallback images. All200 final masks are byte-identical to REF. No threshold/prompt/model changes followed scoring.

## Rule and implementation

`rule.json` was written before target access. The existing `helmet_geometry(pred_helmet, miner_mask)` and its bbox helper were moved unchanged from the v2 legacy controller into `scripts/helmet_geometry.py`; `geometry_reuse_receipt.json` proves exact function-source equality. The legacy v2 module re-exports those functions and retains its direct `ControllerRunner`. No LISA/model code or weights changed.

`memory_consistency_feedback.py` implements exactly: compute REF head score against supplied miner geometry; STOP_ACCEPT/REF_ACCEPT if score≥0.7; otherwise request DIRECT_FALLBACK once and choose direct only if its same-miner score is strictly greater; ties retain REF; then STOP_ACCEPT. The head score tests the predicted bbox center against the miner bounding box and upper45%/60% regions, not a learned anatomical or visual reliability model. The frozen direct query is `Please segment the mining helmet in the image.`

This is a new minimal oracle-free runtime path. It does not convert the old GT-assisted14-step executor into an oracle-free policy; that historical executor remains outside this evaluation. No enhancer, MGR, MSP, MG-DRA, Qwen, RL, new weights or SegLLM changes.

## Target-free planning and execution

The initial request list was built from the frozen Cycle008 REF manifests and Cycle018 condition-image/miner-memory paths, without helmet fields. The source CMA manifest hashes were checked. `actions_before_targets.json` freezes all200 REF scores/actions, hashes and counts before fallback inference:

| Condition | Immediate REF stop | Fallback requests | Extra direct calls after shared-image cache | Direct selected |
|---|---:|---:|---:|---:|
| clean | 98/100 (98%) | 2/100 (2%) | 2 | 0 |
| target15_b | 90/100 (90%) | 10/100 (10%) | 9 | 0 |

Both actions were exercised. All12 fallback comparisons had REF head_score0 and direct head_score0, so the predeclared tie rule retained REF. The 11 calls added cost without improving predictions.

Runtime uses the existing direct `ControllerRunner.predict` with a zero target array of image H×W and no reference input. A model pre-hook verifies inference mode, zero target placeholders and zero reference-valid flags at every forward. The controller's raster-read allowlist permits only frozen REF outputs, supplied miner masks, condition images and its own generated outputs; target/scorer-file reads raise. Both policy code and runtime traces contain no target IoU/CMSA/Fidelity inputs. An isolated test confirms target-mask and metric-file reads are rejected. All selected masks, action traces, direct candidates and hashes were frozen before a separate scoring process rechecked hashes and opened targets.

The first launch failed before inference because the adapter passed string`none` instead of Python`None` for the existing optional SAM initialization path. This argument conversion was corrected; no checkpoint/source/prompt/threshold was substituted. Successful run: `20260918-183325-cma-cycle019-native-direct`, finished18:33:46 +08, exit0,11 forwards. No successful prediction was rerun.

## Metrics and transitions

The unchanged CMF scorer gives identical REF-only and MCF metrics:

| Condition / policy | mIoU | CMSA | Fidelity | Mean margin | Median margin | IER |
|---|---:|---:|---:|---:|---:|---:|
| clean REF-only | 91.97% | 47/50 (94%) | 99/100 (99%) | 0.919718 | 0.963190 | 0% |
| clean MCF | 91.97% | 47/50 (94%) | 99/100 (99%) | 0.919718 | 0.963190 | 0% |
| target15_b REF-only | 69.11% | 29/50 (58%) | 89/100 (89%) | 0.662931 | 0.847463 | 3% |
| target15_b MCF | 69.11% | 29/50 (58%) | 89/100 (89%) | 0.662931 | 0.847463 | 3% |

| Condition | fail→pass | pass→fail | pass→pass | fail→fail |
|---|---:|---:|---:|---:|
| clean | 0 | 0 | 47 | 3 |
| target15_b | 0 | 0 | 29 | 21 |

## Analysis-only oracle headroom and its coverage limit

For each group, the GT oracle chooses one whole-group REF pair or the available shared direct prediction pair by CMSA then group mIoU, ties REF, matching the earlier project's whole-group oracle convention. The direct candidate is identity-agnostic and reused for both requests; this is only candidate-headroom analysis, not a direct model's claimed memory score or runtime selection rule. A separate maximum-group-mIoU oracle is also retained.

Both oracle variants choose REF everywhere and match REF-only: degraded mIoU0.691113/CMSA29/50, gain0 and zero recovery opportunities. Clean also has no measured gain. **Coverage is only2 clean and9 degraded condition images with scheduled direct candidates.** Other groups retain REF because their direct candidates were deliberately not run. This is not a full all-images direct oracle and cannot prove absence of complementarity among the unqueried images. No additional direct calls were made merely to complete that counterfactual.

## Fixed acceptance and exactly one recommendation

Degraded mIoU/CMSA preservation, at-most-one regression, both-action use and clean preservation all pass. The required≥2 CMSA recoveries fails (observed0). The measured candidate oracle meets neither≥0.03 mIoU gain nor≥5 recovery opportunities. The evaluated MCF primitive therefore does **not** qualify as a useful performance-improving agent component.

**One next recommendation: retire this evaluated geometry-triggered direct-vs-REF scheduling path and pivot to one bounded reliability/abstention evaluation on frozen predictions**, with its protocol specified by the next review. Do not tune this rule on val50 or train a verifier on the strength of the measured zero headroom. This recommendation concerns the tested path; it is not a universal claim that every unmeasured direct action lacks value. No next-cycle evaluation or new policy has started.

## Artifacts and checks

`research_log/cycle019/` contains fixed rule/requests, pre-target actions, final prediction/action freeze, read audit, geometry-source equivalence, runtime receipts, full load/failure logs and per-group CMF/scoring/oracle choices. Raw11 direct and200 selected masks remain in the project-local server directory and local replay output. Full replay archive472,394 bytes, SHA256`1ad41f6dadf65dbcfae21315dc50c2c55333eb967683d1f0b10fe2dafe70d09a`; all fetched mask hashes verified, and all200 selected-file hashes equal their frozen REF-file hashes.

Six focused tests passed in1.18s (head/loose-head stop, better fallback, tie/worse retention, changed memory changing the action, empty REF recovery, forbidden target/scorer reads). Changed scripts parse; actual11 forward placeholder checks and post-freeze hashes passed. Direct-call inference/postprocessing/save time totaled3.493s; peak allocated GPU memory17,257,539,072 bytes. REF computation was reused, so these are extra-call engineering costs, not an end-to-end latency comparison.
