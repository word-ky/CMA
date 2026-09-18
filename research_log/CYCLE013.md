# Cycle013 — corrected MG-DRA integration and single experiment

2026-09-18 11:40 +08:00: synced99197d8 and mirrored review012 verbatim. Authorized alpha=1/Up=0 correction; historical double-zero test explicitly resets alpha to0, preserving the old blocker evidence. Corrected first/second-backward CPU test and feature mapping tests pass.

Reuse: base w15 model, full build_item inputs/REF text, SAM preprocessing, frozen get_visual_embs, Cycle006 train300/val50, original BCE+Dice/rank margin0.05, CMF scorer. New helper dual_scale_features only constructs auxiliary evidence from already degraded/clean observation and supplied miner bbox. No helmet information enters ROI/features. Existing target rows become zero shape placeholders during model calls; target arrays used only in external training loss/offline scoring.

Geometry: 1.25x memory ROI from Cycle010. Local SAM1024 padded encoder -> [1,256,64,64]. Remove local padded rows/columns using ceil(resized_valid_dimension/1024*64). Map full-image ROI into global resized pixel coordinates, divide by stride16, floor start/ceil end; bilinear resize valid local feature crop into this rectangle (align_corners=False), zero outside. Binary gate shares rectangle. Record exact tensors/rectangles per real smoke/eval identity. Boundary feature cells may partially overlap original padding; fixed ceil rule, no score-based choice.

Shared adapter Conv1x1(513,16)->GELU->Conv1x1(16,256),alpha1,Up0:12,577 parameters. FP32 adapter master/forward, frozen base BF16. Helmet rows1,3 get their identity's fused grid; miner rows0,2 get global grid. SAM decoder rows are evaluated separately using unchanged prompt/text state; no architecture modifications or MSP tokens. Disabled decoder batching stays unchanged. Extra2 frozen SAM encoder calls per two-identity group,1 per identity, plus normal1 global call. No features are cached across observations/groups.

Smoke: same first training group,pairedC/D; before optimization compare base vszero-adapter logits at fixed BF16 tolerance atol/rtol0.01 and record mask equality. Up nonzero first backward; one clippedAdamW smoke step; Down nonzero and alpha finite second backward; allbasegrads absent. Then zero A/B mapped features separately and require exact unchanged other-identity/miner logits. Discard entire process/state; training launched separately from pristine initialization only after receipt passes. One300-step run,AdamW1e-4,wd1e-4,clip1,seed20260528,onlyfinalcheckpoint. No additional losses/sweeps/diagnostic/confirmation.

11:45 +08:00: real smoke20260918-114421-cma-cycle013-smoke failed zero-init baseline tolerance before backward/optimizer: singleton decode batches changed BF16 outputs (max logit difference0.891598,11.6% elements outside0.01 tolerance). Training attempts0. Repair retains original full prompt batch shape for every decoder invocation; original global decode supplies miner rows, each identity's fused decode supplies only its own helmet row. No SAM architecture/weights change; extra decoder calls now2 per group (3 total). This supersedes the earlier singleton-row detail; output identity routing unchanged. Rerun smoke only after focused checks.

11:47 +08:00: repaired smoke20260918-114658-cma-cycle013-smoke-batch exit0. Exact zero-init logit difference0/masks identical. Up first gradients0.0509087(weight)/0.331616(bias). Second Down.weight0.00116841,alpha0.000552704; allfinite,basegradfree,localnograd. Perturb A: row deltas[0,0.982803,0,0]; perturb B:[0,0,0,0.603127]. One smoke step discarded in terminated process.38CPUtests pass. Proceed one pristine candidate run.

11:48 +08:00: formal run20260918-114821-cma-cycle013-full submitted (train300 -> valC/D -> analysis). On resumption inspect this run/log first; do not start another training run.


## CODEX UPDATE 013 — corrected MG-DRA has modest gains but fails the locked gate

Completed 2026-09-18. One corrected MG-DRA candidate trained for exactly300 groups from pristine base w15. Degraded mIoU improves0.691113→0.706802 (+0.015688),CMSA29/50→30/50,Fidelity89/100→94/100, with clean preserved. Nevertheless **the predeclared gate FAILS**: mIoU gain is below+0.02 and only1 failed group recovers rather than≥3. Accept the modest result without relabeling it as a passing mechanism. Per review, local-focus architecture search is now retired; no further alpha/ROI/bottleneck/LR/loss tuning and no scheduler/confirmation run.

### Exact production insertion and parameter budget

`memory_dual_scale_adapter.py`: Down1x1(513→16),GELU,Up1x1(16→256),learnable alpha. Up weights/bias zero;alpha1;normal seed20260528 Down initialization. Total **12,577** trainable parameters in5 tensors (Down8,224;Up4,352;alpha1). All existing w15 parameters are frozen, including LLM,CLIP,SAM image/prompt/mask modules and REF projections. Adapter forward/master parameters use FP32,base stays BF16.

`dual_scale_features.py`: crop the same observed image with the fixed Cycle0101.25x miner bbox. Apply the existing normal SAM1024 longest-side resize/padding and frozen/no-grad image encoder. Each local feature is[1,256,64,64]. Exclude bottom/right padded local cells using ceil(valid-resized-dimension/1024*64); transform original ROI by full-frame resize into the global padded64x64 grid, floor starts/ceil ends, bilinear resize (align_corners=False) into that rectangle. Zero elsewhere; binary G is[1,1,64,64]. This fixed feature-cell rounding may include partial boundary cells and was not tuned. Exact real-group rectangles/shapes are in the smoke and evaluation receipts.

`LISA.model_forward` accepts optional `memory_local_features_list`; its default remains disabled. Fused features are E_global+alpha*G*R. Only that identity's helmet row uses its fused grid; miner outputs retain baseline E_global. REF/text prompts, SAM/CLIP full-frame observation and auxiliary REF reconstruction remain unchanged. No MSP tokens, crop-only output, predicted/GT helmet geometry, enhancer or GT candidate selection. Training helmet labels are used only by external segmentation/rank losses; model target rows are shape-only zeros. Evaluation retains original labels for offline scoring.

An observed BF16 batching issue required one bounded implementation repair: the initial singleton decoder path failed zero-init equivalence (max logit difference0.891598,11.6% outside fixed tolerance) before any optimizer step. We retained the original complete prompt batch shape for the base decoder call and each identity-specific fused decoder call, then take only the corresponding helmet output from each fused call. This preserves original numerical shapes without modifying SAM architecture. Outputs from other identities in that invocation are discarded; original base call supplies miner rows.

### Software and real smoke receipts

38 CPU tests pass, including historical double-zero failure, corrected staged gradients, local-padding removal, feature-grid gate placement and identity tensor separation. Syntax checks pass. Real corrected smoke `20260918-114658-cma-cycle013-smoke-batch`,11:47:02–11:47:17 +08:00,exit0, used fixed group `cf_d0fa96077bf85e0a` with clean/degraded observations.

- Before optimization, max pre-threshold logit difference from base is **0.0**, masks exactly equal.
- First backward: Up.weight gradient norm0.050908681,Up.bias0.331615806;Down/alpha0 as expected.
- After exactly one discarded smoke AdamW step: Down.weight0.001168412,Down.bias0.000521088,alpha0.000552704;Up.weight0.049945794,Up.bias0.323296338. All finite.
- All frozen base gradients absent in both backwards; local features no-grad.
- Zero only A's mapped local tensor: rowwise maximum logit changes[0,0.982803345,0,0]. Zero only B: [0,0,0,0.603126526]. Other identity and miner logits stay exactly unchanged.
- Smoke process/state terminated; no smoke checkpoint was reused. The failed earlier smoke `20260918-114421-cma-cycle013-smoke` performed zero optimizer steps and is documented separately.

### One training run and inference cost

Exact launch: `bash "$ROOT/research_log/run_cycle013_full.sh" "$ROOT"`; training expands through `run_cycle013_train.sh` to `train_memory_dual_scale.py`. Formal run **20260918-114821-cma-cycle013-full**,11:48:26–12:01:58 +08:00,812s,exit0,includes300-step training and C/D validation. Seed20260528,exact Cycle006 train300 and unchanged val50; training order matches Cycle008 exactly. Paired clean/target15_b,AdamW lr1e-4,wd1e-4,clip1;loss0.5*(segC+segD)+0.5*(rankC+rankD),rank margin0.05. No consistency/teacher/auxiliary loss. Only the final checkpoint was saved/evaluated. Final alpha1.013619. Base gradients remain absent through training.

Cost: **1 additional frozen SAM image-encoder call per identity,2 extra per two-identity group** (3 total with the global call). To preserve numerical batching, main mask-decoder calls are also3 instead of1 per group; auxiliary REF reconstruction is unchanged. No separate CLIP/LLM calls are added. Runtime above includes training/evaluation and is not an isolated inference-latency benchmark.

### Fixed validation50, supplied-memory protocol

| Metric | Base C | Adapter C | Base D | Adapter D |
|---|---:|---:|---:|---:|
| target_miou | 0.919718 | 0.922217 | 0.691113 | 0.706802 |
| cmsa | 0.940000 | 0.960000 | 0.580000 | 0.600000 |
| memory_fidelity | 0.990000 | 0.990000 | 0.890000 | 0.940000 |
| mean_identity_margin | 0.919718 | 0.922217 | 0.662931 | 0.684793 |
| median_identity_margin | 0.963190 | 0.961184 | 0.847463 | 0.851052 |
| identity_error_rate | 0.000000 | 0.000000 | 0.030000 | 0.030000 |

Paired adapter-minus-base:

| Metric | clean | target15_b |
|---|---:|---:|
| target_miou | 0.002499 | 0.015688 |
| cmsa | 0.020000 | 0.020000 |
| memory_fidelity | 0.000000 | 0.050000 |
| mean_identity_margin | 0.002499 | 0.021862 |
| median_identity_margin | -0.002006 | 0.003589 |
| identity_error_rate | 0.000000 | 0.000000 |

Degraded CMSA transitions: **1 fail→pass,0 pass→fail,29 pass→pass,20 fail→fail**. Paired original image/query/seed/identity and exported target/reference masks match the base. Base C/D results reuse Cycle008; this run makes exactly one new final-adapter call per C/D condition on50 groups. Diagnostic30 and confirmation30 were untouched.

### Analysis-only whole-group oracle union

| Metric | GT oracle union (not deployable) | GT max-group-mIoU (not deployable) |
|---|---:|---:|
| target_miou | 0.710247 | 0.710247 |
| cmsa | 0.600000 | 0.600000 |
| memory_fidelity | 0.930000 | 0.930000 |
| mean_identity_margin | 0.688238 | 0.688238 |
| median_identity_margin | 0.849490 | 0.849490 |
| identity_error_rate | 0.030000 | 0.030000 |

The oracle selects a whole group's output using GT CMSA then GT group mIoU, ties base; the separate maximum-group-mIoU rule yields the same aggregate scores here. Union mIoU0.710247 is+0.019134 over base, with CMSA30/50. It does not create additional CMSA recoveries beyond the single adapter recovery. Supporting metrics are not all individually upper-bounded by this selection rule. This is only a diagnostic of possible action complementarity, never deployable performance or runtime selection, and it does not replace the failed gate.

### Gate and exactly one next one-hour recommendation

PASS: degraded CMSA nondecrease,≤1 regression,clean mIoU preservation andclean CMSA preservation. FAIL: degraded mIoU gain≥0.02 and≥3 recoveries. Overall **FAIL**. The result supports a small validation gain for this single trained adapter, not a stable/accepted degradation-recovery contribution. No fresh-confirmation result exists for MG-DRA; reconstructed pseudo-labels, small fixed validation and supplied-memory geometry limit the claim. No architecture freeze-as-success certificate is issued.

**Next one-hour task: define a fixed, reproducible comparative degradation-baseline evaluation plan around the validated identity-memory mechanism**, specifying common data, supplied-memory inputs, scoring and baseline compatibility before porting/running anything. Keep this as paper evidence preparation; do not invent another local recovery module or tune the rejected mechanisms. Await ChatGPT's next explicit task.

Final adapter SHA256 `2b0685fbd768e7bd0a287816038c9786ce96dfeaaf3bd5656607725129249369`; local/remote verified. Replay archive `research_log/cycle013_replay.tgz`,1,380,853 bytes,SHA256 `8b68cb22b45d8ee750ec4de3891e93cee10d47ad3ff284cdedf46bd2fa130fdd`; local/remote verified,includes final adapter,raw masks and logs. Compact evidence is under `research_log/cycle013/`.
