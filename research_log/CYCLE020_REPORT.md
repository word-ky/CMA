# Cycle020 — fixed MCR fails; degraded contrast ranking supports a later training-only calibration review

Implements REVIEW019 (`446c19a`). **No model imports/loads, new forwards, changed masks or fallback calls.** All100 group actions were computed on200 frozen REF predictions and supplied memories, then frozen before targets/scoring. The fixed gate fails; the analysis-only degraded AUROC0.803777 reaches the predeclared0.75 branch for considering training-only calibration later. No threshold was selected on val50.

## Fixed equations and geometry

`memory_contrast_reliability.py` uses the identical `bbox_from_mask` coordinates used by existing `helmet_geometry`, derived from each supplied miner mask. This resolves the existing code's mask bbox convention explicitly; the stored detector bbox is not substituted. Each head region H is the bbox's upper60%, clipped to the image, rasterized by pixel centers inside the half-open rectangle. No contour-dependent support, learned feature, morphology or dilation is added after that bbox extraction. This raster convention was written into the input spec before computation/scoring.

For prediction P_i and miner j, S_ij=area(P_i∩H_j)/max(area(P_i),1). Margins are m_A=S_AA−S_AB and m_B=S_BB−S_BA, with g=min(m_A,m_B); their sum is logged as assignment gap. ACCEPT requires both predictions nonempty, both own supports positive, and both margins strictly positive. Otherwise ABSTAIN_ESCALATE; every tie abstains. All rules, hash sources and fixed acceptance criteria are in `cycle020/input_spec.json`.

## Target-free execution and independence

The model-free runner verified prediction/miner hashes against Cycle019 receipts before reading arrays. Its raster allowlist contains only those frozen predictions and supplied miner masks; no image, helmet target or scorer artifact enters the controller. The function signature is `decide(predictions, miner_memories)` with no GT/scorer fields. Scores, actions, source hashes and actual reads were frozen in `actions_before_targets.json` before separate scoring. The scoring process rechecked hashes and records action-freeze time before scorer-start time.

Before target access the counts were clean48/50 ACCEPT and2/50 ABSTAIN; degraded42/50 ACCEPT and8/50 ABSTAIN. Both pure-function and actual offline-runner synthetic tests change a helmet target file from all-zero to all-one and prove every group score/action stays identical, with that file absent from read logs. This is an independence test, not a model performance result.

## Selective reliability results

These are **conditional accepted-subset metrics plus coverage**, not improvements to the underlying segmentation masks.

| Condition | ACCEPT coverage | ABSTAIN | Accepted mIoU | Accepted CMSA | Accepted Fidelity | Mean margin | Median margin | Accepted IER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 48/50 (96%) | 2/50 (4%) | 92.91% | 46/48 (95.83%) | 96/96 (100%) | 0.929103 | 0.963190 | 0/96 (0%) |
| target15_b | 42/50 (84%) | 8/50 (16%) | 73.89% | 28/42 (66.67%) | 80/84 (95.24%) | 0.738727 | 0.855417 | 0/84 (0%) |

| Condition | success→accept | success→abstain | failure→accept | failure→abstain | Failure recall | Abstention precision / abstained CMSA-failure rate |
|---|---:|---:|---:|---:|---:|---:|
| clean | 46 | 1 | 2 | 1 | 1/3 (33.33%) | 1/2 (50%) |
| target15_b | 28 | 1 | 14 | 7 | 7/21 (33.33%) | 7/8 (87.5%) |

The degraded gate flags a high-error subset and removes all3 baseline identity-error references from acceptance, but still accepts14 CMSA-failing groups. Geometry contrast is therefore insufficient as the fixed acceptance rule for localization correctness. Base clean/degraded full-set mIoU91.97%/69.11% and CMSA47/50/29/50 are unchanged.

## Analysis-only ranking and risk/coverage

Positive class is CMSA success; score is frozen continuous g. No diagnostic threshold is selected or fed back to runtime. Tied scores enter/leave together in the risk/coverage curve; it does not invent an ordering within ties.

| Condition | AUROC | AUPRC (average precision) | Trapezoidal PR area | Positive prevalence |
|---|---:|---:|---:|---:|
| clean | 0.464539 | 0.925608 | 0.944484 | 47/50 (94%) |
| target15_b | 0.803777 | 0.780558 | 0.831819 | 29/50 (58%) |

AUPRC is reported primarily as non-interpolated average precision; trapezoidal area is separately labelled to avoid silently mixing definitions. The clean ranking is weak and has only3 failure groups; high clean AP is not evidence of discrimination when success prevalence is94%. Degraded ranking is promising under the fixed decision branch but remains development-set analysis, not a calibrated safety guarantee.

Full curve points are in `scoring/risk_coverage.csv` and the JSON report. `scoring/risk_coverage.png`/`.svg` mark the fixed rule separately from diagnostic thresholds. The plot was visually checked; no operating point was chosen from it.

## Fixed gate and exactly one next recommendation

Pass: both degraded actions, degraded coverage≥50%, abstention precision≥60%, clean coverage≥80%, clean accepted CMSA≥94%.

Fail: degraded accepted CMSA66.67% is below75%; only7 failures abstained, below the required11. **MCR is not accepted as a validated fixed Layer-2 contribution.**

Because degraded diagnostic AUROC0.803777≥0.75, **the single next recommendation is a review-authorized, one-threshold/calibration experiment learned from training data only**, with frozen evaluation and clean preservation specified in that review. Do not fit or choose that threshold on val50, do not start confirmation-set selection now, and do not revive fallback/recovery actions. No calibration or next-cycle work has been performed.

## Validation, environment and artifacts

Six focused tests passed in1.90s: own-vs-swapped identity, empty/tied abstention, exact upper60% support fraction, pure target-file independence, tie-aware diagnostic AUC formulas, and actual offline-runner target-file independence. Real100-group hash/read/action-freeze checks passed. No new inference or dependencies were installed.

Remote scoring initially stopped on missing sklearn, then on unavailable numpy.trapz. The final analysis uses exact pairwise AUROC with half credit for ties, threshold-group average precision and an explicit trapezoidal sum. All six real-data AUC/AP values match the existing local sklearn1.5.1 implementation within1e-12 (`diagnostic_crosscheck.json`). These were analysis implementation repairs only; the runtime MCR scores/actions remained frozen throughout.

Durable artifacts are under `research_log/cycle020/`: input spec, frozen scores/actions, scoring-start audit, complete base per-group IoU matrices, selective report, continuous diagnostics, plots/CSV, tests and receipts. Result archive SHA256 `d8df5e53777645ebaa2aceb8c3f75ded0007637337c94d3c14bde36e925badec` is retained in local outputs and A6000 project research_log. Original predictions remain unchanged in their previous frozen locations. This remains supplied-memory use on reconstructed pseudo-labelled development val50, not untouched generalization or end-to-end memory writing.
