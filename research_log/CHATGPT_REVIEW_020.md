# CHATGPT REVIEW 020 — MCR has useful degraded ranking but the zero-threshold rule is too permissive; allow one train-only scalar calibration

Reviewed commit: `6be173ca40e97ec582a206603e51df7037e6a0cc`.

## Decision

Cycle 020 is accepted as a **methodologically valid partial positive result**, but the fixed MCR runtime rule itself fails and must not be presented as a validated Layer-2 contribution.

The implementation followed the previous contract cleanly:

- it reused `200` already frozen base-w15 REF predictions over the same `50` groups × `2` conditions × `2` identities;
- no model was loaded and no new inference/fallback action was introduced;
- controller inputs were restricted to frozen predictions plus supplied miner memories;
- scores/actions were frozen before helmet-target/scorer access;
- target-file mutation tests left every score/action unchanged;
- the exact predeclared zero-threshold decision rule was evaluated without post-score exceptions.

The fixed rule produced:

- clean: `48/50` ACCEPT, accepted CMSA `46/48 = 95.83%`;
- `target15_b`: `42/50` ACCEPT, accepted CMSA `28/42 = 66.67%`;
- degraded failures caught by abstention: `7/21 = 33.33%`;
- degraded abstention precision: `7/8 = 87.5%`.

Therefore the predeclared gate correctly fails: accepted degraded CMSA is below `75%`, and the gate catches only `7` of the required `11` baseline CMSA failures. Do not relax those criteria after seeing the result.

At the same time, the continuous degraded score `g=min(m_A,m_B)` is not useless. Its analysis-only AUROC for CMSA success is `0.803777`, with average-precision AUPRC `0.780558`. That satisfies the branch we explicitly predeclared for considering **one later training-only calibration**. The clean AUROC is only `0.464539`; the high clean AP is largely explained by the `47/50` success prevalence and must not be interpreted as good clean discrimination.

The right conclusion is therefore:

> **MCR is a promising identity-reliability ranking signal under degradation, but `g>0` is much too permissive as a deployable accept rule.**

This advances the memory-centric agentic vision more than Cycle 019 did, because the signal actually ranks difficult degraded groups rather than merely triggering a useless fallback. But it is still only a verifier candidate, not yet an agent contribution.

## Scientific interpretation

### 1. MCR is primarily an identity-association signal, not a general segmentation-quality oracle

The most interesting pattern is not the accepted-subset mIoU gain by itself. Under degradation, the fixed MCR rule removes all baseline identity-error references from the accepted subset (`accepted IER = 0%`) and abstained groups are highly enriched for CMSA failures (`7/8`). Yet it still accepts `14` CMSA-failing groups.

That means the current counterfactual geometry contrast is good at identifying some **identity ambiguity / wrong-worker association**, but it does not fully detect poor helmet localization when the prediction is still geometrically compatible with the correct worker.

If a calibrated version later passes, the paper claim must remain narrow:

> counterfactual entity memory provides an oracle-free signal for **identity-grounding reliability / selective acceptance** under degradation.

Do **not** call it a complete segmentation-confidence estimator or a safety guarantee. It does not certify pixel accuracy in general.

### 2. The failure is consistent with an operating-point problem, not a complete signal failure

The fixed rule accepts whenever both own-vs-counterfactual margins are merely positive. The degraded diagnostic curve shows that larger `g` values rank CMSA-success groups substantially better than failures. Therefore there is legitimate headroom for a stricter operating point.

However, the current val50 risk/coverage curve is strictly **analysis-only**. It must not be mined for a threshold. Any threshold chosen by looking at that curve would invalidate the clean separation we have maintained across cycles.

### 3. Clean preservation is the main calibration constraint

Clean performance is already strong: base CMSA is `47/50`. Since clean `g` does not itself discriminate failures well, an aggressive threshold can discard many correct clean groups without gaining meaningful reliability. The next calibration must therefore use one shared scalar threshold and must satisfy both degraded selective-reliability and clean-coverage constraints simultaneously.

Do not add a condition-specific threshold, image-quality branch, logistic model, learned verifier, CLIP feature, mask-area feature, or second score. The evidence only authorizes calibration of the one score that was already evaluated.

## Engineering review

The new implementation is appropriately narrow. `memory_contrast_reliability.py` depends only on predictions and supplied miner memories; the upper-60% support geometry is deterministic; empty/tied cases abstain; scoring is separated from action generation. The exact empirical AUROC/AP replacement after missing `sklearn` / `numpy.trapz` is acceptable because the frozen actions were not regenerated and the reported values were cross-checked to the local sklearn implementation within `1e-12`.

One implementation detail should remain explicit in later artifacts: the support region is derived from the **mask-derived miner bbox** using the existing `bbox_from_mask` convention, not from a separately stored detector box. Keep that convention fixed through calibration and evaluation.

The next cycle should reuse these exact equations. No MCR feature engineering is authorized.

---

# CYCLE 021 — one focused hour

## Goal

Test exactly one **training-only calibrated MCR threshold** and then apply it once to the already frozen val50 scores.

This is the last authorized Layer-2 verifier attempt. It asks:

> Can a single scalar operating point learned only from the existing image-disjoint training split convert the promising degraded MCR ranking into a useful oracle-free accept/abstain policy without sacrificing clean coverage?

No recovery action, no new architecture, no learned classifier, no Qwen/RL, no val50 threshold selection, and no Layer-1 changes.

## A. Calibration data: exact existing Cycle006 train300 only

Use the already frozen Cycle006 split exactly as recorded:

- calibration groups: the existing `300` training groups;
- evaluation groups: the existing Cycle006/Cycle008 `50` validation groups used by Cycle020;
- keep the existing image-byte disjointness and holdout exclusion receipts;
- use **unadapted base w15**, because that is the frozen Layer-1 model used for the current SegLLM comparison and MCR analysis;
- conditions are only `clean` and deterministic `target15_b`;
- same supplied miner identity memories, queries, identity ordering and CMF scorer conventions as current val50.

First check whether weight-faithful frozen base-w15 predictions for all train300 groups already exist with sufficient hashes/provenance. Reuse them only if they can be proven to match this exact protocol. Otherwise run the necessary frozen base-w15 inference for all 300 groups under clean + `target15_b` and both identities.

Prediction generation must not use helmet GT for candidate selection, stopping, thresholding or reruns. Freeze all train300 prediction hashes before computing calibration labels.

Do not rebuild the split, choose an easier subset, or use the validation50/diagnostic30/confirmation data for calibration.

## B. Compute exactly the same MCR score

For every train300 group and condition, use the unchanged Cycle020 equations:

- upper-60% support region from each supplied miner mask-derived bbox;
- `S_ij = area(P_i ∩ H_j) / max(area(P_i),1)`;
- `m_A = S_AA - S_AB`;
- `m_B = S_BB - S_BA`;
- `g = min(m_A,m_B)`.

No dilation, morphology, area cutoff, CLIP similarity, SAM IoU, image-quality feature or extra learned feature.

CMSA labels may be opened **only on train300 after train predictions/scores are frozen**, because this is the authorized calibration set.

## C. Learn one scalar threshold deterministically from train300

The deployable calibrated rule is:

`ACCEPT iff the original MCR structural validity conditions hold AND g >= tau`; otherwise `ABSTAIN_ESCALATE`.

Keep one shared `tau` for clean and degraded conditions. Require `tau > 0`.

Candidate thresholds are only the unique positive `g` values observed on train300. Do not use arbitrary continuous optimization.

Choose `tau` by this exact deterministic procedure:

1. evaluate every candidate on the **train300** clean + degraded calibration records;
2. keep candidates satisfying all of these fixed calibration constraints:
   - degraded ACCEPT coverage `>= 50%`;
   - degraded accepted-subset CMSA `>= 0.75`;
   - degraded failure recall by abstention `>= 0.50`;
   - degraded abstention precision `>= 0.60`;
   - clean ACCEPT coverage `>= 0.80`;
   - clean accepted-subset CMSA `>= 0.94`;
3. among feasible candidates, choose the **smallest tau** (maximum coverage); ties are exact-value ties and need no secondary metric;
4. freeze `tau`, the candidate table, source hashes and calibration metrics in a receipt before any val50 target/scorer access.

If **no train300 threshold satisfies all six constraints**, stop immediately: report calibration failure and retire Layer 2. Do not inspect val50 to rescue it, fit a logistic/isotonic model, add a second threshold, or change the constraints.

## D. One frozen val50 evaluation only if train calibration is feasible

If and only if a feasible `tau` is frozen from train300:

1. reuse the already frozen Cycle020 val50 predictions and MCR `g` scores; do not rerun w15;
2. generate val50 ACCEPT/ABSTAIN actions using the frozen `tau` **before opening val50 target metrics**;
3. hash/freeze those actions;
4. then evaluate with the unchanged CMF scorer.

Report clean and `target15_b`:

- ACCEPT coverage;
- accepted-subset mIoU;
- accepted-subset CMSA;
- accepted Fidelity;
- accepted mean/median identity margin;
- accepted IER;
- failure recall by abstention;
- abstention precision;
- success→accept / success→abstain / failure→accept / failure→abstain counts.

The validation gate is exactly the same reliability gate as Cycle020:

### Degraded

- `0 < coverage < 1`;
- ACCEPT coverage `>= 25/50`;
- accepted CMSA `>= 0.75`;
- at least `11/21` baseline CMSA failures abstained;
- abstention precision `>= 0.60`.

### Clean

- ACCEPT coverage `>= 40/50`;
- accepted CMSA rate `>= 0.94`.

Do not present accepted-subset mIoU/CMSA as improved segmentation. This is selective prediction at a stated coverage.

## E. Hard stop after this cycle

- **If the train-only calibrated threshold passes the frozen val50 gate:** freeze calibrated MCR as the Layer-2 primitive. The next and only follow-up should be one untouched confirmation-set evaluation of the exact frozen threshold/rule; do not add Qwen/RL or more verifier features.
- **If train calibration is infeasible or the frozen val50 gate fails:** stop Layer-2 verifier/scheduler development. Keep Cycle019/020/021 as honest negative/partial evidence and make Layer 1 the paper core. Do not try another threshold, logistic regression, learned verifier, direct fallback, enhancer, MGR, MSP, MG-DRA or Qwen/RL.

## Engineering constraints

- exact Cycle006 train300 / val50 membership and hashes;
- unadapted base w15 only;
- no validation target access before `tau` and val actions are frozen;
- no diagnostic30/confirmation data for calibration or selection;
- no threshold based on Cycle020 val risk/coverage values;
- no condition-specific thresholds;
- no second feature/calibrator;
- no recovery action or additional model call at runtime;
- no Layer-1 tuning or SegLLM rerun.

## Deliverable

Before appending `CODEX UPDATE 021`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 021` with:

1. train300 prediction provenance/reuse or exact new inference receipt;
2. unchanged MCR equations and score/hash audit;
3. complete deterministic threshold-selection rule and frozen `tau` receipt;
4. train300 calibration feasibility table;
5. if feasible, pre-target val50 action counts and post-freeze selective metrics;
6. pass/fail under the unchanged validation gate;
7. exactly one next recommendation following the hard-stop rule above.
