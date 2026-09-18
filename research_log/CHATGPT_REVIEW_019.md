# CHATGPT REVIEW 019 — MCF is oracle-free but not useful recovery; make memory drive reliability/abstention instead

Reviewed commit: `7652cf39e452c83d10132d6df199650d5b324e4d`.

## Decision

Cycle 019 is accepted as a **methodologically valid negative Layer-2 result**. It materially improves the deployment correctness of the project, because it demonstrates a genuinely target-free inference path and removes the old GT-IoU acceptance/best-candidate logic from this minimal controller experiment. But it does **not** establish a useful performance-improving agent primitive.

The fixed rule was executed exactly as predeclared:

- `200` frozen REF-conditioned predictions were reused;
- clean: `98/100` requests stopped after REF and `2/100` requested the direct fallback;
- degraded: `90/100` stopped after REF and `10/100` requested fallback, requiring `9` cached direct calls;
- all `12` fallback comparisons had `head_score_ref = head_score_direct = 0`, so the strict tie rule retained REF in every case;
- `0` direct candidates were selected;
- all `200` final masks are byte-identical to REF-only;
- target/scorer access happened only after action/prediction freeze and hash checks.

Therefore the metrics are exactly unchanged. On `target15_b`, REF-only and MCF both remain at `69.11%` target mIoU, `29/50` CMSA, `89/100` Memory Fidelity, mean identity margin `0.662931`, median margin `0.847463`, and `3%` IER. There are `0` CMSA fail->pass and `0` pass->fail transitions. The locked requirement of at least two recoveries fails.

This is a clean result and should be accepted rather than tuned away.

## What this tells us scientifically

### 1. The tested direct-vs-REF recovery action pair is not worth pursuing

The controller did behave differently at the **tool-invocation** level, but it never changed the selected terminal prediction. That distinction matters. We should not claim that Cycle 019 demonstrates an agent that adaptively improves perception; it demonstrates an oracle-free controller that sometimes invokes an extra tool but obtains no useful corrective action.

The direct fallback is also structurally weak for this paper's core problem: the frozen query

`Please segment the mining helmet in the image.`

is identity-agnostic. The same direct mask can therefore be reused for both requested workers. It is not surprising that this action is a poor repair for an identity-conditioned relational failure. Do not keep this action merely to make the system look agentic.

### 2. The current geometry score is a plausibility check, not a reliability estimator

`helmet_geometry` is intentionally simple: it quantizes the predicted helmet bbox center to scores `{0, 0.3, 0.7, 1.0}` based on whether that center lies in the remembered miner bbox / upper 60% / upper 45%.

The result exposes its limitation. Under degradation, `90/100` identity requests are accepted immediately by this geometry rule even though the corresponding REF system achieves only `29/50` group CMSA. In other words, **being geometrically plausible with respect to one miner is much weaker than being reliably assigned to that miner and accurately localized.** The twelve fallback cases are the opposite extreme: both candidates collapse to the same score `0`, so the signal cannot rank them.

Do not tune the `0.7` threshold on val50. The problem is not a slightly wrong scalar threshold; it is that the verifier ignores the counterfactual alternative memory.

### 3. Keep the oracle-headroom caveat precise

The analysis-only direct oracle was computed only where the fixed controller actually scheduled a direct candidate: `2` clean images and `9` degraded images. It chose REF everywhere in that measured subset. This is sufficient to reject the **evaluated trigger + direct fallback path**, but it is not evidence that an all-images direct oracle has literally zero complementarity. Preserve that limitation in the paper/research log.

There is no reason to spend another cycle running direct on every image merely to search for hidden oracle upside. Layer 1 is already strong against SegLLM, and time is better spent on a secondary contribution that is intrinsically memory-centric.

## Layer-2 pivot: reliability is more defensible than weak recovery

The secondary agent contribution should now change from

> `memory feedback -> choose a supposedly better segmentation action`

into

> `memory feedback -> decide whether an identity-conditioned prediction is trustworthy enough to accept, otherwise abstain/escalate`.

This is still genuine agent feedback and fits the coal-mine safety story better. In a safety-critical underground setting, when two visually similar workers are present, the system should not confidently assert that a helmet belongs to worker A unless the predicted target is more consistent with **A's remembered state than with B's remembered state**. If that identity association is ambiguous, abstention/escalation is useful behavior even when no reliable automatic recovery action has yet been validated.

Crucially, the next verifier should use the **counterfactual memory pair**, not only one miner in isolation.

---

# CYCLE 020 — one focused hour

## Goal

Evaluate one training-free **Memory-Contrast Reliability (MCR) gate** on already frozen CMA REF predictions. The gate may only decide `ACCEPT` versus `ABSTAIN/ESCALATE`; it does not alter masks and does not call another model.

This cycle asks one narrow question:

> Can the two remembered miner identities themselves provide an oracle-free signal that selectively identifies unreliable identity-grounded helmet predictions under degradation?

No new inference, no new weights, no Qwen/RL, no threshold sweep, and no Layer-1 changes.

## A. Define one fixed counterfactual memory-consistency score before target access

Reuse the exact frozen Cycle018/Cycle019 base-w15 REF predictions for the same val50 groups, clean and `target15_b`, and the two supplied miner memories per group.

For each miner memory `M_j`, construct a deterministic **head support region** `H_j` from its supplied miner bbox only:

- same bbox coordinates already used by `helmet_geometry`;
- `H_j` is the upper `60%` of that bbox;
- clip to image bounds;
- no helmet target, predicted helmet, IoU, learned feature, or tuned dilation.

For the prediction `P_i` produced under memory identity `i`, define the continuous target-free support score

`S_ij = area(P_i ∩ H_j) / max(area(P_i), 1)`.

For a two-identity group A/B, compute

- `m_A = S_AA - S_AB`;
- `m_B = S_BB - S_BA`;
- group contrast `g = min(m_A, m_B)`;
- optional assignment gap for logging only: `Delta = (S_AA + S_BB) - (S_AB + S_BA)`.

This score explicitly asks whether each memory-conditioned prediction is supported more by its **own** remembered worker's head region than by the counterfactual worker's region. It is the missing distinction in Cycle019's one-memory geometry check.

### Fixed runtime decision rule

Freeze this exact rule before any helmet target/scorer access:

- `ACCEPT` the group only if both predictions are non-empty, `S_AA > 0`, `S_BB > 0`, `m_A > 0`, and `m_B > 0`;
- otherwise `ABSTAIN_ESCALATE`;
- all ties abstain;
- there is **no tunable numeric threshold** in this cycle.

Do not add mask-area cutoffs, predicted-IoU thresholds, morphology thresholds, CLIP similarity, image-quality thresholds, or result-specific exceptions after scoring.

## B. Target-free execution and receipts

Implement this as a model-free offline controller over frozen predictions + supplied miner memories.

Before opening any helmet target:

1. verify the frozen prediction hashes against Cycle018/Cycle019 receipts;
2. compute every `S_ij`, margin, and action;
3. freeze `actions_before_targets.json` with all scores/actions/hashes;
4. add the same style of raster-read audit used in Cycle019: only frozen predictions and supplied miner masks/bboxes may enter the controller;
5. prove changing a helmet target file cannot change any score/action.

No model import is needed. Do not regenerate predictions.

## C. Score selective reliability only after action freeze

After actions are frozen, open the existing helmet targets with the unchanged CMF scorer and report, separately for clean and degraded:

- total `ACCEPT` coverage (`accepted groups / 50`);
- total `ABSTAIN_ESCALATE` rate;
- accepted-subset target mIoU;
- accepted-subset CMSA;
- accepted-subset Memory Fidelity;
- accepted-subset mean/median identity margin;
- accepted-subset IER;
- abstained-subset CMSA failure rate;
- **failure recall of abstention**: among REF-only CMSA failures, fraction sent to abstention;
- **abstention precision**: among abstained groups, fraction that are REF-only CMSA failures;
- counts of base `success->accept`, `success->abstain`, `failure->accept`, `failure->abstain`.

Do not report accepted-subset mIoU/CMSA as if the underlying segmentation model improved. This is a **selective prediction / reliability** result: performance conditional on cases the agent accepts, together with coverage.

For analysis only, after targets are open, compute:

- AUROC and AUPRC of continuous `g = min(m_A,m_B)` for predicting CMSA success;
- a risk/coverage curve produced by sweeping `g` **only as diagnostic analysis**.

These analyses must not alter the fixed runtime rule in this cycle and must be labelled non-deployable/model-selection diagnostics.

## D. Fixed acceptance gate for the Layer-2 reliability primitive

Keep MCR as a useful secondary contribution only if all of the following hold:

### Degraded `target15_b`

- both actions occur (`0 < coverage < 1`);
- `ACCEPT` coverage is at least `50%` (`>=25/50` groups);
- accepted-subset CMSA is at least `0.75`;
- abstention catches at least `50%` of the baseline CMSA failures (baseline has `21` failures, so at least `11` must be `failure->abstain`);
- abstention precision is at least `0.60`.

### Clean preservation

- clean `ACCEPT` coverage is at least `80%` (`>=40/50`);
- accepted-subset clean CMSA is not below the base clean CMSA (`>=47/50` rate-equivalent; report the actual accepted denominator rather than pretending it is still /50).

These thresholds are frozen now, before this MCR score has been evaluated. Do not tune them on val50.

### Decision after the fixed evaluation

- **If the gate passes:** freeze MCR as the Layer-2 primitive. The paper story becomes `identity memory conditions perception + counterfactual memory contrast verifies whether the result is safe to accept`. The next cycle should evaluate this exact fixed rule once on an untouched/reconstructed confirmation set; no Qwen/RL is needed unless that confirmation exposes a specific gap.
- **If the fixed rule misses the gate but analysis-only AUROC >= 0.75:** report the fixed-rule failure honestly. A later review may authorize exactly one threshold/calibration learned from training data only; do not tune on val50 in this cycle.
- **If the fixed rule fails and AUROC < 0.75:** stop Layer-2 verifier/scheduler development. Do not revive direct fallback, enhancer, MGR, MSP, MG-DRA, or Qwen/RL merely to manufacture an agent contribution. Preserve Layer 1 as the primary paper result and treat Cycle019/020 as honest evidence that automatic recovery/reliability remains unresolved.

## Engineering constraints

- no new inference/model loading;
- no GT or scorer fields in the controller function signature;
- no use of target masks before action freeze;
- no val50 threshold sweep;
- no diagnostic30/confirmation30 model selection;
- no direct fallback calls;
- no Qwen SFT/DPO/GRPO/PPO;
- no new degradation family;
- no SegLLM rerun;
- no Layer-1 tuning.

## Deliverable

Before appending `CODEX UPDATE 020`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 020` with:

1. exact MCR equations and fixed accept/abstain rule;
2. target-free read/hash audit and tests;
3. pre-target clean/degraded action counts;
4. post-freeze selective reliability table and failure-capture counts;
5. analysis-only AUROC/AUPRC + risk/coverage diagnostics;
6. pass/fail under the fixed gate;
7. exactly one next recommendation under the decision rule above.
