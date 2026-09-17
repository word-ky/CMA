# CHATGPT REVIEW 008 — promising direct adaptation, but identity robustness is not yet stable

Cycle 008 is the first positive intervention that directly matches the locked Layer-1 story. The enhancer line is correctly closed; paired clean/degraded counterfactual adaptation of the existing w15 surfaces improves the fixed validation50 under `target15_b` while preserving clean identity behavior. On validation50, target mIoU rises `0.691113 -> 0.713416`, CMSA `0.58 -> 0.62`, Memory Fidelity `0.89 -> 0.93`, mean identity margin `0.662931 -> 0.698181`, and IER `0.03 -> 0.02`; clean mIoU/CMSA are also preserved or slightly improved. The training implementation is controlled: one fixed 300-group epoch, frozen LLM/CLIP/SAM image encoder, explicit trainable-surface receipt, non-zero finite gradients, and no enhancer/agent/GT candidate selection.

However, this is **not yet sufficient to claim robust identity-memory improvement**. The single diagnostic30 does not reproduce the validation CMSA gain: degraded mIoU improves only `0.631131 -> 0.639207` and mean identity margin improves `0.621451 -> 0.639008`, while CMSA drops `14/30 -> 13/30` and Memory Fidelity stays `49/60`. Clean behavior remains strong. Therefore the adaptation currently looks promising for degradation recovery in average mask quality, but the identity-switching gain is not yet stable across splits.

Two consequences follow.

1. Do not write the current result as an identity-robustness breakthrough. The safest statement is: **direct memory-grounded degradation adaptation improves validation performance and preserves clean identity behavior, but held-out CMSA stability remains unresolved.**
2. Do not move to the agent yet. The user explicitly prioritizes identity memory + degradation robustness over scheduling. An agent should be built on a Layer-1 model whose useful action space has real, repeatable upside.

There is also a plausible overfitting/stability risk: about 73.4M parameters are trainable from only 300 counterfactual groups. That does not invalidate Cycle 008—the clean-preservation result is encouraging—but it makes one fresh confirmation split more valuable than another broad architecture change.

The diagnostic30 has now been observed once and must no longer be used for candidate selection or tuning.

---

# CYCLE 009 — one-hour Codex task

## Goal

Answer one narrow question: **does the Cycle-008 direct adaptation produce stable identity-aware degradation gains, and is the degradation-consistency term helping that stability?**

This cycle allows exactly one additional training run (`lambda_cons = 0`) and one newly locked confirmation set. No other hyperparameter/architecture changes.

## Priority A — lock a fresh confirmation30 BEFORE the new training result is inspected

Before running the no-consistency candidate, construct and commit a new `confirmation30` manifest from counterfactual holdout groups that have never been used in:

- Cycle006 train300;
- Cycle006 validation50;
- Cycle003 diagnostic30;
- any Cycle008 training/model-selection operation.

Use deterministic source order only (prefer the next 30 eligible holdout groups after the existing diagnostic first30). Do not select by model score, degradation behavior, object size, or visual difficulty. Verify image-byte SHA disjointness against train300/val50/diagnostic30 and record provenance for recovered/regenerated masks exactly as in earlier cycles.

If fewer than 30 eligible complete groups exist, use the maximum deterministic eligible set only if it has at least 20 groups and record the shortfall; otherwise stop and report the data blocker rather than relaxing the protocol.

**Do not run any model on confirmation30 until candidate selection in Priority C is complete.**

## Priority B — exactly one consistency ablation

Train one candidate from the exact same original w15 initialization as Cycle008, using:

- the exact same train300 split and order/seed contract;
- the exact same trainable parameter surfaces;
- AdamW, lr `5e-6`, weight decay `1e-4`, clip norm `1.0`;
- one epoch / exactly 300 groups;
- the same paired clean + `target15_b` observations;
- the same clean/degraded BCE+Dice target loss;
- the same clean/degraded counterfactual rank loss with margin `0.05`;
- **only one change: set `lambda_cons = 0` and remove the clean->degraded consistency BCE term.**

No sweep, no restart, no intermediate checkpoint selection, no change to rank/seg weights, no change to trainable surfaces, and no diagnostic30 inference.

Record the same gradient/frozen-parameter checks as Cycle008. This is an ablation of the existing candidate, not a new method search.

## Priority C — choose between Cycle008 and no-consistency using validation50 only

Evaluate the new no-consistency checkpoint on the exact validation50 under clean and `target15_b`. Reuse the already recorded Cycle008 validation metrics for comparison.

Because **identity is the first priority**, use the following predeclared lexicographic selection rule:

1. Candidate must satisfy clean preservation: clean mIoU no more than `0.01` below base and clean CMSA no more than `0.02` below base.
2. Between eligible candidates, choose the one with higher degraded CMSA.
3. If degraded CMSA is equal, choose the one with higher degraded target mIoU.
4. If both are effectively equal (`|delta mIoU| < 0.002` after equal CMSA), keep Cycle008 to avoid needless candidate churn.

Do not consult diagnostic30 or confirmation30 for this selection.

Report the ablation as a mechanism result: whether `L_cons` helps, hurts, or is neutral for identity-aware degradation robustness on validation.

## Priority D — one fresh confirmation test, once

After Priority C has selected exactly one adapted candidate, evaluate only:

- unadapted base w15;
- the selected adapted candidate;

on the locked confirmation set under both clean and `target15_b`, no enhancer, one prediction per supplied identity memory, same CMF scorer.

Report:

- target mIoU;
- CMSA;
- Memory Fidelity;
- mean/median identity margin;
- IER;
- paired base->adapted deltas;
- CMSA transition counts by group: `fail->pass`, `pass->fail`, `pass->pass`, `fail->fail`.

Also report how many individual references change from correct-IoU `<0.5` to `>=0.5` and vice versa. This will distinguish threshold instability from true identity swaps.

### Fresh-confirmation acceptance rule

Treat direct w15 degradation adaptation as a stable Layer-1 component only if, on confirmation:

- degraded target mIoU improves by at least `+0.01` over base;
- degraded CMSA does not decrease;
- clean target mIoU drops by no more than `0.01`;
- clean CMSA drops by at most one group.

Memory Fidelity / identity margin / IER are supporting diagnostics, not post-hoc replacements for this gate.

If this gate passes, **freeze the selected Layer-1 checkpoint and stop tuning it**. The next cycle should move to the secondary contribution: a minimal oracle-free agent feedback/scheduling mechanism that exploits degradation recovery without GT.

If this gate fails, do not tune on confirmation30 and do not run another loss-weight sweep. Report the instability clearly; the next research decision should be based on the failure pattern rather than another blind adaptation attempt.

## Non-goals

- no enhancer work;
- no new degradation family;
- no LLM/CLIP/SAM-image-encoder unfreezing;
- no new memory taxonomy;
- no predicted-memory protocol;
- no controller/agent/verifier/RL yet;
- no baseline ports;
- no use of diagnostic30 or confirmation30 for model selection;
- no GT-based candidate selection.

## Deliverable

Before `CODEX UPDATE 009`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` because the review was committed as a separate file. Then append `CODEX UPDATE 009` with:

1. confirmation-set manifest/provenance/disjointness receipt;
2. exact no-consistency training command and gradient receipt;
3. Cycle008-vs-no-consistency validation table and the predeclared selected candidate;
4. one base-vs-selected fresh-confirmation C/D table plus CMSA transition audit;
5. pass/fail under the fresh-confirmation gate;
6. exactly one next one-hour recommendation.
