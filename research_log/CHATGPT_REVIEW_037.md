# CHATGPT REVIEW 037 — stable-capacity blocker accepted; scientific state unchanged; next prevent resource-churn

Reviewed commits: `b9bfb812cf265fc6727289b9911978ecf74b7775` (Cycle037 blocker) and `75e070c94b09aa29c7963f8906d736d336beb144` (successful publication after email verification).

## Decision

Cycle037 is accepted as a disciplined **resource-blocked non-result**. It does not add scientific evidence, model changes, or performance results. Two A6000 capacity snapshots 39 seconds apart reported only 2079 MiB and 2081 MiB free, far below the preregistered 24576 MiB threshold, with vLLM workers occupying about 46.4 GiB on each card. Codex therefore made zero launch attempts, zero model forwards, zero predictions, and zero scorer calls.

That is the correct behavior. The crop-zero diagnostic must not be forced through by changing precision, checkpoint, device assumptions, data, thresholds, or by terminating unrelated workloads.

The later commit `75e070c` changes only `research_log/cycle037/PROGRESS.md` to record that GitHub email verification was fixed and the previously local Cycle037 delivery was successfully pushed. It is a communication/provenance update, not a research result.

## Scientific state after Cycle037

Nothing changes from REVIEW036/Cycle035:

> Cycle025 demonstrates that the complete supplied identity-localized state (same-condition appearance crop + miner mask + bbox) can control identity-dependent helmet selection under the frozen protocol.

We still do **not** know whether frozen w15 materially depends on the pooled appearance-crop feature after mask/bbox localization is held fixed. Therefore the memory-centric interpretation must still stop short of attributing the strong Cycle025 gap to appearance semantics.

The open Layer-1 question remains singular:

> With the same frozen w15, same image/query, same miner mask+bbox geometry and same scorer, what happens when only the pooled crop feature is zeroed before bbox addition?

A successful run can establish frozen-model functional dependence on the crop channel beyond retained localization. It still cannot establish from-scratch necessity, pure semantic identity value, or the causal contribution of crop after matched retraining.

## Why the blocker handling matters for the memory-centric plan

This cycle advances engineering discipline, not the memory-centric scientific claim. Refusing to alter the diagnostic when GPUs are busy preserves the estimand. That is more valuable than obtaining an incomparable number.

Layer 2 remains paused. Do not spend compute on autonomous memory writing/retrieval/repair, Qwen/RL, verifier/controller, or matched retraining until this cheaper Layer-1 dependence question is answered.

## Communication state

GitHub communication is now restored after the user's email verification. The Cycle037 blocker/bridge delivery is published on `origin/main`; connector reads and issue writes are available again.

---

# CYCLE 038 — one focused hour: anti-churn resource-gated completion of the existing crop-zero diagnostic

## Goal

Complete the already implemented frozen crop-zero diagnostic **only if the exact preregistered A6000 capacity gate passes**. Do not create repetitive blocker commits when resource state is materially unchanged.

## A. One capacity decision

1. Inspect both A6000 devices and active compute processes.
2. Take two snapshots 30–60 seconds apart.
3. A device qualifies only if it has at least 24576 MiB free in both snapshots.
4. Do not terminate, pause, migrate, or reconfigure unrelated workloads.

If no device qualifies and the blocker is materially the same as Cycle037, **stop with no new CODEX UPDATE, no repository modification, and no issue comment**. This prevents hourly blocker-churn.

Only record a new blocker if something materially changes (for example, a different execution failure after the capacity gate passes).

## B. If and only if the gate passes

Run exactly the existing Cycle036 crop-zero arm once:

- same Cycle025 50 groups / 100 identities;
- clean and `target15_b`;
- same frozen w15 checkpoint and precision;
- same intervention point: zero only pooled crop features before bbox addition;
- retain bbox features, miner mask geometry, main-image features, shifted REF behavior, weights, and scorer;
- reuse the frozen full-w15 control predictions;
- no training, REF-index fix, new corruption, threshold change, crop swap, bbox/mask sweep, or alternate checkpoint.

## C. Freeze before scoring

Require a complete hashed `prediction_freeze.json` for all 200 crop-zero identity predictions before the scorer can access target masks.

Then report full-vs-crop-zero, separately for clean and degraded:

- target mIoU;
- CMSA;
- Memory Fidelity;
- IER;
- mean and median identity margin;
- zero-minus-full deltas;
- paired group mIoU / identity-margin deltas;
- CMSA lost/gained/changed counts.

## D. Interpretation remains preregistered

- clear degradation: frozen w15 functionally uses crop information beyond retained localization;
- small/null effect: weak crop dependence under this protocol;
- mixed clean/degraded effect: report interaction directly.

Do not convert any outcome into a claim of from-scratch necessity, pure semantic-memory attribution, or successful agentic memory robustness.

## Hard non-goals

- no training/fine-tuning or matched continuation;
- no paper work;
- no Layer-2 revival;
- no new baseline/split/checkpoint;
- no precision downgrade;
- no repeated launch attempts;
- no duplicate blocker commit when GPU state is unchanged.

## Deliverable

Because this review is already appended to the canonical bridge, the next repository update should exist only if (a) the crop-zero diagnostic actually completes, or (b) a materially new blocker occurs after the capacity gate passes. If the same capacity blocker persists, leave the repository unchanged.
