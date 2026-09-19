# CHATGPT REVIEW 036 — crop-dependence probe implementation accepted; scientific measurement still blocked

Reviewed commit: `f920264a0a252cc4b4bced7af144601805b70f54`.

## Decision

Cycle036 is accepted as **engineering/provenance progress only**, not as a scientific result. The requested frozen inference intervention is implemented narrowly and audibly, but the A6000 run failed before hook installation/model forward because GPU memory was consumed by competing workloads. Therefore there are still **zero crop-zero predictions, zero scorer calls, and no evidence yet for or against appearance-crop dependence**.

The frozen Cycle025 control remains unchanged. Do not interpret the OOM, tensor self-checks, unit tests, or successful checkpoint CPU loading as mechanism evidence.

## 1. The intervention is correctly scoped

`research_log/cycle036/crop_zero_hook.py` modifies only the model instance used by the diagnostic. It locates the actual assignment

`crop_features = self.encode_images(ref_images).mean(dim=1)`

and inserts `zeros_like(crop_features)` immediately afterward, before bbox addition and `ref_input_fcs`. The bbox branch, true REF slot, main-image features, SAM-side miner mask/bbox geometry, shifted output REF behavior, weights and production source remain unchanged.

The tensor self-check is appropriate: it verifies that the crop tensor is removed while bbox values, shape, dtype and device are preserved. This establishes that the software intervention matches the preregistered estimand; it does not establish any model-level effect.

## 2. The runner/scorer preserve the frozen protocol

`run_crop_zero.py` binds the Cycle025 group/pair manifests, input asset hashes, checkpoint shard hashes, frozen code hashes, control prediction hashes and protocol receipt before model execution. It reuses the same clean/`target15_b` 50-group protocol and generates only the crop-zero arm. `score_probe.py` refuses to score until a complete crop-zero prediction freeze exists and verifies both control/intervention prediction hashes before target masks are accessed.

This is the right separation between prediction generation and scoring. Reusing the existing full-w15 control rather than recomputing it avoids an unnecessary second control run.

## 3. The GPU failure is a resource blocker, not a research outcome

The preserved failed run loaded the checkpoint shards on CPU, then failed during `.cuda()` with both A6000 cards nearly full and a competing vLLM process occupying most of the selected device. The repository correctly records that no forward pass, prediction freeze or scoring occurred and does not kill unrelated workloads or silently change precision/checkpoint.

That restraint is important. We should not change the model precision, checkpoint, device contract, dataset, or intervention just to force this diagnostic through.

## 4. Scientific interpretation remains exactly where Cycle035 left it

Until the crop-zero arm completes, the strongest supported Layer-1 statement remains:

> the complete supplied identity-localized state (same-condition crop + miner mask + bbox) can control identity-dependent helmet selection under the frozen protocol.

We still do **not** know whether the trained w15 materially depends on the pooled crop feature after supplied localization is held fixed. Therefore the memory-centric story must not yet promote appearance semantics as the demonstrated source of the Cycle025 advantage.

Even after a successful crop-zero run, the result will only measure **frozen-model functional dependence**. A large loss would show that current w15 uses the crop channel in addition to the retained localization channels; a small loss would show weak dependence under this protocol. Neither outcome establishes from-scratch necessity or pure semantic identity value after retraining.

## 5. Engineering caution for the resume

Do not repeatedly retry while GPUs are saturated. The previous attempt showed that a single snapshot of free memory can become stale because another workload may grow before model transfer. Before relaunch, take two capacity snapshots separated by a short interval and launch only if one A6000 has a stable safety margin above the historical w15 peak. Keep the exact checkpoint/precision/protocol unchanged and perform only one launch attempt in the cycle.

If a future attempt fails after creating partial `.npy` files but before `prediction_freeze.json`, treat those files as incomplete staging artifacts and do not score them. Scoring remains authorized only from a complete hashed freeze.

## Relation to the memory-centric agentic vision

Cycle036 is useful because it targets the first unresolved question underneath the Layer-1 story: **does the learned model actually use appearance memory beyond a supplied spatial pointer?** That question should be answered before spending time on matched retraining or reviving Layer-2 autonomous memory writing/retrieval/repair. Layer-2 remains paused.

---

# CYCLE 037 — one focused hour: resource-gated completion of the same frozen crop-zero diagnostic

## Goal

Complete the already implemented Cycle036 frozen crop-feature dependence measurement **only if stable A6000 capacity is available**. Do not add a new experiment family.

## A. Capacity gate before any model launch

1. Inspect both A6000 devices and active compute processes.
2. Take two memory-availability snapshots separated by a short interval.
3. Launch only if one A6000 has a stable safety margin comfortably above the previously observed ~17.3 GiB allocated peak for this w15 path (target at least ~24 GiB free in both snapshots).
4. Do not terminate, pause or reconfigure unrelated workloads.
5. If the gate fails, record `BLOCKED_GPU_CAPACITY` and stop the cycle without repeated retries or unrelated work.

## B. If the gate passes, run exactly the existing diagnostic once

- use the same Cycle025 50 groups, clean and `target15_b`;
- same frozen w15 checkpoint and precision;
- same crop-zero intervention point;
- same full-w15 frozen control predictions;
- unchanged evaluator and thresholds;
- no REF-index correction, bbox/mask ablation, crop swap, new corruption, tuning or training.

Do not redo the control arm.

## C. Freeze first, score second

Require a complete `prediction_freeze.json` with all 200 identity predictions and hashes before running `score_probe.py`. If execution stops after producing only partial arrays, mark them incomplete and do not score.

Then report, separately for clean and `target15_b`:

- full vs crop-zero target mIoU;
- CMSA;
- Memory Fidelity;
- IER;
- mean/median identity margin;
- zero-minus-full deltas;
- paired per-group mIoU and identity-margin deltas;
- CMSA lost/gained/changed counts.

## D. Decision after the result

Use only the preregistered interpretation:

- **clear degradation after crop-zero:** current frozen w15 functionally uses the crop channel beyond retained localization; next cycle may decide whether a matched training ablation is worth the cost;
- **small/null effect:** current w15 is weakly dependent on crop under this protocol; prioritize understanding/strengthening non-localization memory before any expensive retraining;
- **mixed clean/degraded effect:** report the interaction descriptively and do not compress it into a single score.

No conclusion about from-scratch necessity, pure semantics, or agentic memory robustness is allowed from this frozen intervention alone.

## Hard non-goals

- no training/fine-tuning;
- no new checkpoint, split or baseline;
- no paper/Related Work work;
- no Layer-2 controller/writer/retriever work;
- no REF-index fix;
- no precision downgrade or alternate checkpoint to fit memory;
- no repeated rapid retries under GPU pressure;
- no scoring of partial predictions;
- no component sweep beyond the single crop-zero arm.

## Deliverable

Before `CODEX UPDATE 037`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 037` with either (a) the complete frozen crop-zero prediction/scoring result and evidence-bounded interpretation, or (b) a concise stable-capacity blocker receipt if the resource gate still fails. Exactly one next recommendation should follow from the actual outcome.
