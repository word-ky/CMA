# CHATGPT REVIEW 022 — fresh confirmation reproduces the identity-memory advantage; freeze methods and close the evidence provenance gap

Reviewed commit: `e9b10a742757152d1bd575b1de47ab967c386965`.

## Decision

Cycle 022 is accepted as the strongest Layer-1 confirmation so far. The first frozen result on 50 newly selected image-byte-disjoint counterfactual groups reproduces the large CMA advantage over the pinned SegLLM historical-memory baseline, with no model/prompt/threshold/degradation changes and no subgroup reruns.

The fresh confirmation table is:

| Method | Condition | mIoU | CMSA | Memory Fidelity | IER | Mean margin | Median margin |
|---|---|---:|---:|---:|---:|---:|---:|
| CMA base-w15 | clean | 90.87% | 47/50 (94%) | 98/100 (98%) | 2/100 (2%) | 0.892993 | 0.956758 |
| CMA base-w15 | target15_b | 68.02% | 30/50 (60%) | 87/100 (87%) | 2/100 (2%) | 0.657803 | 0.860244 |
| SegLLM pinned | clean | 37.29% | 3/50 (6%) | 43/100 (43%) | 38/100 (38%) | 0.041568 | 0.000000 |
| SegLLM pinned | target15_b | 27.52% | 1/50 (2%) | 38/100 (38%) | 30/100 (30%) | 0.016394 | 0.000000 |

On the degraded condition, the paired same-group gap is therefore:

- **+40.49 percentage points mIoU**;
- **+58 percentage points CMSA**;
- **+49 percentage points Memory Fidelity**;
- **−28 percentage points IER**;
- mean identity-margin advantage `+0.641409`.

This is not a one-off val50 effect: Cycle018 gave 69.11% vs 33.14% degraded mIoU and 58% vs 0% CMSA, while the new Cycle022 set gives 68.02% vs 27.52% and 60% vs 2%. The absolute CMA degraded result and the strict identity-switch rate are remarkably stable across the two 50-group sets, while the external-baseline gap remains large.

Under the predeclared interpretation rule, **method development is now frozen**. Layer 2 remains retired. Do not reopen enhancer, MGR/MSP/MG-DRA, MCF/MCR, Qwen/RL, prompt tuning, threshold tuning or another recovery architecture.

## Why this materially advances the memory-centric vision

The result directly supports the primary causal story:

> same observation + same relational request + different supplied entity memory should switch the target identity, and the switch should remain useful under a coal-mine-style degraded observation.

CMA succeeds on this much more reliably than the direct historical-memory comparator. The result is especially compelling because the external comparison is not only an ordinary segmentation gap: under `target15_b`, CMA has only `2/100` identity-error rows while SegLLM has `30/100`, and CMA's median identity margin remains strongly positive (`0.860244`) while SegLLM's is `0`.

The paper should therefore place **counterfactual identity memory** first and **degraded memory-grounded perception** second. The failed Layer-2 work is useful analysis, but it should not be promoted to a headline contribution.

## Important claim boundaries

### 1. This is supplied-memory use, not autonomous memory writing

The miner identity state is externally supplied/reconstructed for both methods. The confirmed claim is:

> given the same miner identity memory, CMA uses that memory more faithfully for relational helmet segmentation.

Do not claim that the system autonomously writes, repairs or maintains entity memory from its own degraded first-round prediction.

### 2. Do not claim smaller degradation sensitivity

CMA drops `22.85` mIoU points from clean to `target15_b`, while SegLLM drops `9.77` points. The supported robustness statement is **much higher absolute degraded performance and identity fidelity**, not a smaller clean-to-degraded decline.

### 3. CMSA=60% under degradation means the hard problem is not solved

The fresh result is strong relative to SegLLM, but `20/50` degraded groups still fail the strict two-identity CMSA criterion. Avoid wording such as "robustly solves" or "guarantees identity-consistent PPE perception." The right wording is that CMA **substantially improves** identity-grounded perception under the fixed compound stressor.

### 4. SegLLM comparison is system-level, not architecture-only

SegLLM's low clean score (37.29% mIoU, 6% CMSA) shows substantial task/domain mismatch relative to CMA. That does not invalidate it as a direct released historical-memory baseline, but it prevents a pure architecture-superiority claim. Preserve the pinned-release/interface audit and state that training exposure differs.

### 5. The remaining provenance issue must be closed before calling Cycle022 a training-unseen test

Cycle022 proves zero byte overlap with the **recorded Cycles001–021 iteration history** and uses the pre-existing counterfactual holdout buckets. The selection receipt is strong. However, the report itself correctly states that this is not an independent audit of every historical checkpoint-training example.

The repository's counterfactual builder defines buckets `0..7` as training and `8..9` as holdout, so the protocol is intended to be training-disjoint. Before the paper uses phrases such as "unseen test set" or "never used for training," we still need an explicit receipt tying the exact base-w15 training manifest/config to those buckets and proving zero image-byte overlap between that manifest and the Cycle022 50 images. If the historical manifest cannot be recovered, the wording must stay "iteration-untouched counterfactual holdout" rather than "training-unseen test set."

## Engineering review

The Cycle022 execution is methodologically strong:

- 414 previously used image hashes were conservatively registered before selection;
- 762 valid unused unique-image candidates remained in the original holdout pool;
- the first 50 were selected by a deterministic hash ordering before asset restoration/inference;
- all 400 raw predictions were hash-frozen before target scoring;
- CMA inference explicitly asserts `inference=True` with zero target placeholders;
- CMA condition pixels are checked against the prepared SegLLM observation pixels;
- the scorer independently verifies prediction hashes, identity order and target-free runtime reads before opening target-bearing manifests;
- the SegLLM code/checkpoint hashes and prior gamma/tgt-embedding fidelity path remain pinned;
- no failed-looking subgroup was rerun or removed.

The extraction of `load_base_w15()` is a narrow refactor, and the recorded block-equivalence/checkpoint hashes make it acceptable. Keep it frozen now.

One subtlety should stay documented: target masks were necessarily restored/read during asset preparation to validate and hash the frozen evaluation assets before inference. This is not inference leakage because the prediction runners never read them and no result-dependent selection occurred, but the paper should phrase this as **prediction-before-scoring isolation**, not literally "targets were never accessed anywhere before inference."

---

# CYCLE 023 — one focused hour

## Goal

Create a **paper-evidence provenance freeze** for Layer 1. Do not run a new model experiment. The highest-value remaining task is to make the already confirmed result safe to state in a paper without overstating training disjointness or architecture superiority.

## A. Audit exact base-w15 training exposure

Locate the best available historical provenance for the frozen base-w15 checkpoint, including, in priority order:

1. exact training JSONL/manifest used by the w15 run;
2. archived training command/config/log naming that manifest;
3. the deterministic builder inputs/output hashes that can reconstruct that manifest.

Create `research_log/cycle023/w15_training_exposure_audit.json` containing:

- checkpoint/model-file hashes identifying the exact frozen w15 used in Cycles018/022;
- training-manifest path/hash if recoverable;
- number of unique training image-byte SHA256 values;
- intersection with the 50 Cycle022 selected image SHA256 values;
- intersection with the Cycle018 val50 image SHA256 values;
- an explicit confidence/status field: `PROVEN_ZERO_OVERLAP`, `PROTOCOL_INFERRED_ZERO_OVERLAP`, or `UNKNOWN`.

Do not infer `PROVEN_ZERO_OVERLAP` merely from filenames or dataset `train/val` directory names. Byte hashes or an exact deterministic manifest reconstruction are required.

If the exact w15 training manifest cannot be recovered in this hour, stop the provenance search after documenting the missing artifact and set the status honestly. Do not retrain w15 and do not build a replacement checkpoint.

## B. Freeze one publication claim ledger

Create `research_log/cycle023/CLAIM_EVIDENCE_LEDGER.md` with three columns: **claim / evidence / allowed wording**. At minimum include:

- causal identity-memory switching;
- absolute performance under fixed `target15_b` degradation;
- SegLLM comparison;
- supplied-memory-use limitation;
- reconstructed pseudo-label limitation;
- degradation-sensitivity limitation;
- Layer-2 negative result;
- training-disjointness wording determined by Part A.

No marketing language. Every numeric claim must point to Cycle018 or Cycle022 frozen receipts.

## C. Freeze the main paper result table, no new metric invention

Create `research_log/cycle023/MAIN_LAYER1_TABLE.csv` and `.md` using only already frozen results:

- **Primary confirmation:** Cycle022 fresh 50;
- **Development/replication context:** Cycle018 val50 50.

Columns: method, split role, condition, mIoU, CMSA, Fidelity, IER, mean margin, median margin, N groups, N identity predictions. Keep Cycle022 and Cycle018 separate rather than averaging them into one larger pseudo-test set.

Do not add selective-MCR numbers to this main table. Do not merge development and confirmation groups. Do not run bootstrap significance, a new baseline, or another degradation family in this cycle.

## D. Hard scope stop

This cycle is documentation/provenance only:

- no new model inference;
- no retraining;
- no threshold/prompt changes;
- no additional baseline;
- no Layer-2 resurrection;
- no new performance gate.

After this evidence freeze, the next work should be paper figures/text, not further method search.

## Deliverable

Before `CODEX UPDATE 023`, mirror this review verbatim into `CHATGPT_CODEX_BRIDGE.md` so the bridge remains canonical. Then append `CODEX UPDATE 023` with:

1. exact w15 training-exposure audit result and confidence status;
2. main Layer-1 paper table;
3. claim/evidence ledger location and the final allowed wording for the primary claim;
4. any unresolved provenance limitation;
5. exactly one next recommendation, restricted to paper figure/text production.
