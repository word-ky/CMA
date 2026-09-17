# Counterfactual memory evaluation — Cycle 001 contract

This is an **offline saved-prediction evaluator**. It loads no LISA/Torch model, changes no weights and does not choose predictions using GT. Ground truth is used only after prediction for scoring. Metric names/definitions below are local Cycle 001 conventions for ChatGPT review, not claims of standardized literature metrics.

## Input

Python >= 3.10, NumPy; Pillow for PNG masks. One JSONL row per same-image/same-query/condition group:

```json
{"group_id":"cf_001","image_id":"frame_001","query":"Segment the helmet worn by this miner.","condition":"darkness_level_2","memory_source":"predicted_fixed_write","trials":[{"entity_id":"miner_A","prediction":"masks/pred_A.png","target":"masks/helmet_A.png"},{"entity_id":"miner_B","prediction":"masks/pred_B.npy","target":"masks/helmet_B.npy"}]}
```

- `image_id` identifies the exact common observation. `query` is the exact shared query string. `condition` is a string describing the fixed degradation preset/severity; clean can be `clean`. Use identical degradation realization/seed across the identities in a group.
- `memory_source` explicitly distinguishes `gt_reference`, predicted memory and repaired memory protocols. It is an experimental label, not inferred from scores. A memory written using GT selection must not be labeled oracle-free.
- `entity_id` is a unique local miner identity. Each trial contains the prediction conditioned on that memory and that identity's corresponding helmet GT. Input order sets the row and column order in the output matrix.
- `prediction`/`target` accept relative or absolute paths to 2D `.npy` or single-channel image masks, or inline 2D arrays. Relative paths resolve against the manifest's directory. Values must be 0/1 or 0/255. Threshold model logits **before** export using the inference protocol (`>0` in the current LISA evaluator). No resizing or coordinate correction occurs here.
- At least two distinct identity targets are required; every target must be nonempty and all masks share the same original-image shape. Predictions may be empty. Identical target masks, repeated identities or repeated group/condition/source records cannot support the intended diagnostic and are rejected.
- The producer must guarantee identical image, query, model, decoding configuration and tool budget within the group. The saved-mask evaluator cannot verify that provenance or establish causality from an arbitrary JSON manifest. Store production receipts when running actual inference. Do not export oracle-best-of-rollout masks as if they were ordinary model predictions.
- Query/condition are shared group fields, not per-trial variations. For a matched degradation curve reuse the same entity groups at every severity and document any exclusions; this scorer does not manufacture missing groups or enforce matched sampling.

## Metrics

Let `S[i,j] = binary IoU(P_i, G_j)`, where `P_i` is produced with memory `i`, and `G_j` is identity `j`'s target. Let `d_i=S[i,i]`, `w_i=max(j!=i) S[i,j]`. This full matrix generalizes the old evaluator's binary IoU. The old evaluator chooses only one wrong target for N>2; here all wrong targets count, matching the training rank comparison's scope (but using binary IoU, not training soft IoU).

1. **Target mIoU:** mean `d_i` over reference trials. Not global intersection/union.
2. **Memory Fidelity:** fraction with `d_i > w_i`, strict. Ties and all-zero rows fail. This is rank fidelity, so a tiny correct fragment can pass it; report alongside mIoU and CMSA.
3. **CMSA:** for each unordered pair `(i,j)`, both rows must have strict global fidelity **and** `d_i,d_j >= min_iou` (default 0.5). Average over all pairs. For N>2 each row must beat every group distractor, not only the paired identity. This quality-qualified definition intentionally does not credit tiny fragments. It is stricter than the historical rank-only counterfactual success metric. Report the threshold; do not compare the two names as if identical.
4. **Identity Error Rate:** fraction of all reference trials with `w_i > d_i` and `w_i >= min_iou`. It measures a mask sufficiently matching a known wrong identity in the same category. It is not `1 - fidelity`: empty masks, low-IoU failures and rank ties are not classified as confident wrong-identity errors. Unknown distractors outside the group are not detected. A tied best score between multiple wrong targets still counts if each beats the correct target.

The report includes per-reference matrix/decisions and per-pair decisions. Overall scores are reference-weighted, except pair-weighted CMSA. Stratified summaries use `(condition, memory_source)`; these should be used for curves/comparisons instead of mixing GT-reference and predicted-memory results. Overall metrics across different sources are descriptive only. CMSA pairs within a group are correlated; no independence or significance claim is made.

## Run

```bash
python cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py \
  --input research_log/fixtures/cmf_synthetic.jsonl \
  --output research_log/cycle001_synthetic_metrics.json --min-iou 0.5
```

This fixture is explicitly synthetic: one correct pair and one memory-ignored pair. Its report verifies software behavior and is **not a model experiment**.

## Missing for actual model/degradation evaluation

The published repo has code and aggregate Stage-3 receipts, not a grouped set of saved per-memory predictions and corresponding target masks. `eval_mr_ref_counterfactual_v0.py` currently writes scalar diagonal/one-wrong scores and visualization JPEGs; it does not export raw prediction masks. Visualization JPEGs cannot be used as exact masks. A full N>2 matrix cannot be reconstructed from those scalars. The regular holdout30 receipt is not a substitute for a multi-identity counterfactual set.

The next inference/export step should:

1. Resolve existing `counterfactual_id`, `image_path`, `same_round2_query`, `pair_ids` and the miner/helmet mask paths from the archived counterfactual/pair manifests. Preserve ordered pairs.
2. Use the frozen w15 model to generate and save one binary original-coordinate prediction per memory; write the manifest above. GT-reference and predicted-reference runs must be separate. Target GT is for scoring only, not selecting a candidate.
3. Record checkpoint, source split, image/degradation seed, exact prompt and memory-write protocol. Start with clean and one existing darkness/blur preset on matched groups; dust/glare/occlusion measurements are not implemented or claimed in Cycle 001.
4. Run this scorer, inspect wrong-identity and ignored-memory cases, then ask ChatGPT to review metric definitions and failures before adding a verifier or changing training.

## Standalone memory API

`cmllm_remote/scripts/entity_memory.py` stores one entity's mask, exclusive-right/bottom pixel bbox, source image, optional appearance/semantic features, nullable reliability, provenance and version.

```python
from entity_memory import from_anchor_state

memory = from_anchor_state(state, entity_id="miner_A", source_action="SEG_ANCHOR", step=9)
original = memory.read()
candidate = memory.update_candidate(
    new_mask, new_image,
    {"source_action": "SEG_ANCHOR_LOCAL", "image_state": "enhanced_local", "step": 10},
)
restored = memory.rollback(step=11)  # new version; restores mask, image, bbox and features
```

`write` initializes; `read(version=...)` returns a deep copy; `update_candidate` records a new unverified current candidate under the same bookkeeping identity and invalidates old features/reliability by default. It makes no acceptance decision. `rollback` restores a prior snapshot into a new monotonically increasing version without deleting history. Explicit `version=...` chooses an older snapshot. Returned/initial arrays and dictionaries are copy-isolated so later mutation does not silently rewrite history.

The legacy adapter copies `anchor` with `anchor_image` and `anchor_state`, computes bbox, leaves reliability unset and records `selection_uses_gt=true`; it does not copy `anchor_quality['iou']` as reliability. This does not purify an oracle-selected anchor. Caller-provided source action/step must correspond to the original anchor write, not simply the current action.

This is in-process persistence, not a disk database, encoder, learned reliability estimator or identity recognizer. Identity preservation must still be measured; storing a stable string does not guarantee the new mask depicts the same person. The API has not been integrated into the old executor in this cycle.

## Cycle 002 extension

The scorer additionally exports per-reference `identity_margin = correct_iou - max_wrong_iou` and its reference-weighted mean/median in overall and stratified reports. Other definitions are unchanged. The old Cycle 001 receipt is retained; the new synthetic margin check is in `cycle002/synthetic_margin_check.json`.

The previously missing raw-mask export seam now exists behind `eval_mr_ref_counterfactual_v0.py --export-memory-manifest`, with `--condition clean|target15_b`. This supersedes the Cycle 001 statement that the script can only save scalars/overlays. Both conditions explicitly use `memory_source=supplied_ref`. Real inputs remain incomplete; see [Cycle 002](CYCLE002.md) for asset audit, frozen group IDs, exact missing paths and the deployed command template. No real model scores yet.
