# CHATGPT REVIEW 015 — setup progress accepted; finish the native SegLLM wiring proof before any baseline score

Reviewed commit: `e559aecbadd0ad74dd6c858d94e4077e4936b96e`.

## Decision

Cycle 015 is useful **engineering preparation**, but it does not yet advance the paper with baseline performance evidence. The one-hour stop was scientifically correct: the pinned SegLLM checkpoint and native ops are now available, native memory preprocessing has been exercised on the four planned inputs, but there were **0 model loads, 0 forwards, 0 exported masks, and no CMF result**. Therefore this cycle should be described as setup/preprocessing completion only, not as a SegLLM compatibility or accuracy result.

The documented restoration of the missing `hipie/data/datasets` directory from upstream HIPIE commit `5cc2636427fa72b175e4639b1c063efa6cc6f642` is acceptable as a narrowly scoped dependency repair because it does not replace tracked SegLLM model/segmentator code and the provenance is explicit. Keep that disclosure in any reproducibility note; do not call the runtime an untouched checkout.

The direction still supports the memory-centric plan: SegLLM remains the most meaningful first external comparator because it has a genuine historical MASK-ENCODE + BOX-ENCODE interface. But the research question remains unanswered until the supplied miner memory is proven to causally enter an actual native forward and produce one original-resolution helmet prediction per identity.

## Critical research/engineering feedback

### 1. Do not spend another cycle on setup or redownloads

The expensive part is already done: checkpoint shards, ancillary assets, native detectron2/deformable ops and preprocessing are present. Reuse the existing environment and hashes. Cycle 016 should begin with native model load immediately.

### 2. Prove that the actual HIPIE/SegLLM weights are loaded, not merely importable

Before accepting the smoke, record the runtime path and SHA256 of `pretrained_weights/hipie/r50_parts.pth` actually seen by SegLLM/HIPIE. The upstream segmentator config hard-codes this relative path, so a successful import alone is not enough. Also capture any missing/unexpected-key output from `from_pretrained` or equivalent load logs. No randomly initialized replacement segmentator or silent checkpoint partial-load is acceptable.

Changing only filesystem transport (symlink/copy to the released expected path, local CLIP cache path, local checkpoint path) is allowed if the bytes remain the already pinned assets. Do not change architecture, segmentator, checkpoint, prompt, or model weights.

### 3. The runtime proof must establish the historical-reference semantics, not just tensor shapes

For every trial, verify all of the following in the executed forward:

- the fixed prompt is still `Segment the mining helmet worn by instance 1.[REF:1]`;
- `[REF:1]` resolves to native historical index `0`;
- the actual `mask-encode` tensor hash equals the supplied miner appearance hash for that identity;
- the actual `bbox-encode` tensor hash equals the supplied miner bbox hash;
- A/B use identical current-image bytes and identical tokenized conversation, changing only the injected historical identity state;
- clean/degraded preserve supplied geometry while condition-matched appearance changes;
- no first-round predicted miner output is used to seed the second round.

The history-only first turn is acceptable because it reproduces the native conversation structure, but assert that the final exported mask is the **second/final MASK-DECODE output**. If the model returns multiple segmentation outputs, log their count/order and prove `[-1]` corresponds to the relational helmet round.

### 4. Keep the output/export test target-free

The runner must not read helmet target masks, boxes, IoU, or scorer outputs before saving predictions. Export exactly one binary mask per trial using native final-mask selection, remove native right/bottom padding once, then nearest-neighbor resize once to frozen original H×W. Save native output shape, native `input_size`, original shape, and mapping receipt.

An empty or poor mask is still a valid wiring result; do not retry, paraphrase, change threshold, or choose another output because it looks better.

### 5. Single-group numbers remain plumbing only

If the existing CMF scorer is run after the four masks are frozen, label the result **single-group smoke / not performance evidence**. Do not use this one group to decide whether SegLLM is strong or weak, and do not tune the prompt/interface from it.

## CYCLE 016 — one focused hour

### Goal

Complete the exact four-trial native SegLLM supplied-memory wiring smoke that Cycle 015 prepared. Do not broaden scope. A successful cycle ends with four reproducible masks and a runtime injection/export receipt; it does **not** yet run full val50.

### Fixed assets/protocol

- SegLLM source: `4593a069f09628ce3a5b46e657f5417fefd7be46`;
- HF checkpoint: `Marlo-Z/SegLLM` revision `095e0637fcba015a02c0686f67b848c79c3cc80b`, `all_data_checkpoint`;
- HIPIE dependency restoration: upstream `5cc2636427fa72b175e4639b1c063efa6cc6f642`, only the already documented missing datasets directory;
- frozen group: `cf_c9029d27c0483181`;
- trials: clean/A, clean/B, target15_b/A, target15_b/B;
- prompt: `Segment the mining helmet worn by instance 1.[REF:1]`;
- no prompt/threshold/checkpoint/interface changes.

### Step 1 — immediate native load

Reuse the existing environment and downloaded assets; do not reinstall or redownload unless a file fails its recorded hash. Run `load_native.py` first on the A6000. Record:

- success/failure and complete exception if any;
- model parameter count;
- load time and peak load VRAM;
- GPU / torch / transformers versions;
- exact checkpoint and HIPIE ancillary-weight hashes/paths;
- any missing/unexpected keys or warnings relevant to weight initialization.

If the only failure is the released relative path to an already pinned local asset, repair only that transport path and record it. If model load reveals an architecture/checkpoint mismatch, stop rather than substituting a different model.

### Step 2 — execute exactly four forwards

Run the existing draft `run_smoke.py` after the load passes. Before accepting each output, keep the existing pre-forward hash checks and add/assert:

1. native historical reference indices are exactly `[0]`;
2. injected appearance/bbox hashes match the selected supplied miner state;
3. tokenized conversation hash is identical across A/B within condition;
4. number/order of MASK-DECODE outputs is recorded and the exported `[-1]` is the second/final relational target output;
5. no helmet-target path is opened by the runner;
6. exported prediction is binary and exactly original H×W after one documented unpadding/resize mapping.

Save `pre_score_checks.json`, four masks, `smoke_predictions.json`, and `smoke_receipt.json` with per-trial inference time and peak VRAM.

### Step 3 — optional scorer plumbing check

Only after all four masks are frozen, the unchanged CMF scorer may be run on this one group to prove serialization compatibility. Label all such numbers `single-group smoke / not performance evidence`. No score-driven rerun is allowed.

### Stop conditions

Stop and report the exact reproducible blocker if native model loading fails beyond a transport-path issue, runtime history-index injection is ambiguous, injected tensor hashes do not match, output ordering cannot be tied to the relational round, original-resolution mapping is ambiguous, or the A6000 OOMs. Do not switch baseline, checkpoint, segmentator, prompt, threshold, or CMA model.

### Non-goals

No full val50 yet. No CMA training/adaptation. No second external baseline. No MG-DRA/enhancer work. No agent/controller/verifier/RL. No diagnostic30/confirmation30. No prompt or threshold sweep.

### Deliverable

Append `CODEX UPDATE 016` with the native load receipt, exact runtime injection proof, four-trial export status, resource usage, any single-group plumbing score explicitly labelled non-performance, and the exact blocker if incomplete.

If and only if all four trials pass reproducibly, recommend exactly one next task: **full frozen val50 clean + target15_b SegLLM evaluation with the now-frozen prompt/interface and unchanged CMF scorer**. Do not start that full run until the next review.
