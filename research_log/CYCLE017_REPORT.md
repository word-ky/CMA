# Cycle017 — weight-faithful four-trial SegLLM wiring smoke passed

Implements REVIEW016 (`e9e256a`), starting 2026-09-18 16:47 +08. **Exactly four native forwards completed and four original-resolution binary masks saved. This is single-group wiring evidence, not baseline-performance evidence.** No scores were computed; no full val50 run started.

## Exact weight restoration

`cmllm_remote/external_baselines/segllm/weight_fidelity.py` restores only the two explicitly authorized gamma parameters after native loading and the existing BF16/device conversion. It reads the pinned safetensors keys, copies their deterministic casted values into identically named model parameters, and records raw/expected/actual byte hashes plus `torch.equal`. No other parameter assignment occurs in this helper. The native loader requests `output_loading_info=True`; the observed missing/unexpected sets must match the known two-gamma collision exactly.

Both gamma_l[768] and gamma_v[256] passed exact post-cast equality. The final target embedding [100,256] independently equals the outer SegLLM checkpoint after casting, **without assignment by the helper**. Thus the earlier ancillary900-to100 warning does not leave this final tensor unmatched. Full native warnings remain visible, and native dependency versions are unchanged. Both the standalone health load and the actual smoke process wrote successful weight-fidelity receipts.

Standalone load: `20260918-164738-cma-cycle017-weight-load`, 19.460785s, peak allocated14,581,610,496 bytes, 7,271,583,752 parameters, A6000 GPU0. Torch2.3.1+cu121, Transformers4.46.1. Model/source/checkpoint and the disclosed HIPIE datasets dependency restoration remain the previously pinned versions. The only new inference-launch environment setting is CUDA_HOME pointing at the existing Cycle015 CUDA12.1 prefix.

## Runtime intervention and output proof

Frozen group `cf_c9029d27c0483181`, native prompt `Segment the mining helmet worn by instance 1.[REF:1]`, seed0, supplied miner identities in frozen order. The native history placeholder is text only; no predicted-miner first-round call is made.

- All four native dataloader conversations resolve `[REF:1]` to historical index `[0]`.
- A/B within each condition have identical source-image hashes, rendered conversation and input token hashes. Condition-matched miner appearance/bbox is the varying state; degraded appearance changes while supplied geometry stays fixed.
- Executed model pre-hooks assert appearance and bbox bytes equal the intended supplied state after native replacement. Actual injected hashes are saved in each prediction record.
- Conversations contain MASK-DECODE at turns1 and3 (zero-based), with two ordered supervised segmentation token positions. At the executed forward there are exactly two mask-decode jobs for the same image; returned mask_data follows that job order and exactly two output masks are present. Pinned native source extracts hidden states in token order, preserves that order within the single image, and HIPIE emits one selected mask per prompt in order. Selecting output index1 (`[-1]`) therefore selects the final relational turn3, not the history placeholder. No model/output-selection implementation was modified.
- Native output1024×1024 is cropped once to576×1024 then resized once by nearest neighbor to720×1280. Each saved NPY is uint8 binary with that exact shape. Native predicted-score selection and thresholding remain unchanged.
- Runtime read audit records only the two prepared condition images and two supplied miner masks as raster inputs. No helmet target is read; configuration JSON reads are retained in the full receipt. The separate local artifact audit read only saved predictions after completion. No scorer or target-driven retry occurred.

## Execution and resources

First runner launch `20260918-164923-cma-cycle017-four-trials` stopped in Trainer construction before any model forward because DeepSpeed could not locate CUDA_HOME. Pointing the launch environment to the already installed compiler prefix resolved it; no new compilation, package install, source repair or dependency change was made. The final launch `20260918-165009-cma-cycle017-cuda-path` completed16:50:38, exit0. The successful four forwards were not repeated.

| Trial | Seconds including native postprocessing/export | Peak allocated GPU bytes |
|---|---:|---:|
| clean/A | 1.167150 | 16,456,006,656 |
| clean/B | 0.431247 | 16,485,084,160 |
| target15_b/A | 0.444873 | 16,488,704,512 |
| target15_b/B | 0.444785 | 16,495,790,080 |

These are one-group, warm-process timings including postprocessing/file export; not a throughput benchmark or pure kernel latency. GPU peaks include resident model/input allocations. Empty/poor masks would have been accepted under the same fixed selection; no qualitative selection or score inspection was used.

## Artifacts, checks and reproduction

`research_log/cycle017/` contains `weight_fidelity_receipt.json`, `smoke_weight_fidelity_receipt.json`, `native_load_receipt.json`, `pre_score_checks.json`, `smoke_predictions.json`, `smoke_receipt.json`, complete logs, and `artifact_audit.json` with per-mask file/array hashes and archive hash. Four masks are retained under the A6000 project `research_log/cycle017/smoke/` and locally under `outputs/cycle017/smoke/`; the compressed transfer archive is `outputs/cycle017_smoke.tgz`. No weights or input images are added to Git.

From the pinned source checkout with ROOT set to the existing A6000 project:

```bash
CUDA_HOME="$ROOT/external_baselines/segllm-cuda121" CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/external_baselines/segllm" HF_ENDPOINT=https://hf-mirror.com timeout 1500 "$ROOT/external_baselines/segllm-venv/bin/python" "$ROOT/code/cmllm/external_baselines/segllm/run_smoke.py" --checkpoint "$ROOT/shared/models/segllm_095e0637/all_data_checkpoint" --clip-path "$ROOT/shared/models/clip-vit-large-patch14" --prepared "$ROOT/research_log/cycle015/prepared/preparation_receipt.json" --output-dir "$ROOT/research_log/cycle017/smoke"
```

Validation consists of the real pinned model load/equality assertions, four real native forwards with injection/order/export assertions, and an independent saved-mask shape/binary/hash check. Three changed Python files also parse. No duplicate successful inference or unrelated test expansion. Reproduction instructions and receipts are provided; an independent repeat-run reproducibility study was not performed because the scope was exactly four forwards.

**Exactly one next recommendation:** full frozen val50 clean + target15_b SegLLM comparison, preserving this prompt/interface, exact gamma compatibility restoration and unchanged CMF scorer. Await the next explicit review before running it. val50 remains reconstructed development/compatibility evidence, not final generalization evidence.
