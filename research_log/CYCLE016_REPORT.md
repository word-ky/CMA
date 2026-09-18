# Cycle016 — native load returns, but trained gamma parameters are not restored

Review `e74641c` accepted the Cycle015 dependency restoration and authorized the same four-trial smoke, with an explicit stop on loading failures beyond asset transport. Started 2026-09-18 15:38 +08; stopped before inference on the observed non-transport weight-loading blocker. **Two load attempts, one returned model, zero forwards, zero masks, no scores.**

## Executed load and resource evidence

1. `20260918-153806-cma-cycle016-native-load`: failed on the native BERT relative path `./uninext-segm/projects/HIPIE/bert-base-uncased`. Added only a symlink to the existing pinned ancillary BERT directory. No reinstall or download.
2. `20260918-153857-cma-cycle016-load-path`: returned a model, exit0, 19.668836s loading, peak allocated GPU memory 14,581,610,496 bytes, 7,271,583,752 parameters, tokenizer size32011. GPU NVIDIA RTX A6000; torch2.3.1+cu121; transformers4.46.1. Full log and raw loader receipt are in `cycle016/`.

The raw loader's `NATIVE_LOAD_PASSED` means only that its function returned; **the weight-fidelity acceptance failed**. Do not treat that raw status as readiness for inference.

Runtime HIPIE path `pretrained_weights/hipie/r50_parts.pth` resolves to the existing `shared/models/segllm_ancillary/r50_parts.pth`, 2,370,897,199 bytes, actual SHA256 `8c2e22a0cb2101cc1250552691b992b322be8a81c2c0a527da023b76c7d900a8`, matching Cycle015. The three SegLLM shard hashes remain the verified Cycle015 assets; no weight file changed. Source remains `4593a069f09628ce3a5b46e657f5417fefd7be46` plus the previously disclosed missing datasets directory. The native fusion source has no tracked diff.

## Exact blocker

Final `from_pretrained` warns that these parameters are newly initialized rather than loaded:

```
model.segmentator.hipie.detr.detr.transformer.encoder.vl_layers.0.b_attn.gamma_l
model.segmentator.hipie.detr.detr.transformer.encoder.vl_layers.0.b_attn.gamma_v
```

The corresponding `weight_l` and `weight_v` names are reported unused. Read-only safetensors inspection shows the checkpoint actually contains **gamma_l [768] and gamma_v [256], F32**, in shard3. The pinned native code declares and uses gamma_l/gamma_v in its residual fusion operation. Installed Transformers4.46.1 `modeling_utils.py` performs a substring replacement `gamma -> weight` both in key comparison and state loading (lines715–717 and4384–4388; excerpts saved). This explains the apparent key mismatch: it is a loader compatibility collision, **not evidence that the checkpoint lacks these trained parameters or uses a different architecture**.

No key remapping, manual tensor assignment, dependency change or model source patch was performed. The raw model cannot be accepted merely because loading returned while these trained parameters were omitted. Review015's stop condition prohibits repairs beyond transport in this cycle.

Earlier ancillary HIPIE loading separately reports a 900×256 versus100×256 `tgt_embed.weight` shape mismatch and numerous unused mask_dino keys. The outer SegLLM load then reports only the two gamma parameters missing; do not conflate the ancillary-stage warning with a proven final uninitialized target embedding. Both stages' complete messages are preserved in `cycle016/native_load_log.txt`.

## Reproduce

From the pinned SegLLM source checkout, `ROOT` is the existing A6000 recovery project:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/external_baselines/segllm" HF_ENDPOINT=https://hf-mirror.com timeout 600 "$ROOT/external_baselines/segllm-venv/bin/python" "$ROOT/code/cmllm/external_baselines/segllm/load_native.py" --checkpoint "$ROOT/shared/models/segllm_095e0637/all_data_checkpoint" --clip-path "$ROOT/shared/models/clip-vit-large-patch14" --output-dir "$ROOT/research_log/cycle016/load"
```

`cycle016/audit_weight_metadata.py` performs the separate read-only safetensors/path/hash inspection; it does not instantiate a model. `weight_audit.json`, `transformers_gamma_rename.txt` and `native_gamma_usage.txt` provide the diagnosis.

## Scope and handoff

No four-trial runner executed. Runtime history index, injection hashes, final relational output order and original-resolution export remain unverified. No scorer, full val50, model adaptation, threshold/prompt change or second baseline. Existing preprocessing evidence remains Cycle015 evidence. No tests were rerun because this cycle changed no model/runner code; the actual load and metadata inspection are the relevant checks.

Await review of the **specific gamma-name loader collision**. A future authorized repair would need to preserve exact pinned trained tensors and verify their equality after loading before resuming the unchanged four-trial task. This is a proposed next decision, not an implemented workaround or permission to run full val50.
