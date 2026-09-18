# Cycle015 — pinned SegLLM setup and native preprocessing; smoke incomplete

Review: `b082c95`, CHATGPT REVIEW 014. One-hour window: 2026-09-18 13:53:13–14:53:13 +08. Closed on the next heartbeat after the deadline. **Zero model loads, zero model forwards, zero exported predictions, no scores.** Full val50 is not recommended or authorized.

## What actually passed

- Separate Python 3.10 environment, Torch 2.3.1/cu121 and torchvision 0.18.1; CMA environment/model code untouched. Exact installed packages: `cycle015/environment_freeze.txt`.
- Pinned SegLLM source `4593a069f09628ce3a5b46e657f5417fefd7be46`. Native detectron2 and both deformable extensions built/imported. Official float/double CUDA forward checks passed (maximum absolute errors 4.66e-10 and 8.67e-19). These are operator checks, not model results.
- All three all_data_checkpoint shards from HF revision `095e0637fcba015a02c0686f67b848c79c3cc80b` downloaded and SHA256-verified at 14:46:30 (+08). Total shard bytes: 14,603,240,632. Receipt: `cycle015/checkpoint_acquisition.txt`. Official HIPIE r50_parts ancillary weights also hash-verified; sources/revisions recorded separately.
- Native inference imports passed after restoring the missing `hipie/data/datasets` dependency directory from official HIPIE commit `5cc2636427fa72b175e4639b1c063efa6cc6f642`. This is a documented source addition, not an untouched checkout claim. No tracked SegLLM model files were replaced. Per-file hashes and provenance: `cycle015/hipie_dependency_restore.json`.
- Actual native memory preprocessing passed for clean/A, clean/B, target15_b/A and target15_b/B. A/B main pixels are identical; supplied miner state changes appearance/bbox; degradation changes appearance while supplied geometry stays fixed. Original 720×1280 becomes native 576×1024 plus bottom padding to 1024². Native clipped masked RGB crop, top-left square padding, CLIP preprocessing and online yxyx/1024 box convention are recorded in `cycle015/native_preprocessing.json`.

## Remaining blocker and engineering failures

The wiring smoke was **not completed within the allotted hour**. Setup consumed most of the window, and execution did not resume between checkpoint completion at 14:46 and the deadline. The next heartbeat checked at 15:01. This is incomplete execution/time-budget exhaustion, **not evidence that the downloaded checkpoint cannot load**. No remaining reproducible model-load error has been observed because loading has not been attempted. No Cycle015 process remained at closeout.

Observed setup failures and minimal repairs: server source clone HTTP2 EOF (transfer exact locally cloned revision); missing cusparse headers (installed NVIDIA wheel include directories); broken isolated libcudart linker symlink (cached matching CUDA 12.1 runtime); stalled SAM/coco git acquisition (exact official commit archives); missing Cython for coco build (official pin); missing HIPIE datasets package (official dependency directory restored as above). Official old Google Drive ancillary link failed; official HIPIE README's author HF release was pinned instead. No checkpoint, segmentator, prompt or threshold sweep occurred.

## Code and unverified path

`cmllm_remote/external_baselines/segllm/` contains environment scripts, native loader, common-image preparation, native memory seeding/check, and a **draft, unexecuted** four-trial runner. The runner's intended seam is native history-only first turn plus `[REF:1]`, zero-based historical state 0, then unchanged native inference. The fixed query is `Segment the mining helmet worn by instance 1.[REF:1]`. No helmet targets enter these input scripts. Source reading suggests native predicted-score mask selection and final-round output; actual history-index selection, injected model tensors, native model forward and output-to-original mapping are **unverified**. Syntax checks do not establish correctness of that path. There is no exported CMF record.

## Resume commands (await a new explicit review; not executed)

From the pinned remote source checkout, with `ROOT` the existing project root:

```bash
export PYTHONPATH="$ROOT/external_baselines/segllm"
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT/external_baselines/segllm"
"$ROOT/external_baselines/segllm-venv/bin/python" "$ROOT/code/cmllm/external_baselines/segllm/load_native.py" --checkpoint "$ROOT/shared/models/segllm_095e0637/all_data_checkpoint" --clip-path "$ROOT/shared/models/clip-vit-large-patch14" --output-dir "$ROOT/research_log/cycle015/load"
```

Only after native loading passes, the existing `run_smoke.py` accepts the same checkpoint/clip options plus `--prepared "$ROOT/research_log/cycle015/prepared/preparation_receipt.json" --output-dir "$ROOT/research_log/cycle015/smoke"`. Its checks must establish the actual rendered prompt/history index before its four forwards. Request review of the dependency restoration and unfinished port; any continuation should remain the same four-trial wiring task, not full val50.

GPU target: A6000 GPU0. Inference time and peak inference VRAM are unavailable, not zero-valued measurements. Acquisition completed 53m17s after cycle start; this includes overlapping setup/download, not GPU inference time. Raw weights/images remain on the server; only compact code/provenance is published.

Closeout checks: five new Python files parse; existing scorer/export regression suite **15 passed in 12.86s**. These verify syntax and existing plumbing only; no native model execution.
