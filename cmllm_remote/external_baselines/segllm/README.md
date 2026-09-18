# Pinned SegLLM supplied-memory port (Cycle015)

Status: one-hour cycle closed incomplete. Native ops/imports, checkpoint hashes and memory preprocessing passed; model loader and four-trial runner are unexecuted drafts. Zero predictions or model scores. See `research_log/CYCLE015_REPORT.md`.

Only upstream `berkeley-hipie/segllm` revision
`4593a069f09628ce3a5b46e657f5417fefd7be46` and `Marlo-Z/SegLLM`
revision `095e0637fcba015a02c0686f67b848c79c3cc80b`, subfolder
`all_data_checkpoint`, are in scope. The source checkout and Python3.10 venv
live separately under the remote project's `external_baselines/`; upstream
code and model files are not vendored here. CMA model code is unchanged.

Fixed prompt, before any inference:
`Segment the mining helmet worn by instance 1.[REF:1]`.
Fixed group: `cf_c9029d27c0483181`; clean/A, clean/B, target15_b/A,
target15_b/B only, seed0. See `research_log/CHATGPT_REVIEW_014.md`.

## Reuse map from the pinned source

| Responsibility | Native owner | Intended port operation |
|---|---|---|
| Model and tokenizer loading | `llava/train/inference_cli.py::main` | Reuse model, tokenizer and native segmentator configuration; no replacement of HIPIE. |
| Source image and mask resize/pad | `hipie_utils.py::Preprocessor`, `ResizeLongestSide(1024)` | Use native processor on condition-matched RGB and supplied miner mask. Keep original and native input H/W. |
| Historical appearance | `inference_cli.py::inference`, crop construction after model output | Seed the same masked RGB crop, top-left square padding and native CLIP preprocessing from supplied miner geometry instead of predicted first-round output. |
| Historical bbox | Same native online inference function | Scale supplied bbox into native image coordinates; preserve native integer clipping and `[y0,x0,y1,x1]/1024`. Training loader uses a different ordering; do not silently substitute it. |
| History turn / reference indexing | `build_conversation` and `train.py` inference dataset | History text only, no predicted-miner round. Native human `ind=[1]` becomes zero-based `mask_encode_ref=[0]`; verify at runtime rather than relying solely on this source reading. |
| State injection | `inference_cli.py::inference` replacement loop | `all_mask_encode_torch[0]` and `all_box_encode_torch[0]` replace matching native entries. Image and text remain fixed across A/B. |
| Native output choice | `hipie_utils.py` inference branch | Scores for label0 set to -1, then native predicted-score argmax; CLI takes final round mask. No helmet GT in this branch's selection. Dummy inference target placeholders must never be replaced by actual helmet targets. |
| Original-resolution export | Final mask and native `mask_data.input_size` | Remove right/bottom native padding, nearest-neighbor binary resize once to frozen original H/W. Requires runtime shape check before CMF export. |

This is a source-derived map, **not runtime proof**. Actual geometry, history
index, tensor hash, prompt, mask and resource receipts are required before
claiming the four-trial smoke passed. The draft adapter has not been verified against a loaded
baseline. The one-hour boundary is approximately 2026-09-18 14:53 +08:00.

Bootstrap command on the remote project:

```bash
bash cmllm_remote/external_baselines/segllm/setup_cycle015.sh "$PROJECT_ROOT"
```

The script uses requirements.txt's torch2.3.1/torchvision0.18.1 versions and
the documented editable HIPIE install. Source checkout must finish first.
Its CUDA compiler location is a machine-specific receipt, not a portable
installation claim. Additional native dependencies and both deformable-op
builds must pass before model loading. Download scripts and acquisition
receipts are in `research_log/cycle015/`.
